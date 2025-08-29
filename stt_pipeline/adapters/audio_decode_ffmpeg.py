import subprocess, tempfile, wave, io

from contextlib import contextmanager
from pathlib import Path

class AudioDecodeError(RuntimeError): ...

def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
        raise AudioDecodeError(err) from e

def _tool_exists(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def _ffmpeg_exists() -> bool:
    return _tool_exists(["ffmpeg", "-version"])

def _mpg123_exists() -> bool:
    return _tool_exists(["mpg123", "--version"])

def _is_wav_mono_pcm16_sr(p: Path, sr: int) -> bool:
    try:
        with wave.open(str(p), "rb") as wf:
            return wf.getnchannels() == 1 and wf.getsampwidth() == 2 and wf.getframerate() == sr
    except wave.Error:
        return False

def _read_magic(p: Path) -> bytes:
    with open(p, "rb") as f: return f.read(12)

def _looks_like_mp3(p: Path) -> bool:
    m = _read_magic(p)
    return m.startswith(b"ID3") or (len(m) >= 2 and m[0] == 0xFF and (m[1] & 0xE0) == 0xE0)

def _ffmpeg_decode_generic(inp: Path, sr: int, out_wav: Path) -> None:
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(inp), "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(out_wav)])

def _ffmpeg_decode_mp3_hard(inp: Path, sr: int, out_wav: Path) -> None:
    _run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "warning", "-fflags", "+discardcorrupt", "-err_detect", "ignore_err", "-probesize", "200M", "-analyzeduration", "200M", "-f", "mp3", "-i", str(inp), "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(sr), "-af", "aresample=async=1:first_pts=0", "-f", "wav", str(out_wav)])

def _ffmpeg_remux_mp3(inp: Path) -> Path:
    fixed = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name)
    try:
        _run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "warning", "-fflags", "+discardcorrupt", "-err_detect", "ignore_err", "-probesize", "200M", "-analyzeduration", "200M", "-i", str(inp), "-map", "0:a:0", "-c:a", "libmp3lame", "-q:a", "2", str(fixed)])
        return fixed
    except Exception:
        try:
            fixed.unlink(missing_ok=True)
        except Exception:
            pass
        raise

def _mpg123_to_wav16k(inp: Path, sr: int) -> Path:
    if not _mpg123_exists():
        raise AudioDecodeError("mpg123 недоступен")

    raw = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    out = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    try:
        _run(["mpg123", "--quiet", "--ignore-crc", "--resync-limit", "1000000", "-w", str(raw), str(inp)])
        _run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "warning", "-y", "-i", str(raw), "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(sr), "-f", "wav", str(out)])
        return out
    finally:
        try:
            raw.unlink(missing_ok=True)
        except Exception:
            pass

def _ffmpeg_decode_window(inp: Path, start: float, dur: float, sr: int, out_wav: Path) -> None:
    _run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "warning", "-ss", str(start), "-t", str(dur), "-err_detect", "ignore_err", "-fflags", "+discardcorrupt", "-i", str(inp), "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(sr), "-af", "aresample=async=1:first_pts=0", "-f", "wav", str(out_wav)])

def _concat_wavs_mono16(out_path: Path, part_paths: list[Path], sr: int) -> None:
    frames = io.BytesIO()
    total_samples = 0
    for pp in part_paths:
        with wave.open(str(pp), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                continue

            if wf.getframerate() != sr:
                continue

            data = wf.readframes(wf.getnframes())
            if not data:
                continue

            frames.write(data)
            total_samples += wf.getnframes()

    frames_bytes = frames.getvalue()
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(frames_bytes)

def _salvage_mp3_segmented(inp: Path, sr: int, win_sec: float = 30.0, max_minutes: int = 180) -> Path:
    parts: list[Path] = []
    try:
        dur_guess = max_minutes * 60
        try:
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(inp)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            dur_str = probe.stdout.decode("utf-8", "ignore").strip()
            dur_guess = int(float(dur_str)) if dur_str else dur_guess
        except Exception:
            pass

        start = 0.0
        while start < dur_guess:
            tmp = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
            try:
                _ffmpeg_decode_window(inp, start, win_sec, sr, tmp)
                ok = _is_wav_mono_pcm16_sr(tmp, sr)
                if ok:
                    with wave.open(str(tmp), "rb") as wf:
                        if wf.getnframes() > 0:
                            parts.append(tmp)
                            start += win_sec
                            continue

                tmp.unlink(missing_ok=True)
            except AudioDecodeError:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

            start += win_sec
        if not parts:
            raise AudioDecodeError("не удалось спасти ни одного окна")
        out = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
        _concat_wavs_mono16(out, parts, sr)
        return out
    finally:
        for p in parts:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

def _decode_to_wav16k(inp: Path, sr: int) -> Path:
    if not _ffmpeg_exists():
        raise AudioDecodeError("ffmpeg не найден")
    
    out = Path(tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name)
    try:
        _ffmpeg_decode_generic(inp, sr, out)
        return out
    except AudioDecodeError as e1:
        if inp.suffix.lower() == ".mp3" or _looks_like_mp3(inp):
            try:
                _ffmpeg_decode_mp3_hard(inp, sr, out)
                return out
            except AudioDecodeError as e2:

                try:
                    fixed = _ffmpeg_remux_mp3(inp)
                    try:
                        _ffmpeg_decode_generic(fixed, sr, out)
                        return out
                    finally:
                        try:
                            fixed.unlink(missing_ok=True)
                        except Exception:
                            pass
                except AudioDecodeError as e3:

                    try:
                        return _mpg123_to_wav16k(inp, sr)
                    except AudioDecodeError as e4:

                        try:
                            return _salvage_mp3_segmented(inp, sr, win_sec=30.0, max_minutes=180)
                        except AudioDecodeError as e5:
                            raise AudioDecodeError(
                                "MP3 сильно повреждён. Шаги: ffmpeg -> ffmpeg(mp3) -> ремультиплекс -> mpg123 -> сегментное спасение"
                                ""f"Ошибки:\n"f"1) {e1}\n\n2) {e2}\n\n3) {e3}\n\n4) {e4}\n\n5) {e5}"
                            )
        raise AudioDecodeError(f"ffmpeg decode failed: {e1}")

@contextmanager
def temp_wav_16k(input_path: Path, sr: int = 16000):
    input_path = Path(input_path)
    tmp = None
    try:
        if input_path.suffix.lower() == ".wav" and _is_wav_mono_pcm16_sr(input_path, sr):
            yield input_path
        else:
            tmp = _decode_to_wav16k(input_path, sr)
            yield tmp
    finally:
        if tmp:
            try:
                Path(tmp).unlink(missing_ok=True)
            except Exception:
                pass
