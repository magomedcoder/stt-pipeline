import os
import json
import tempfile
import subprocess
import librosa
import soundfile as sf
import numpy as np
import torch

from typing import Optional, List, Tuple
from vosk import Model, KaldiRecognizer, SetLogLevel
from speechbrain.inference.classifiers import EncoderClassifier
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path

# Целевая частота дискретизации wav для распознавания (Гц)
conf_sample_rate = 16000
# Ограничение длительности входного файла в секундах (0 - без ограничения)
conf_max_seconds = 5400
# 0 - автооценка
# > 0 - фиксированное число кластеров
conf_num_speakers = 0
# Длина окна (сек) для спикер-эмбеддингов
conf_window_sec = 1.5
# Шаг окна (сек)
conf_hop_sec = 0.75
# Склейка соседних сегментов одного спикера, если пауза между ними не превышает указанное значение сек
conf_min_silence_merge = 0.4
# Батчинг сегментов при диаризации (0 = выкл)
conf_batch_seconds = 0.0

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}

def _ffmpeg_decode_wav(path: str, sr: int) -> tuple[str, float]:
    fd, wav_path = tempfile.mkstemp(prefix="stt_", suffix=".wav")
    os.close(fd)

    try:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception:
        pass

    dur = _ffmpeg_duration(path)

    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le", wav_path]
    subprocess.run(cmd, check=True)

    if not os.path.exists(wav_path) or os.path.getsize(wav_path) <= 44:
        raise subprocess.CalledProcessError(returncode=1, cmd=" ".join(cmd), output="ffmpeg produced invalid WAV")

    return wav_path, dur

def _agglo_fit_predict(X, n_clusters: int, metric: str = "cosine", linkage: str = "average"):
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        return model.fit_predict(X)
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity=metric, linkage=linkage)
        return model.fit_predict(X)

def _ffmpeg_duration(path: str) -> Optional[float]:
    try:
        out = subprocess.check_output(
            ["ffmpeg", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()

        return float(out) if out else None
    except Exception:
        return None

def _read_wav_to_tensor(wav_path: str, target_sr: int) -> torch.Tensor:
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != target_sr:
        try:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except Exception:
            pass

    return torch.from_numpy(np.ascontiguousarray(audio))

def _run_vosk_stt(wav_path: str, sr: int) -> Tuple[str, List[dict]]:
    model_dir = "./models/vosk-model-small-ru-0.22"

    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Модель Vosk не найдено")

    SetLogLevel(-1)

    model = Model(model_dir)
    rec = KaldiRecognizer(model, sr)

    try:
        rec.SetWords(True)
    except Exception:
        pass

    text_parts: List[str] = []
    words_all: List[dict] = []

    with open(wav_path, "rb") as f:
        _ = f.read(44)
        while True:
            chunk = f.read(4000)
            if not chunk:
                break

            if rec.AcceptWaveform(chunk):
                part = json.loads(rec.Result())
                if part.get("text"):
                    text_parts.append(part["text"])

                if part.get("result"):
                    words_all.extend(part["result"])

        final = json.loads(rec.FinalResult())
        if final.get("text"):
            text_parts.append(final["text"])

        if final.get("result"):
            words_all.extend(final["result"])

    full_text = " ".join([t for t in text_parts if t]).strip()
    words_norm = []
    for w in words_all:
        if "word" in w and "start" in w and "end" in w:
            words_norm.append({
                "word": w["word"],
                "start": float(w["start"]),
                "end": float(w["end"])
            })

    return full_text, words_norm

def _speaker_diarization(wav_path: str, sr: int, window_sec: float, hop_sec: float, num_speakers: int = 0, min_merge_gap: float = 0.4, batch_seconds: float = 0.0) -> List[dict]:
    classifier = EncoderClassifier.from_hparams(
        source="./models/spkrec-ecapa-voxceleb",
        savedir="./runtime/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"} # cpu / cuda
    )

    wav = _read_wav_to_tensor(wav_path, sr)
    total_len = wav.shape[0]
    win = int(sr * window_sec)
    hop = int(sr * hop_sec)
    # (s_i, e_i, t_s, t_e)
    frames: List[Tuple[int, int, float, float]] = []
    feats: List[np.ndarray] = []

    batch_win = max(1, int((batch_seconds / hop_sec))) if batch_seconds and batch_seconds > 0 else 1

    pos = 0
    with torch.no_grad():
        while pos + win <= total_len:
            batch_segments = []
            batch_times = []

            for _ in range(batch_win):
                if pos + win > total_len:
                    break

                seg = wav[pos: pos + win]
                batch_segments.append(seg)
                t_s = pos / sr
                t_e = (pos + win) / sr
                batch_times.append((pos, pos + win, t_s, t_e))
                pos += hop

            if not batch_segments:
                break

            batch_tensor = torch.stack(batch_segments, dim=0)

            emb = classifier.encode_batch(batch_tensor)
            if emb.ndim == 3 and emb.size(1) == 1:
                emb = emb.squeeze(1)
            elif emb.ndim != 2:
                emb = emb.reshape(emb.shape[0], -1)

            emb = emb.cpu().numpy()

            for (s_i, e_i, t_s, t_e), e in zip(batch_times, emb):
                frames.append((s_i, e_i, t_s, t_e))
                feats.append(e)

    if not feats:
        return []

    X = np.stack(feats)

    if num_speakers and num_speakers > 0:
        labels = _agglo_fit_predict(X, n_clusters=int(num_speakers), metric="cosine", linkage="average")
    else:
        best_labels = None
        best_score = None

        for k in range(2, min(6, X.shape[0] + 1)):
            cand_labels = _agglo_fit_predict(X, n_clusters=k, metric="cosine", linkage="average")
            score = _cluster_compactness(X, cand_labels)
            if best_score is None or score < best_score:
                best_score = score
                best_labels = cand_labels

        labels = best_labels if best_labels is not None else _agglo_fit_predict(X, n_clusters=2, metric="cosine", linkage="average")

    segs: List[dict] = []
    cur_spk = int(labels[0])
    cur_start = frames[0][2]
    cur_end = frames[0][3]

    for i in range(1, len(frames)):
        t_s, t_e = frames[i][2], frames[i][3]
        if int(labels[i]) == cur_spk and (t_s - cur_end) <= min_merge_gap:
            cur_end = t_e
        else:
            segs.append({
                "start": float(cur_start),
                "end": float(cur_end),
                "spk": int(cur_spk),
            })
            cur_spk = int(labels[i])
            cur_start = t_s
            cur_end = t_e

    segs.append({
        "start": float(cur_start),
        "end": float(cur_end),
        "spk": int(cur_spk),
    })

    return segs

def _cluster_compactness(X: np.ndarray, labels: np.ndarray) -> float:
    score = 0.0
    for lab in np.unique(labels):
        grp = X[labels == lab]
        c = grp.mean(axis=0, keepdims=True)
        d = 1.0 - (grp @ c.T) / (np.linalg.norm(grp, axis=1, keepdims=True) * np.linalg.norm(c))
        score += float(np.mean(d))

    return score

def _assign_words_to_speakers(words: List[dict], segs: List[dict]) -> List[dict]:
    if not words:
        return []

    if not segs:
        return [{
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "spk": 0,
            "text": " ".join(w["word"] for w in words),
        }]

    def _overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        s = max(a[0], b[0]); e = min(a[1], b[1])
        return max(0.0, e - s)

    def word_spk(w):
        w_s, w_e = w["start"], w["end"]
        best_spk, best_overlap = 0, 0.0
        for s in segs:
            ov = _overlap((w_s, w_e), (s["start"], s["end"]))
            if ov > best_overlap:
                best_overlap = ov
                best_spk = s["spk"]
        return best_spk

    labeled = []
    for w in words:
        spk = word_spk(w)
        labeled.append({
            "start": w["start"],
            "end": w["end"],
            "spk": spk,
            "word": w["word"],
        })

    blocks = []
    cur_spk = labeled[0]["spk"]
    cur_start = labeled[0]["start"]
    cur_end = labeled[0]["end"]
    cur_text = [labeled[0]["word"]]

    for w in labeled[1:]:
        if w["spk"] == cur_spk and (w["start"] - cur_end) <= 0.7:
            cur_end = w["end"]
            cur_text.append(w["word"])
        else:
            blocks.append({
                "start": cur_start,
                "end": cur_end,
                "spk": cur_spk,
                "text": " ".join(cur_text),
            })
            cur_spk = w["spk"]
            cur_start = w["start"]
            cur_end = w["end"]
            cur_text = [w["word"]]

    blocks.append({
        "start": cur_start,
        "end": cur_end,
        "spk": cur_spk,
        "text": " ".join(cur_text),
    })

    return blocks

def _srt_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    ms = int((sec - int(sec)) * 1000)

    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _to_srt(blocks: List[dict]) -> str:
    lines = []
    for i, b in enumerate(blocks, 1):
        spk_name = f"Пользователь {b['spk'] + 1}"
        lines.append(f"{i}")
        lines.append(f"{_srt_time(b['start'])} --> {_srt_time(b['end'])}")
        lines.append(f"{spk_name}: {b['text']}")
        lines.append("")

    return "\n".join(lines)

def _process_one_file(input_path: Path, out_dir: Path):
    print(f"Обработка: {input_path.name}")
    wav_path = None
    try:
        wav_path, dur = _ffmpeg_decode_wav(str(input_path), conf_sample_rate)
    except FileNotFoundError:
        print("Не найден ffmpeg")
        return
    except subprocess.CalledProcessError:
        print("Не удалось перекодировать через ffmpeg")
        return

    try:
        if conf_max_seconds > 0 and dur and dur > conf_max_seconds:
            print(f"Файл слишком длинный {int(dur)} сек")
            return

        text, words = _run_vosk_stt(wav_path, conf_sample_rate)

        try:
            segments = _speaker_diarization(
                wav_path=wav_path,
                sr=conf_sample_rate,
                window_sec=float(conf_window_sec),
                hop_sec=float(conf_hop_sec),
                num_speakers=int(conf_num_speakers),
                min_merge_gap=float(conf_min_silence_merge),
                batch_seconds=float(conf_batch_seconds),
            )
        except Exception as e:
            print(f"Диаризация недоступна: {e}")
            segments = []

        blocks = _assign_words_to_speakers(words, segments) if words else []

        result_obj = {
            "text": text,
            "words": words,
            "speakers": segments,
            "blocks": blocks,
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        base = input_path.stem
        save_json_path = out_dir / f"{base}.json"
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(result_obj, f, ensure_ascii=False, indent=2)

        if blocks:
            save_srt_path = out_dir / f"{base}.srt"
            srt = _to_srt(blocks)
            with open(save_srt_path, "w", encoding="utf-8") as f:
                f.write(srt)

        if not text:
            print("Распознать не удалось или файл пуст")
        else:
            print("Готово")

    finally:
        try:
            if wav_path:
                os.unlink(wav_path)
        except Exception:
            pass

def run():
    input_root = Path("./files/audio")
    out_dir = Path("./files/out")

    if not input_root.exists():
        print("Укажите существующий путь ./files/audio")
        return

    if input_root.is_file():
        if input_root.suffix.lower() in AUDIO_EXTS:
            _process_one_file(input_root, out_dir)
        else:
            print(f"Файл {input_root} не является поддерживаемым аудио")
        return

    any_found = False
    for p in sorted(input_root.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            any_found = True
            _process_one_file(p, out_dir)

    if not any_found:
        print("В ./files/audio не найдено поддерживаемых аудиофайлов")

if __name__ == "__main__":
    run()
