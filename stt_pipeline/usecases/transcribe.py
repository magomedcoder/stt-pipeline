import re

from pathlib import Path
from typing import Optional, List, Dict, Any
from ..config import PipelineConfig
from ..domain.ports import STTEngine, Diarizer
from ..domain.entities import TranscriptResult, Utterance, Word
from ..infra.io_utils import ensure_dir, write_json
from ..adapters.audio_decode_ffmpeg import temp_wav_16k

def _to_srt_ts(t: float) -> str:
    if t < 0:
        t = 0.0

    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _spk_index(label: Optional[str]) -> int:
    if not label:
        return 0

    m = re.search(r"(\d+)", label)
    return int(m.group(1)) if m else 0

def _collect_words(result: TranscriptResult) -> List[Word]:
    words: List[Word] = []
    for u in result.utterances or []:
        if u.words:
            words.extend(u.words)

    if not words and result.utterances:
        base = result.utterances[0]
        if base.words:
            words = list(base.words)
 
    words.sort(key=lambda w: (w.start, w.end))
    return words

def _block_from_utterance(u: Utterance) -> Dict[str, Any]:
    spk = _spk_index(u.speaker)
    if u.words and len(u.words) > 0:
        start = u.words[0].start
        end = u.words[-1].end
        text = " ".join(w.text for w in u.words)
    else:
        start = u.start
        end = u.end
        text = (u.text or "").strip()

    return {
        "start": float(start),
        "end": float(end),
        "spk": spk,
        "text": text
    }

def _result_to_schema(result: TranscriptResult) -> Dict[str, Any]:
    words = _collect_words(result)
    words_json = [{
        "word": w.text,
        "start": float(w.start),
        "end": float(w.end)
    } for w in words]

    speakers = []
    for u in result.utterances or []:
        speakers.append({
            "start": float(u.start if u.start is not None else (u.words[0].start if u.words else 0.0)),
            "end": float(u.end if u.end is not None else (u.words[-1].end if u.words else 0.0)),
            "spk": _spk_index(u.speaker),
        })

    blocks = []
    for u in result.utterances or []:
        blocks.append(_block_from_utterance(u))

    if words:
        full_text = " ".join(w["word"] for w in words_json)
    else:
        full_text = " ".join(b["text"] for b in blocks)

    return {
        "text": full_text.strip(),
        "words": words_json,
        "speakers": speakers,
        "blocks": blocks,
    }

"""
    blocks: [{"start": float, "end": float, "spk": int, "text": str}]
    Формат строки:
        N
        00:00:01,200 --> 00:00:03,800
        Пользователь {spk+1}: {text}
"""
def _write_srt_blocks(path: Path, blocks: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for i, b in enumerate(blocks, start=1):
            start_ts = _to_srt_ts(float(b["start"]))
            end_ts = _to_srt_ts(float(b["end"]))
            spk = int(b.get("spk", 0))
            text = (b.get("text") or "").strip()
            line = f"Пользователь {spk + 1}: {text}" if text else f"Пользователь {spk + 1}:"

            f.write(f"{i}\n{start_ts} --> {end_ts}\n{line}\n\n")

"""
    1) Декодируем вход в WAV mono 16k (временный файл)
    2) STT (Vosk)
    3) Диаризация (опционально)
    4) Сохранение JSON/SRT в целевом формате
"""
def transcribe_file(
    audio_path: Path,
    cfg: PipelineConfig,
    stt: STTEngine,
    diarizer: Optional[Diarizer] = None,
) -> TranscriptResult:
    with temp_wav_16k(audio_path, sr=cfg.vosk.sample_rate) as wav_path:
        res = stt.transcribe(wav_path)

    if cfg.diarization.enabled and diarizer is not None:
        res = diarizer.assign_speakers(res, audio_path)

    ensure_dir(cfg.output.out_dir)
    stem = audio_path.stem

    schema = _result_to_schema(res)
    write_json(cfg.output.out_dir / f"{stem}.json", schema)

    _write_srt_blocks(cfg.output.out_dir / f"{stem}.srt", schema["blocks"])

    return res
