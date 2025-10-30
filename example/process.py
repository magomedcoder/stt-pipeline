from pathlib import Path
from pathlib import Path
from typing import Optional, List, Dict, Any

from example.config import OUT_DIR, SB_MODEL_DIR, VOSK_MODEL_DIR
from stt_pipeline import PipelineConfig, VoskConfig, DiarizationConfig, OutputConfig, VoskEngine, SB_Diarizer, transcribe_file
from stt_pipeline.config import PipelineConfig, VoskConfig, DiarizationConfig, OutputConfig
from stt_pipeline.adapters.stt_vosk import VoskEngine
from stt_pipeline.adapters.diarization_speechbrain import SB_Diarizer
from stt_pipeline.domain.entities import TranscriptResult

"""
ffmpeg -v warning -fflags +discardcorrupt -err_detect ignore_err -probesize 200M -analyzeduration 200M -i "audio.mp3" -map 0:a:0 -c:a libmp3lame -q:a 2 fixed.mp3

ffmpeg -v warning -y -i fixed.mp3 -ac 1 -ar 16000 out.wav
"""

def local_process_one(audio_path: Path, out_dir: Path):
    if not VOSK_MODEL_DIR.exists():
        raise FileNotFoundError(f"Не найдена модель Vosk")

    if not SB_MODEL_DIR.exists():
        raise FileNotFoundError(f"Не найдена модель SpeechBrain")

    cfg = PipelineConfig(
        vosk=VoskConfig(model_path=VOSK_MODEL_DIR),
        diarization=DiarizationConfig(speechbrain_model_dir=SB_MODEL_DIR),
        output=OutputConfig(out_dir=out_dir),
    )

    stt = VoskEngine(cfg.vosk)
    diar = SB_Diarizer(cfg.diarization) if cfg.diarization.enabled else None

    res = transcribe_file(audio_path, cfg, stt, diar)
    if getattr(res, "utterances", None):
        print(f"[OK] {audio_path.name}: {len(res.utterances)} сегм. пример: {res.utterances[0].text[:120]!r}")
    else:
        print(f"[EMPTY] {audio_path.name}")


def make_pipeline(diarize_enabled: bool) -> tuple[VoskEngine, Optional[SB_Diarizer], PipelineConfig]:
    if not VOSK_MODEL_DIR.exists():
        raise FileNotFoundError(f"Не найдена модель Vosk")

    if not SB_MODEL_DIR.exists():
        raise FileNotFoundError(f"Не найдена модель SpeechBrain")

    cfg = PipelineConfig(
        vosk=VoskConfig(model_path=VOSK_MODEL_DIR),
        diarization=DiarizationConfig(
            enabled=diarize_enabled,
            speechbrain_model_dir=SB_MODEL_DIR if diarize_enabled else None,
        ),
        output=OutputConfig(out_dir=OUT_DIR),
    )

    stt = VoskEngine(cfg.vosk)
    diar = SB_Diarizer(cfg.diarization) if cfg.diarization.enabled else None

    return stt, diar, cfg


def speaker_str_to_int(s: Optional[str]) -> int:
    if not s:
        return 0

    try:
        if s.upper().startswith("SPEAKER_"):
            return int(s.split("_", 1)[1])
        return int(s)
    except Exception:
        return 0


def build_response_dto(result: TranscriptResult) -> Dict[str, Any]:
    full_text = " ".join((u.text or "").strip() for u in result.utterances).strip()

    words: List[Dict[str, Any]] = []
    for u in result.utterances:
        if u.words:
            for w in u.words:
                words.append({
                    "word": w.text,
                    "start": float(w.start),
                    "end": float(w.end)
                })

    speakers: List[Dict[str, Any]] = []
    for u in result.utterances:
        spk = speaker_str_to_int(u.speaker)
        speakers.append({
            "start": float(u.start),
            "end": float(u.end),
            "spk": spk
        })

    blocks: List[Dict[str, Any]] = []
    for u in result.utterances:
        spk = speaker_str_to_int(u.speaker)
        blocks.append({
            "start": float(u.start),
            "end": float(u.end),
            "spk": spk,
            "text": (u.text or "").strip(),
        })

    return {
        "text": full_text,
        "words": words,
        "speakers": speakers,
        "blocks": blocks
    }
