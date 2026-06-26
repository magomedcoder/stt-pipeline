from pathlib import Path
from typing import Optional

from src.adapters.diarization_speechbrain import SB_Diarizer
from src.adapters.stt_vosk import VoskEngine
from src.config import PipelineConfig, DiarizationConfig, validate_pipeline_config
from src.usecases.transcribe import transcribe_file

"""
 ffmpeg -v warning -fflags +discardcorrupt -err_detect ignore_err -probesize 200M -analyzeduration 200M -i "audio.mp3" -map 0:a:0 -c:a libmp3lame -q:a 2 fixed.mp3
 ffmpeg -v warning -y -i fixed.mp3 -ac 1 -ar 16000 out.wav
"""


def local_process_one(audio_path: Path):
    cfg = PipelineConfig()
    validate_pipeline_config(cfg)

    stt = VoskEngine(cfg.vosk)
    diar = SB_Diarizer(cfg.diarization) if cfg.diarization.enabled else None

    res = transcribe_file(audio_path, cfg, stt, diar)
    if res.utterances:
        print(f"[OK] {audio_path.name}: {len(res.utterances)} сегм. пример: {res.utterances[0].text[:120]!r}")
    else:
        print(f"[EMPTY] {audio_path.name}")


def make_pipeline(diarize_enabled: bool) -> tuple[VoskEngine, Optional[SB_Diarizer], PipelineConfig]:
    cfg = PipelineConfig(diarization=DiarizationConfig(enabled=diarize_enabled))
    validate_pipeline_config(cfg)

    stt = VoskEngine(cfg.vosk)
    diar = SB_Diarizer(cfg.diarization) if cfg.diarization.enabled else None

    return stt, diar, cfg
