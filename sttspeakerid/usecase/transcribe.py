from pathlib import Path
from typing import Optional

from sttspeakerid.adapters.audio_decode_ffmpeg import temp_wav_16k
from sttspeakerid.config import PipelineConfig, validate_pipeline_config
from sttspeakerid.domain.entities import TranscriptResult
from sttspeakerid.domain.ports import STTEngine, Diarizer
from sttspeakerid.infra.io_utils import write_json, write_srt_blocks
from sttspeakerid.infra.schema import build_response_dto


def transcribe_audio(
        audio_path: Path,
        cfg: PipelineConfig,
        stt: STTEngine,
        diarizer: Optional[Diarizer] = None,
) -> TranscriptResult:
    """
        Декодирует аудио, выполняет STT и (опционально) диаризацию на одном WAV 16 kHz, чтобы таймкоды слов и сегментов спикеров совпадали.
    """
    with temp_wav_16k(audio_path, sr=cfg.vosk.sample_rate) as wav_path:
        res = stt.transcribe(wav_path)
        if cfg.diarization.enabled and diarizer is not None:
            res = diarizer.assign_speakers(res, wav_path)
    return res


def transcribe_file(
        audio_path: Path,
        cfg: PipelineConfig,
        stt: STTEngine,
        diarizer: Optional[Diarizer] = None,
) -> TranscriptResult:
    """
        1) Декодируем вход в WAV mono 16k (временный файл)
        2) STT (Vosk)
        3) Диаризация (опционально)
        4) Сохранение JSON/SRT в целевом формате
    """
    validate_pipeline_config(cfg)
    res = transcribe_audio(audio_path, cfg, stt, diarizer)

    schema = build_response_dto(res)
    stem = audio_path.stem
    out_dir = cfg.output.out_dir

    if cfg.output.save_json:
        write_json(out_dir / f"{stem}.json", schema)

    if cfg.output.save_srt:
        write_srt_blocks(out_dir / f"{stem}.srt", schema["blocks"])

    return res
