from dataclasses import dataclass
from pathlib import Path
from typing import Optional

model_vosk_path = Path("./models/vosk-model-small-ru-0.22")
speechbrain_model_dir = Path("./models/spkrec-ecapa-voxceleb")

INPUT_ROOT = Path("./files/audio")
OUT_DIR = Path("./files/out")
TMP_DIR = Path("./files/tmp")


@dataclass(frozen=True)
class VoskConfig:
    model_path: Path = model_vosk_path
    # Целевая частота дискретизации входного WAV (Гц)
    sample_rate: int = 16000
    # Включить ли возврат списка слов с таймкодами (True -> подробный JSON)
    return_words: bool = True
    # Максимальная длительность обрабатываемого файла (сек), None = без ограничения
    max_seconds: Optional[int] = None


@dataclass(frozen=True)
class DiarizationConfig:
    speechbrain_model_dir: Optional[Path] = speechbrain_model_dir
    # Включена ли диаризация
    enabled: bool = True
    # Число спикеров:
    #   None или 0 -> автооценка (кластеризация сама подберёт)
    #   >0 -> фиксированное количество
    n_speakers: Optional[int] = None
    # Длина окна (сек) для извлечения эмбеддингов
    window_sec: float = 1.5
    # Шаг окна (сек) - насколько окна перекрываются
    hop_sec: float = 0.75
    # Сливать ли соседние сегменты одного спикера, если пауза между ними ≤ этого значения (сек)
    min_silence_merge: float = 0.4
    # Размер батча при прогоне сегментов (секунд аудио на батч)
    # 0.0 = батчинг отключён
    batch_seconds: float = 0.0


@dataclass(frozen=True)
class OutputConfig:
    # Папка для сохранения результатов (JSON, SRT)
    out_dir: Path = OUT_DIR
    # Сохранять ли "сырые" результаты распознавания в JSON
    save_json: bool = True
    # Сохранять ли субтитры в формате .srt
    save_srt: bool = True
    # Озвучивать ли результат распознавания голосом (если есть TTS)
    say_result: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    vosk: VoskConfig = VoskConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    output: OutputConfig = OutputConfig()


def validate_pipeline_config(cfg: PipelineConfig | None = None) -> None:
    cfg = cfg or PipelineConfig()

    if not cfg.vosk.model_path.exists():
        raise FileNotFoundError(f"Не найдена модель Vosk: {cfg.vosk.model_path}")

    if cfg.diarization.enabled:
        model_dir = cfg.diarization.speechbrain_model_dir
        if not model_dir or not model_dir.exists():
            raise FileNotFoundError(f"Не найдена модель SpeechBrain: {model_dir}")

    cfg.output.out_dir.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
