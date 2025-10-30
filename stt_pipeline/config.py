from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VoskConfig:
    # Путь к папке с распакованной моделью Vosk
    model_path: Path
    # Целевая частота дискретизации входного WAV (Гц)
    sample_rate: int = 16000
    # Включить ли возврат списка слов с таймкодами (True -> подробный JSON)
    return_words: bool = True
    # Максимальная длительность обрабатываемого файла (сек), None = без ограничения
    max_seconds: Optional[int] = None


@dataclass(frozen=True)
class DiarizationConfig:
    # Включена ли диаризация
    enabled: bool = True
    # Путь к модели SpeechBrain
    speechbrain_model_dir: Optional[Path] = None
    # Число спикеров:
    #   None или 0 -> автооценка (кластеризация сама подберёт)
    #   >0 -> фиксированное количество
    n_speakers: Optional[int] = None
    # Длина окна (сек) для извлечения эмбеддингов
    window_sec: float = 1.5
    # Шаг окна (сек) — насколько окна перекрываются
    hop_sec: float = 0.75
    # Сливать ли соседние сегменты одного спикера, если пауза между ними ≤ этого значения (сек)
    min_silence_merge: float = 0.4
    # Размер батча при прогоне сегментов (секунд аудио на батч)
    # 0.0 = батчинг отключён
    batch_seconds: float = 0.0


@dataclass(frozen=True)
class OutputConfig:
    # Папка для сохранения результатов (JSON, SRT)
    out_dir: Path
    # Сохранять ли "сырые" результаты распознавания в JSON
    save_json: bool = True
    # Сохранять ли субтитры в формате .srt
    save_srt: bool = True
    # Озвучивать ли результат распознавания голосом (если есть TTS)
    say_result: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    vosk: VoskConfig
    diarization: DiarizationConfig
    output: OutputConfig
