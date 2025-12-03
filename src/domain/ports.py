from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities import TranscriptResult


class STTEngine(ABC):
    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptResult:
        ...


class Diarizer(ABC):
    @abstractmethod
    def assign_speakers(self, result: TranscriptResult, audio_path: Path) -> TranscriptResult:
        ...
