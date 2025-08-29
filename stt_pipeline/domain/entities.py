from dataclasses import dataclass
from typing import List, Optional, Any

@dataclass
class Word:
    text: str
    start: float
    end: float
    conf: float

@dataclass
class Utterance:
    speaker: Optional[str]
    start: float
    end: float
    text: str
    words: Optional[List[Word]] = None

@dataclass
class TranscriptResult:
    language: Optional[str]
    utterances: List[Utterance]
    raw: Optional[Any] = None
