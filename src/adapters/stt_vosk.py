import json, wave, contextlib

from pathlib import Path
from typing import List
from vosk import Model, KaldiRecognizer, SetLogLevel

from src.config import VoskConfig
from src.domain.entities import TranscriptResult, Utterance, Word
from src.domain.ports import STTEngine
from src.infra.caching import load_cached


SetLogLevel(-1)


def _read_mono_16k_pcm_wav(path: Path, sample_rate: int) -> wave.Wave_read:
    wf = wave.open(str(path), "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != sample_rate:
        wf.close()
        raise ValueError("Ожидается WAV: mono, 16-bit PCM, sample_rate == cfg.sample_rate")

    return wf

class VoskEngine(STTEngine):
    def __init__(self, cfg: VoskConfig):
        self.cfg = cfg
        key = ("vosk_model", str(cfg.model_path.resolve()))
        self.model: Model = load_cached(key, lambda: Model(str(cfg.model_path)))

    def transcribe(self, audio_path: Path) -> TranscriptResult:
        wf = _read_mono_16k_pcm_wav(audio_path, self.cfg.sample_rate)
        rec = KaldiRecognizer(self.model, self.cfg.sample_rate)
        if self.cfg.return_words:
            rec.SetWords(True)

        with contextlib.closing(wf):
            total_frames = wf.getnframes()
            duration_sec = total_frames / float(self.cfg.sample_rate)
            if self.cfg.max_seconds and duration_sec > self.cfg.max_seconds:
                raise ValueError(f"Длительность {duration_sec:.1f}s > max_seconds={self.cfg.max_seconds}")

            while True:
                data = wf.readframes(4000)
                if not data:
                    break

                rec.AcceptWaveform(data)

        final = json.loads(rec.FinalResult())
        words: List[Word] = []
        if self.cfg.return_words and "result" in final:
            for w in final["result"]:
                words.append(Word(
                    text=w.get("word", ""),
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    conf=float(w.get("conf", 0.0))
                ))

        text = final.get("text", "").strip()
        utt = Utterance(
            speaker=None,
            start=0.0,
            end=float(final.get("end", 0.0)) if "end" in final else duration_sec,
            text=text,
            words=words or None,
        )
        return TranscriptResult(language=None, utterances=[utt], raw=final)
