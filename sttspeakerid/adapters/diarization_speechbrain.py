from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering

from sttspeakerid.config import DiarizationConfig
from sttspeakerid.domain.entities import TranscriptResult
from sttspeakerid.domain.ports import Diarizer
from sttspeakerid.utils import (
    assign_word_speaker_labels,
    build_utterances_from_words,
    choose_n_clusters,
    dedupe_words,
    energy_ok,
    frame_windows,
    load_classifier,
    normalize_embeddings,
    resample_to_16k,
)


class SB_Diarizer(Diarizer):
    def __init__(self, cfg: DiarizationConfig):
        if not cfg.speechbrain_model_dir:
            raise ValueError("Укажите DiarizationConfig.speechbrain_model_dir")

        self.cfg = cfg
        self.clf = load_classifier(cfg.speechbrain_model_dir)

    def assign_speakers(self, result: TranscriptResult, audio_path: Path) -> TranscriptResult:
        if not result.utterances:
            return result

        base = result.utterances[0]
        words = dedupe_words(base.words or [])

        if not words:
            base.speaker = "SPEAKER_0"
            result.utterances = [base]
            return result

        y, sr = sf.read(str(audio_path), always_2d=False)
        if isinstance(y, tuple):
            y = np.array(y)

        if y.ndim > 1:
            y = y.mean(axis=1)

        y = y.astype(np.float32, copy=False)
        y, sr = resample_to_16k(y, sr, 16000)

        frames = frame_windows(len(y), sr, self.cfg.window_sec, self.cfg.hop_sec)
        if not frames:
            base.speaker = "SPEAKER_0"
            base.words = words
            result.utterances = [base]
            return result

        chunks: List[Tuple[int, int, float, float]] = []
        tensors: List[torch.Tensor] = []
        sec_per_sample = 1.0 / sr

        for (s_idx, e_idx, s_t, e_t) in frames:
            seg = y[s_idx:e_idx]
            if not energy_ok(seg):
                continue

            wav = torch.from_numpy(seg).unsqueeze(0)
            tensors.append(wav)
            chunks.append((s_idx, e_idx, s_t, e_t))

        if not tensors:
            base.speaker = "SPEAKER_0"
            base.words = words
            result.utterances = [base]
            return result

        if len(tensors) == 1:
            base.speaker = "SPEAKER_0"
            base.words = words
            base.text = " ".join(w.text for w in words)
            result.utterances = [base]
            return result

        X = self._embed_segments(tensors, sec_per_sample)
        X_norm = normalize_embeddings(X)

        n_clusters = choose_n_clusters(self.cfg.n_speakers, X_norm.shape[0], X_norm)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(X_norm).tolist()

        word_labels = assign_word_speaker_labels(words, chunks, labels)
        new_utts = build_utterances_from_words(words, word_labels)

        if not new_utts:
            base.speaker = "SPEAKER_0"
            base.words = words
            result.utterances = [base]
        else:
            result.utterances = new_utts

        return result

    def _embed_segments(self, tensors: List[torch.Tensor], sec_per_sample: float) -> np.ndarray:
        embs_list: List[np.ndarray] = []
        if self.cfg.batch_seconds and self.cfg.batch_seconds > 0.0:
            max_sec = float(self.cfg.batch_seconds)
            cur_batch: List[torch.Tensor] = []
            cur_sec = 0.0
            for wav in tensors:
                cur_batch.append(wav)
                cur_sec += wav.shape[-1] * sec_per_sample
                if cur_sec >= max_sec:
                    embs_list.append(self._encode_batch(cur_batch))
                    cur_batch = []
                    cur_sec = 0.0

            if cur_batch:
                embs_list.append(self._encode_batch(cur_batch))

            return np.vstack(embs_list)

        X = []
        with torch.no_grad():
            for wav in tensors:
                emb = self.clf.encode_batch(wav).mean(dim=1).squeeze().cpu().numpy()
                X.append(emb)

        return np.vstack(X)

    def _encode_batch(self, batch_tensors: List[torch.Tensor]) -> np.ndarray:
        batch = torch.nn.utils.rnn.pad_sequence(
            [w.squeeze(0) for w in batch_tensors],
            batch_first=True,
        )
        with torch.no_grad():
            return self.clf.encode_batch(batch).mean(dim=1).cpu().numpy()
