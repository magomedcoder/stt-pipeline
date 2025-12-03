import numpy as np
import torch
import soundfile as sf

from pathlib import Path
from typing import List, Tuple
from sklearn.cluster import AgglomerativeClustering

from src.config import DiarizationConfig
from src.utils import words_in_span, choose_n_clusters, energy_ok, frame_windows, load_classifier, merge_segments, resample_to_16k
from src.domain.entities import TranscriptResult, Utterance
from src.domain.ports import Diarizer


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
            result.utterances = [base]
            return result

        embs_list: List[np.ndarray] = []
        if self.cfg.batch_seconds and self.cfg.batch_seconds > 0.0:
            max_sec = float(self.cfg.batch_seconds)
            cur_batch: List[torch.Tensor] = []
            cur_sec = 0.0
            for wav in tensors:
                cur_batch.append(wav)
                cur_sec += wav.shape[-1] * sec_per_sample
                if cur_sec >= max_sec:
                    batch = torch.nn.utils.rnn.pad_sequence(
                        [w.squeeze(0) for w in cur_batch],
                        batch_first=True
                    )

                    with torch.no_grad():
                        emb = self.clf.encode_batch(batch).mean(dim=1).cpu().numpy()
 
                    embs_list.append(emb)
                    cur_batch = []
                    cur_sec = 0.0

            if cur_batch:
                batch = torch.nn.utils.rnn.pad_sequence(
                    [w.squeeze(0) for w in cur_batch],
                    batch_first=True
                )
                with torch.no_grad():
                    emb = self.clf.encode_batch(batch).mean(dim=1).cpu().numpy()

                embs_list.append(emb)
            X = np.vstack(embs_list)
        else:
            X = []
            with torch.no_grad():
                for wav in tensors:
                    emb = self.clf.encode_batch(wav).mean(dim=1).squeeze().cpu().numpy()
                    X.append(emb)
            X = np.vstack(X)

        n_clusters = choose_n_clusters(self.cfg.n_speakers, X.shape[0])
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X)

        times = [(s_t, e_t) for (_, _, s_t, e_t) in chunks]
        merged = merge_segments(labels.tolist(), times, self.cfg.min_silence_merge)

        new_utts: List[Utterance] = []
        for lab, s, e in merged:
            seg_words = words_in_span(base.words or [], s, e)
            seg_text = " ".join(w.text for w in seg_words) if seg_words else base.text
            new_utts.append(Utterance(
                speaker=f"SPEAKER_{int(lab)}",
                start=s,
                end=e,
                text=seg_text,
                words=seg_words if seg_words else None
            ))

        if not new_utts:
            base.speaker = "SPEAKER_0"
            result.utterances = [base]
        else:
            result.utterances = new_utts

        return result
