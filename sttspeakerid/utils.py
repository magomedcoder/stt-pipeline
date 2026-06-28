from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from speechbrain.inference import EncoderClassifier

from sttspeakerid.domain.entities import Utterance, Word
from sttspeakerid.infra.caching import load_cached


def resample_to_16k(y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
        Простой безопасный ресемпл (линейная интерполяция), без внешних зависимостей
    """
    if sr == target_sr:
        return y, sr

    if y.ndim > 1:
        y = y.mean(axis=1)

    duration = y.shape[0] / sr
    new_len = int(round(duration * target_sr))
    if new_len <= 1 or y.size <= 1:
        return y.astype(np.float32), target_sr

    old_idx = np.linspace(0, y.shape[0] - 1, num=y.shape[0], dtype=np.float64)
    new_idx = np.linspace(0, y.shape[0] - 1, num=new_len, dtype=np.float64)
    y_new = np.interp(new_idx, old_idx, y).astype(np.float32)

    return y_new, target_sr


def frame_windows(total_samples: int, sr: int, win_sec: float, hop_sec: float) -> List[Tuple[int, int, float, float]]:
    """
        Возвращает список окон как (s_idx, e_idx, s_time, e_time)
    """
    win = max(int(round(win_sec * sr)), 1)
    hop = max(int(round(hop_sec * sr)), 1)
    idxs: List[Tuple[int, int, float, float]] = []
    s = 0
    while s < total_samples:
        e = min(s + win, total_samples)
        s_t = s / sr
        e_t = e / sr
        if e > s:
            idxs.append((s, e, s_t, e_t))

        if e == total_samples:
            break

        s += hop
    return idxs


def energy_ok(x: np.ndarray) -> bool:
    """
        Фильтруем тишину/очень слабые фрагменты
    """
    if x.size == 0:
        return False

    rms = float(np.sqrt(np.mean(np.square(x))))
    return rms > 1e-4


def dedupe_words(words: List[Word]) -> List[Word]:
    """
        Убирает подряд идущие дубликаты слов с одинаковыми таймкодами (артефакт Vosk)
    """
    if not words:
        return []

    out = [words[0]]
    for w in words[1:]:
        prev = out[-1]
        if w.text == prev.text and w.start == prev.start and w.end == prev.end:
            continue
        out.append(w)

    return out


def normalize_embeddings(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.clip(norms, 1e-8, None)


def choose_n_clusters(
        n_req: int | None,
        n_items: int,
        X: np.ndarray | None = None,
        max_speakers: int = 4,
        min_silhouette: float = 0.12,
) -> int:
    if n_items <= 0:
        return 1

    if n_req and n_req > 0:
        return min(int(n_req), n_items)

    if n_items == 1:
        return 1

    if X is None or X.shape[0] != n_items:
        return 1

    X_norm = normalize_embeddings(X)
    best_k = 1
    best_score = -1.0

    for k in range(2, min(max_speakers, n_items) + 1):
        clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = clustering.fit_predict(X_norm)
        if len(np.unique(labels)) < 2:
            continue

        score = float(silhouette_score(X_norm, labels, metric="cosine"))
        if score > best_score:
            best_score = score
            best_k = k

    if best_score < min_silhouette:
        return 1

    return best_k


def assign_word_speaker_labels(
        words: List[Word],
        chunks: List[Tuple[int, int, float, float]],
        labels: List[int],
) -> List[int]:
    """
        Назначает каждому слову ровно одного спикера по перекрытию с окнами эмбеддингов
    """
    if not words:
        return []

    window_labels = list(zip(chunks, labels))
    out: List[int] = []

    for w in words:
        scores: Dict[int, float] = {}
        for (_, _, s_t, e_t), lab in window_labels:
            overlap = min(e_t, w.end) - max(s_t, w.start)
            if overlap > 0:
                scores[lab] = scores.get(lab, 0.0) + overlap

        if scores:
            out.append(max(scores, key=scores.get))  # type: ignore[arg-type]
            continue

        mid = 0.5 * (w.start + w.end)
        _, nearest = min(
            (abs(0.5 * (s_t + e_t) - mid), lab)
            for (_, _, s_t, e_t), lab in window_labels
        )
        out.append(nearest)

    return out


def relabel_chronologically(word_labels: List[int]) -> List[int]:
    """
        Переименовывает кластеры: первый появившийся спикер -> 0, второй -> 1, ...
    """
    mapping: Dict[int, int] = {}
    out: List[int] = []
    for lab in word_labels:
        if lab not in mapping:
            mapping[lab] = len(mapping)
        out.append(mapping[lab])
    return out


def build_utterances_from_words(words: List[Word], speaker_labels: List[int]) -> List[Utterance]:
    if not words:
        return []

    speaker_labels = relabel_chronologically(speaker_labels)
    utterances: List[Utterance] = []
    cur_lab = speaker_labels[0]
    cur_words: List[Word] = [words[0]]

    for i in range(1, len(words)):
        if speaker_labels[i] == cur_lab:
            cur_words.append(words[i])
        else:
            utterances.append(_utterance_from_words(cur_lab, cur_words))
            cur_lab = speaker_labels[i]
            cur_words = [words[i]]

    utterances.append(_utterance_from_words(cur_lab, cur_words))
    return utterances


def _utterance_from_words(lab: int, words: List[Word]) -> Utterance:
    text = " ".join(w.text for w in words)
    return Utterance(
        speaker=f"SPEAKER_{lab}",
        start=words[0].start,
        end=words[-1].end,
        text=text,
        words=words,
    )


def load_classifier(model_dir: Path) -> EncoderClassifier:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = ("speechbrain_ecapa", str(model_dir.resolve()), device)
    return load_cached(
        key,
        lambda: EncoderClassifier.from_hparams(
            source=str(model_dir),
            run_opts={"device": device}
        )
    )
