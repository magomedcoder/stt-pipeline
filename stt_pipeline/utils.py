import numpy as np
import torch

from pathlib import Path
from typing import List, Tuple
from speechbrain.inference import EncoderClassifier
from .domain.entities import Word
from .infra.caching import load_cached

"""
    Простой безопасный ресемпл (линейная интерполяция), без внешних зависимостей
"""
def resample_to_16k(y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
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

"""
    Возвращает список окон как (s_idx, e_idx, s_time, e_time)
"""
def frame_windows(total_samples: int, sr: int, win_sec: float, hop_sec: float) -> List[Tuple[int, int, float, float]]:  
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

"""
    Фильтруем тишину/очень слабые фрагменты
"""
def energy_ok(x: np.ndarray) -> bool:
    if x.size == 0:
        return False

    rms = float(np.sqrt(np.mean(np.square(x))))
    return rms > 1e-4

"""
    Сливает соседние окна одного спикера, если пауза между ними <= min_silence_merge
    Возвращает список (label, start, end)
"""
def merge_segments(
    labels: List[int],
    times: List[Tuple[float, float]],
    min_silence_merge: float,
) -> List[Tuple[int, float, float]]:
    if not labels or not times:
        return []

    merged: List[Tuple[int, float, float]] = []
    cur_lab = labels[0]
    cur_s, cur_e = times[0]
    for i in range(1, len(labels)):
        lab = labels[i]
        s, e = times[i]
        gap = s - cur_e

        if lab == cur_lab and gap <= min_silence_merge:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_lab, cur_s, cur_e))
            cur_lab = lab
            cur_s, cur_e = s, e

    merged.append((cur_lab, cur_s, cur_e))
    return merged

"""
    Возвращает слова, чьи центры попадают в [s, e]
"""
def words_in_span(words: List[Word], s: float, e: float) -> List[Word]:
    out: List[Word] = []
    for w in words or []:
        mid = 0.5 * (w.start + w.end)
        if s <= mid <= e:
            out.append(w)

    return out

def choose_n_clusters(n_req: int | None, n_items: int) -> int:
    if n_req and n_req > 0:
        return int(n_req)
    return max(2, min(4, max(1, n_items)))

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