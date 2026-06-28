import json

from pathlib import Path
from typing import List


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(p: Path, data) -> None:
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _to_srt_ts(t: float) -> str:
    if t < 0:
        t = 0.0

    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt_blocks(path: Path, blocks: List[dict]) -> None:
    """
    blocks: [{"start": float, "end": float, "spk": int, "text": str}]
    Формат строки:
        N
        00:00:01,200 --> 00:00:03,800
        Пользователь {spk+1}: {text}
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for i, b in enumerate(blocks, start=1):
            start_ts = _to_srt_ts(float(b["start"]))
            end_ts = _to_srt_ts(float(b["end"]))
            spk = int(b.get("spk", 0))
            text = (b.get("text") or "").strip()
            line = f"Пользователь {spk + 1}: {text}" if text else f"Пользователь {spk + 1}:"

            f.write(f"{i}\n{start_ts} --> {end_ts}\n{line}\n\n")


def write_srt(p: Path, items: List[dict]) -> None:
    ensure_dir(p.parent)

    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(f"{it['index']}\n")
            f.write(f"{_to_srt_ts(it['start'])} --> {_to_srt_ts(it['end'])}\n")
            f.write(f"{it['text']}\n\n")
