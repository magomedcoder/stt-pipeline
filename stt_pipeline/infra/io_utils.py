import json

from pathlib import Path
from typing import List

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(p: Path, data) -> None:
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_srt(p: Path, items: List[dict]) -> None:
    ensure_dir(p.parent)

    def to_ts(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)

        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with p.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(f"{it['index']}\n")
            f.write(f"{to_ts(it['start'])} --> {to_ts(it['end'])}\n")
            f.write(f"{it['text']}\n\n")