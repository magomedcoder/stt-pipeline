import re
from typing import Optional, List, Dict, Any

from src.domain.entities import TranscriptResult, Utterance, Word


def speaker_str_to_int(s: Optional[str]) -> int:
    if not s:
        return 0

    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))

    try:
        return int(s)
    except (TypeError, ValueError):
        return 0


def _collect_words(result: TranscriptResult) -> List[Word]:
    words: List[Word] = []
    for u in result.utterances or []:
        if u.words:
            words.extend(u.words)

    words.sort(key=lambda w: (w.start, w.end))
    return words


def _block_from_utterance(u: Utterance) -> Dict[str, Any]:
    spk = speaker_str_to_int(u.speaker)
    if u.words:
        start = u.words[0].start
        end = u.words[-1].end
        text = " ".join(w.text for w in u.words)
    else:
        start = u.start
        end = u.end
        text = (u.text or "").strip()

    return {
        "start": float(start),
        "end": float(end),
        "spk": spk,
        "text": text,
    }


def build_response_dto(result: TranscriptResult) -> Dict[str, Any]:
    words = _collect_words(result)
    words_json = [{
        "word": w.text,
        "start": float(w.start),
        "end": float(w.end),
    } for w in words]

    speakers: List[Dict[str, Any]] = []
    blocks: List[Dict[str, Any]] = []
    for u in result.utterances or []:
        spk = speaker_str_to_int(u.speaker)
        start = float(u.start if u.start is not None else (u.words[0].start if u.words else 0.0))
        end = float(u.end if u.end is not None else (u.words[-1].end if u.words else 0.0))
        speakers.append({"start": start, "end": end, "spk": spk})
        blocks.append(_block_from_utterance(u))

    if words_json:
        full_text = " ".join(w["word"] for w in words_json)
    else:
        full_text = " ".join(b["text"] for b in blocks)

    return {
        "text": full_text.strip(),
        "words": words_json,
        "speakers": speakers,
        "blocks": blocks,
    }
