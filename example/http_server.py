from __future__ import annotations

import os
import traceback
import uvicorn

from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from stt_pipeline.config import PipelineConfig, VoskConfig, DiarizationConfig, OutputConfig
from stt_pipeline.adapters.audio_decode_ffmpeg import temp_wav_16k, AudioDecodeError
from stt_pipeline.adapters.stt_vosk import VoskEngine
from stt_pipeline.adapters.diarization_speechbrain import SB_Diarizer
from stt_pipeline.domain.entities import TranscriptResult
from stt_pipeline.infra.io_utils import write_srt, write_json

vosk_model_dir = Path("./example/models/vosk-model-small-ru-0.22")
sb_model_dir = Path("./example/models/spkrec-ecapa-voxceleb")
out_dir = Path("./example/files/out")
tmp_dir = Path("./example/tmp")

def make_pipeline(diarize_enabled: bool) -> tuple[VoskEngine, Optional[SB_Diarizer], PipelineConfig]:
    if not vosk_model_dir.exists():
        raise FileNotFoundError(f"Не найдена модель Vosk")

    if not sb_model_dir.exists():
        raise FileNotFoundError(f"Не найдена модель SpeechBrain")

    cfg = PipelineConfig(
        vosk=VoskConfig(
            model_path=vosk_model_dir,
            sample_rate=16000,
            max_seconds=5400,
            return_words=True,
        ),
        diarization=DiarizationConfig(
            enabled=diarize_enabled,
            speechbrain_model_dir=sb_model_dir if diarize_enabled else None,
            n_speakers=0,
        ),
        output=OutputConfig(out_dir=out_dir, save_json=True, save_srt=True),
    )
    stt = VoskEngine(cfg.vosk)
    diar = SB_Diarizer(cfg.diarization) if cfg.diarization.enabled else None

    return stt, diar, cfg

def speaker_str_to_int(s: Optional[str]) -> int:
    if not s:
        return 0

    try:
        if s.upper().startswith("SPEAKER_"):
            return int(s.split("_", 1)[1])
        return int(s)
    except Exception:
        return 0

def build_response_dto(result: TranscriptResult) -> Dict[str, Any]:
    full_text = " ".join((u.text or "").strip() for u in result.utterances).strip()

    words: List[Dict[str, Any]] = []
    for u in result.utterances:
        if u.words:
            for w in u.words:
                words.append({
                    "word": w.text,
                    "start": float(w.start),
                    "end": float(w.end)
                })

    speakers: List[Dict[str, Any]] = []
    for u in result.utterances:
        spk = speaker_str_to_int(u.speaker)
        speakers.append({
            "start": float(u.start),
            "end": float(u.end),
            "spk": spk
        })

    blocks: List[Dict[str, Any]] = []
    for u in result.utterances:
        spk = speaker_str_to_int(u.speaker)
        blocks.append({
            "start": float(u.start),
            "end": float(u.end),
            "spk": spk,
            "text": (u.text or "").strip(),
        })

    return {
        "text": full_text,
        "words": words,
        "speakers": speakers,
        "blocks": blocks
    }

def main():
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
   
    @app.get("/")
    def root():
        return {"status": "ok"}

    @app.post("/stt-speaker")
    async def stt_speaker(diarize: str = Form("true"), return_srt: str = Form("false"), file: UploadFile = File(...)):
        if not vosk_model_dir.exists():
            raise HTTPException(status_code=500, detail="Vosk модель не найдена")

        diarize_enabled = (diarize or "").strip().lower() == "true"
        return_srt_flag = (return_srt or "").strip().lower() == "true"

        suffix = Path(file.filename or "").suffix or ".bin"
        tmp_in = tmp_dir / f"upload_{os.getpid()}_{id(file)}{suffix}"
        with open(tmp_in, "wb") as f:
            f.write(await file.read())

        try:
            stt, diar, cfg = make_pipeline(diarize_enabled)

            with temp_wav_16k(tmp_in, sr=cfg.vosk.sample_rate) as wav16:
                result = stt.transcribe(Path(wav16))

                if diar is not None:
                    try:
                        result = diar.assign_speakers(result, Path(wav16))
                    except Exception as e:
                        print(f"[WARN] Диаризация не удалась: {e}")

                dto = build_response_dto(result)

            dto = build_response_dto(result)

            if return_srt_flag:
                srt_items = []
                for i, b in enumerate(dto["blocks"], start=1):
                    srt_items.append({
                        "index": i,
                        "start": float(b["start"]),
                        "end": float(b["end"]),
                        "text": f"spk {b['spk']}: {b['text']}"
                    })
                srt_path = out_dir / f"{tmp_in.stem}.srt"
                write_srt(srt_path, srt_items)
                dto["srt"] = srt_path.read_text(encoding="utf-8")

            raw_path = out_dir / f"{tmp_in.stem}.json"
            write_json(raw_path, {
                "result": dto
            })

            return JSONResponse(content={"result": dto})

        except AudioDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Ошибка декодирования: {e}")
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {e}\n{tb}")
        finally:
            try:
                tmp_in.unlink(missing_ok=True)
            except Exception:
                pass

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="error",
    )

if __name__ == "__main__":
    main()
