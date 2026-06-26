import os
import traceback
from pathlib import Path

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.adapters.audio_decode_ffmpeg import AudioDecodeError
from src.config import OUT_DIR, TMP_DIR
from src.infra.io_utils import write_json, write_srt_blocks
from src.infra.schema import build_response_dto
from src.process import make_pipeline
from src.usecases.transcribe import transcribe_audio


def main():
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/")
    def root():
        return {"status": "ok"}

    @app.post("/stt-speaker")
    async def stt_speaker(diarize: str = Form("true"), return_srt: str = Form("false"), file: UploadFile = File(...)):
        diarize_enabled = (diarize or "").strip().lower() == "true"
        return_srt_flag = (return_srt or "").strip().lower() == "true"

        suffix = Path(file.filename or "").suffix or ".bin"
        tmp_in = TMP_DIR / f"upload_{os.getpid()}_{id(file)}{suffix}"
        with open(tmp_in, "wb") as f:
            f.write(await file.read())

        try:
            stt, diar, cfg = make_pipeline(diarize_enabled)
            result = transcribe_audio(tmp_in, cfg, stt, diar)
            dto = build_response_dto(result)

            if return_srt_flag:
                srt_path = OUT_DIR / f"{tmp_in.stem}.srt"
                write_srt_blocks(srt_path, dto["blocks"])
                dto["srt"] = srt_path.read_text(encoding="utf-8")

            raw_path = OUT_DIR / f"{tmp_in.stem}.json"
            write_json(raw_path, {"result": dto})

            return JSONResponse(content={"result": dto})

        except AudioDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Ошибка декодирования: {e}")
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {e}\n{tb}")
        finally:
            try:
                tmp_in.unlink(missing_ok=True)
            except Exception:
                pass

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")


if __name__ == "__main__":
    main()
