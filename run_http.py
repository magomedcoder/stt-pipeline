import os
import traceback
import uvicorn

from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import OUT_DIR, TMP_DIR
from src.process import build_response_dto, make_pipeline
from src.adapters.audio_decode_ffmpeg import temp_wav_16k, AudioDecodeError
from src.infra.io_utils import write_srt, write_json


def main():
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
                srt_path = OUT_DIR / f"{tmp_in.stem}.srt"
                write_srt(srt_path, srt_items)
                dto["srt"] = srt_path.read_text(encoding="utf-8")

            raw_path = OUT_DIR / f"{tmp_in.stem}.json"
            write_json(raw_path, { "result": dto })

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

    uvicorn.run(app,host="0.0.0.0", port=8000, log_level="error")


if __name__ == "__main__":
    main()
