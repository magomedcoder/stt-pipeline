from pathlib import Path
from stt_pipeline import PipelineConfig, VoskConfig, DiarizationConfig, OutputConfig, VoskEngine, SB_Diarizer, transcribe_file

"""
ffmpeg -v warning -fflags +discardcorrupt -err_detect ignore_err -probesize 200M -analyzeduration 200M -i "audio.mp3" -map 0:a:0 -c:a libmp3lame -q:a 2 fixed.mp3

ffmpeg -v warning -y -i fixed.mp3 -ac 1 -ar 16000 out.wav
"""

def process_one(audio_path: Path, out_dir: Path):
    vosk_model_dir = Path("./example/models/vosk-model-small-ru-0.22")
    sb_model_dir = Path("./example/models/spkrec-ecapa-voxceleb")

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
            enabled=True,
            speechbrain_model_dir=sb_model_dir,
            n_speakers=0,
            window_sec=1.5,
            hop_sec=0.75,
            min_silence_merge=0.4,
            batch_seconds=0.0,
        ),
        output=OutputConfig(
            out_dir=out_dir,
            save_json=True,
            save_srt=True,
            say_result=False,
        ),
    )

    stt = VoskEngine(cfg.vosk)
    diar = SB_Diarizer(cfg.diarization) if cfg.diarization.enabled else None

    res = transcribe_file(audio_path, cfg, stt, diar)
    if getattr(res, "utterances", None):
        print(f"[OK] {audio_path.name}: {len(res.utterances)} сегм. пример: {res.utterances[0].text[:120]!r}")
    else:
        print(f"[EMPTY] {audio_path.name}")

def main():
    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}

    input_root = Path("./example/files/audio")
    out_dir = Path("./example/files/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        print(f"Положите в {input_root} аудиофайлы и перезапустите скрипт")
        return

    if input_root.is_file():
        if input_root.suffix.lower() in audio_exts:
            try:
                process_one(input_root, out_dir)
            except Exception as e:
                print(f"[FAIL] {input_root.name}: {e}")
        else:
            print(f"Файл {input_root} не является поддерживаемым аудио")
        return

    any_found = False
    for p in sorted(input_root.iterdir()):
        if p.is_file() and p.suffix.lower() in audio_exts:
            any_found = True
            try:
                process_one(p, out_dir)
            except Exception as e:
                print(f"[SKIP] {p.name}: {e}")

    if not any_found:
        print("В ./files/example/audio не найдено поддерживаемых аудиофайлов")

if __name__ == "__main__":
    main()
