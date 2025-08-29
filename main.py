from pathlib import Path
from stt_pipeline import PipelineConfig, VoskConfig, DiarizationConfig, OutputConfig, VoskEngine, SB_Diarizer, transcribe_file

def process_one(audio_path: Path, out_dir: Path):
    cfg = PipelineConfig(
        vosk=VoskConfig(
            model_path=Path("./models/vosk-model-small-ru-0.22"),
            sample_rate=16000,
            max_seconds=5400,
            return_words=True,
        ),
        diarization=DiarizationConfig(
            enabled=True,
            speechbrain_model_dir=Path("./models/spkrec-ecapa-voxceleb"),
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
    if res.utterances:
        print(f"[OK] {audio_path.name}: {len(res.utterances)} сегм. пример: {res.utterances[0].text[:120]!r}")
    else:
        print(f"[EMPTY] {audio_path.name}")

def main():
    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
    input_root = Path("./files/audio")
    out_dir = Path("./files/out")

    if not input_root.exists():
        print("Укажите существующий путь ./files/audio")
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
        print("В ./files/audio не найдено поддерживаемых аудиофайлов")

if __name__ == "__main__":
    main()
