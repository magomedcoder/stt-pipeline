from src.config import INPUT_ROOT
from src.process import local_process_one


def main():
    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}

    if not INPUT_ROOT.exists():
        print(f"Положите в {INPUT_ROOT} аудиофайлы и перезапустите скрипт")
        return

    if INPUT_ROOT.is_file():
        if INPUT_ROOT.suffix.lower() in audio_exts:
            try:
                local_process_one(INPUT_ROOT)
            except Exception as e:
                print(f"[FAIL] {INPUT_ROOT.name}: {e}")
        else:
            print(f"Файл {INPUT_ROOT} не является поддерживаемым аудио")
        return

    any_found = False
    for p in sorted(INPUT_ROOT.iterdir()):
        if p.is_file() and p.suffix.lower() in audio_exts:
            any_found = True
            try:
                local_process_one(p)
            except Exception as e:
                print(f"[SKIP] {p.name}: {e}")

    if not any_found:
        print(f"В {INPUT_ROOT} не найдено поддерживаемых аудиофайлов")
        return


if __name__ == "__main__":
    main()
