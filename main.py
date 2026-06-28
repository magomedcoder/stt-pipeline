from pathlib import Path

from src.process import local_process_one


def main():
    input_files = Path("./files/audio")
    audio_exts = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}

    if not input_files.exists():
        print(f"Положите в {input_files} аудиофайлы и перезапустите скрипт")
        return

    if input_files.is_file():
        if input_files.suffix.lower() in audio_exts:
            try:
                local_process_one(input_files)
            except Exception as e:
                print(f"[FAIL] {input_files.name}: {e}")
        else:
            print(f"Файл {input_files} не является поддерживаемым аудио")
        return

    any_found = False
    for p in sorted(input_files.iterdir()):
        if p.is_file() and p.suffix.lower() in audio_exts:
            any_found = True
            try:
                local_process_one(p)
            except Exception as e:
                print(f"[SKIP] {p.name}: {e}")

    if not any_found:
        print(f"В {input_files} не найдено поддерживаемых аудиофайлов")
        return


if __name__ == "__main__":
    main()
