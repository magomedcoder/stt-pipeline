# STT + Speaker Diarization

Инструмент для офлайн STT и диаризации из аудиофайлов

---

- Поддержка популярных форматов: `.wav, .mp3, .m4a, .flac, .ogg, .opus, .aac, .wma`
- Автоматическая конвертация через ffmpeg в моно-WAV 16kHz
- Распознавание текста с таймкодами по словам (Vosk)
- Диаризация - выделение сегментов речи разных говорящих (SpeechBrain + агломеративная кластеризация)
- Автоопределение числа спикеров или фиксированное значение
- Пакетная обработка: все файлы из `./example/files/audio` -> результаты в `./example/files/out`
  - `JSON` — структурированный результат (для парсинга в приложениях)
  - `SRT` — готовые субтитры с разметкой `Пользователь Н: текст`

---

#### Установка

```bash
sudo apt install ffmpeg mpg123
pip install git+https://github.com/magomedcoder/stt-pipeline.git
```

#### Запуск примера

Пакетная обработка: все файлы из ./example/files/audio автоматически обрабатываются и результаты сохраняются в ./example/files/out (JSON + SRT)

Пример запуска из корня проекта:

```bash
python3 -m venv .venv

source .venv/bin/activate

pip install -r example/requirements.txt

wget https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip

unzip vosk-model-small-ru-0.22.zip -d example/models/

git clone https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb example/models/spkrec-ecapa-voxceleb

# Запуск пакетной обработки (обработает все файлы из ./example/files/audio)
python3 -m example.main

# Альтернативный вариант - запуск http-сервера
python3 -m example.http_server
```

---

#### Формат JSON результата

```json
{
  "text": "", // распознанный текст
  "words": [
    // слова поштучно с таймкодами
    {
      "word": "", // слово
      "start": 0.0, // начало слова в секундах
      "end": 0.0 // конец слова в секундах
    }
  ],
  "speakers": [
    // куски аудио кто говорит
    {
      "start": 0.0, //  начало сегмента (сек)
      "end": 0.0, // конец сегмента (сек)
      "spk": 0 // индекс спикера: 0,1,2
    }
  ],
  "blocks": [
    // фразы собранные из words и размеченные по спикерам
    {
      "start": 0.0, // начало фразы (первое слово блока)
      "end": 0.0, // конец фразы (последнее слово блока)
      "spk": 0, // индекс спикера для блока
      "text": "" // текст фразы
    }
  ]
}
```

---

#### Формат SRT результата

```
1 // номер строки - порядковый номер блока
00:00:01,200 --> 00:00:03,800 // время - диапазон начала и конца фразы (формат чч:мм:сс,мсс)
Пользователь 1: привет мир // текст - фраза, с префиксом Пользователь N, где N = spk+1
```
