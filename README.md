# RUAccent

RUAccent — библиотека для автоматической расстановки ударений и буквы «ё» в русском тексте.

Лицензия — MIT.

## Установка

```
pip install ruaccent
```

Свежая версия из репозитория:

```
pip install git+https://github.com/Den4ikAI/ruaccent.git
```

## Пример использования

```python
from ruaccent import RUAccent

accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True)

print(accentizer.process_all('на двери висит замок.'))
# на двер+и вис+ит зам+ок.

print(accentizer.process_yo('Все елки зеленые.'))
# Все ёлки зелёные.
```

Ударение ставится знаком `+` перед ударной гласной. Остальной текст (пробелы, переносы строк, символы, эмодзи) возвращается без изменений. Слова, в которых ударение уже стоит (`пр+ивет`), не трогаются.

## Параметры

```python
load(omograph_model_size='turbo2', use_dictionary=False, custom_dict=None,
     custom_homographs=None, device='CPU', workdir=None, tiny_mode=False)
```

- **omograph_model_size** — модель для омографов: `tiny`, `tiny2`, `tiny2.1`, `turbo`, `turbo2`, `turbo3`, `turbo3.1`, `small_poetry`, `medium_poetry`, `big_poetry`.
- **use_dictionary** — загрузить полный словарь ударений (требуется больше ОЗУ). Иначе слова, которых нет в малом словаре, размечает нейросеть.
- **custom_dict** — свои ударения: `{'слово': 'сл+ово'}`.
- **custom_homographs** — свои омографы: `{'слово': ['вариант+1', 'вар+иант2']}`.
- **device** — `CPU` (по умолчанию), `CUDA` (нужен `onnxruntime-gpu`), `CoreML` для Apple Silicon или `auto` — выбрать лучший доступный.
- **workdir** — папка для моделей. По умолчанию они скачиваются в кэш Hugging Face (`HF_HOME`). Для работы без сети после первой загрузки: `HF_HUB_OFFLINE=1`.
- **tiny_mode** — облегчённый режим: без модели выбора слов для ударения и без полного словаря.

`process_all(text, skip_regex=None)` и `process_yo(text, skip_regex=None)` не трогают фрагменты, подходящие под `skip_regex` (например, `r"<[^>]+>"` для HTML-тегов).

Для стабильной работы требуется минимум 512 МБ ОЗУ (модель омографов `tiny`).

Файлы моделей и словарей: [huggingface.co/ruaccent/accentuator](https://huggingface.co/ruaccent/accentuator). Будем признательны за фидбек в [Telegram](https://t.me/bceloss).

## Разработка

```
pip install -e .[test]
pytest                              # быстрые тесты
RUACCENT_TEST_MODELS=1 pytest       # с загрузкой моделей
```

## Донат

Вы можете поддержать проект деньгами. Это поможет быстрее разрабатывать более качественные новые версии.
CloudTips: https://pay.cloudtips.ru/p/b9d86686
