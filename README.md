# RUAccent

RUAccent - это библиотека для автоматической расстановки ударений на русском языке. 

## Установка
   С помощью pip
   ```
   pip install ruaccent
   ```
   С помощью GIT
   ```
   pip install git+https://github.com/Den4ikAI/ruaccent.git
   ```
## Методы

RUAccent предоставляет следующие методы:

- `load(omograph_model_size='medium', dict_load_startup=False), disable_accent_dict=False`: Загрузка моделей и словарей. На данные момент доступны две модели: medium    (рекомендуется к использованию) и small. Переменная dict_load_startup отвечает за загрузку всего словаря (требуется больше ОЗУ), либо во время работы для необходимых слов (экономит ОЗУ, но требует быстрыq ЖД и работает медленее). Переменная disable_accent_dict отключает использование словаря (все ударения расставляет нейросеть). Данная функция экономит ОЗУ, по скорости работы сопоставима со всем словарём в ОЗУ.

- `process_all(text)`: Обрабатывает текст всем сразу (ёфикация, расстановка ударений и расстановка ударений в словах-омографах)

- `process_omographs(text)`: Расстановка ударений только в омографах.

- `process_yo(text)`: Ёфикация текста.

## Пример использования
```python
from ruaccent import RUAccent

accentizer = RUAccent()
accentizer.load(omograph_model_size='medium', dict_load_startup=False, disable_accent_dict=False)

text = 'на двери висит замок'
print(text_processor.process_all(text))

text = 'ежик нашел в лесу ягоды'
print(text_processor.process_yo(text))
```


Файлы моделей и словарей располагаются по [ссылке](https://huggingface.co/TeraTTS/accentuator). Датасеты будут скоро опубликованы. Мы будем признательны, если вы будете расширять словари и загружать их в репозиторий. Это поможет улучшать данный проект.
