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
## Параметры работы

    load(omograph_model_size='big', use_dictionary=False, custom_dict={}, custom_homographs={}, load_yo_homographs_model=False)


 - На данный момент доступны две модели: **big** (рекомендуется к использованию), **medium** и **small**. 
 - Модель **big** имеет 178 миллионов параметров, **medium** 85 миллионов, а **small** 42 миллиона
 - Переменная **use_dictionary** отвечает за загрузку всего словаря (требуется больше ОЗУ), иначе все ударения расставляет нейросеть. 
 - Переменная **custom_homographs** отвечает за добавление своих омографов. Формат такой: `{'слово-омограф': ['вариант ударения 1', 'вариант ударения 2']}`. 
 - Функция **custom_dict** отвечает за добавление своих вариантов ударений в словарь. Формат такой: `{'слово': 'сл+ово с удар+ением'}`
 - Также вы можете протестировать **beta-функцию** разрешения Ё-омографов, установив `load_yo_homographs_model=True` в `load()`, а также `accentizer.process_all(text, process_yo_omographs=True)` или `accentizer.process_yo(text, process_yo_omographs=True)`.


## Пример использования
```python
from ruaccent import RUAccent

accentizer = RUAccent()
accentizer.load(omograph_model_size='big', use_dictionary=False)

text = 'на двери висит замок.'
print(accentizer.process_all(text))

text = 'ежик нашел в лесу ягоды.'
print(accentizer.process_yo(text))
```

## Датасеты

- [Датасет](https://huggingface.co/datasets/TeraTTS/nkrja_cleaned) собранный с [НКРЯ](https://ruscorpora.ru/) **Warning!!! Много поэзии!**
- [Датасет](https://huggingface.co/datasets/TeraTTS/open_accent) словосочетаний и предложений собранных со всего интернета
- [Датасет](https://huggingface.co/datasets/TeraTTS/stress_dataset_sft) использовавшийся для обучения моделей акцентуатора

Файлы моделей и словарей располагаются по [ссылке](https://huggingface.co/TeraTTS/accentuator). Мы будем признательны фидбеку на [telegram аккаунт](https://t.me/chckdskeasfsd)