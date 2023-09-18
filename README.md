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

    load(omograph_model_size='big', use_dictionary=False, custom_dict={}, custom_homographs={}


 - На данный момент доступны две модели: **big** (рекомендуется к использованию) и **small**. 
 - Модель **big** имеет 178 миллионов параметров, а small 10 миллионов
 - Переменная **use_dict** отвечает за загрузку всего словаря (требуется больше ОЗУ), иначе все ударения расставляет нейросеть. 
 - Переменная **custom_homographs** отвечает за добавление своих омографов. Формат такой: `{'слово-омограф': ['вариант ударения 1', 'вариант ударения 2']}`. 
 - Функция **custom_dict** отвечает за добавление своих вариантов ударений в словарь. Формат такой: `{'слово': 'сл+ово с удар+ением'}`



## Пример использования
```python
from ruaccent import RUAccent

accentizer = RUAccent()
accentizer.load(omograph_model_size='big', dict_load_startup=False, disable_accent_dict=False)

text = 'на двери висит замок.'
print(text_processor.process_all(text))

text = 'ежик нашел в лесу ягоды.'
print(text_processor.process_yo(text))
```

## Датасеты

- [Датасет](https://huggingface.co/datasets/TeraTTS/nkrja_cleaned) собранный с [НКРЯ](https://ruscorpora.ru/) **Warning!!! Много поэзии!**
- [Датасет](https://huggingface.co/datasets/TeraTTS/open_accent) словосочетаний и предложений собранных со всего интернета
- [Датасет](https://huggingface.co/datasets/TeraTTS/stress_dataset_sft) использовавшийся для обучения моделей акцентуатора

Файлы моделей и словарей располагаются по [ссылке](https://huggingface.co/TeraTTS/accentuator). Мы будем признательны фидбеку на [telegram аккаунт](https://t.me/chckdskeasfsd)