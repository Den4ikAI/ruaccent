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

    load(omograph_model_size='big_poetry', use_dictionary=True, custom_dict={}, custom_homographs={})


 - На данный момент доступно 6 моделей. **big** (рекомендуется к использованию), **medium** и **small**. Рекомендуются к использованию модели версии **poetry**. Их названия **big_poetry**, **medium_poetry**, **small_poetry**.
 - Модель **big** имеет 178 миллионов параметров, **medium** 85 миллионов, а **small** 42 миллиона
 - Переменная **use_dictionary** отвечает за загрузку всего словаря (требуется больше ОЗУ), иначе все ударения расставляет нейросеть. 
 - Переменная **custom_homographs** отвечает за добавление своих омографов. Формат такой: `{'слово-омограф': ['вариант ударения 1', 'вариант ударения 2']}`. 
 - Функция **custom_dict** отвечает за добавление своих вариантов ударений в словарь. Формат такой: `{'слово': 'сл+ово с удар+ением'}`


## Пример использования
```python
from ruaccent import RUAccent

accentizer = RUAccent()
accentizer.load(omograph_model_size='big_poetry', use_dictionary=True)

text = 'на двери висит замок.'
print(accentizer.process_all(text))

text = 'ежик нашел в лесу ягоды.'
print(accentizer.process_yo(text))
```

## Датасеты

- [Датасет](https://huggingface.co/datasets/TeraTTS/nkrja_raw) собранный с [НКРЯ](https://ruscorpora.ru/)
- [Датасет](https://huggingface.co/datasets/TeraTTS/stress_dataset_sft_proza) использовавшийся для обучения моделей акцентуатора (версия только с прозой)
- [Датасет](https://huggingface.co/datasets/TeraTTS/stress_dataset_sft_poetry) использовавшийся для обучения моделей акцентуатора (версия проза + поэзия)

Файлы моделей и словарей располагаются по [ссылке](https://huggingface.co/TeraTTS/accentuator). Мы будем признательны фидбеку на [telegram аккаунт](https://t.me/chckdskeasfsd)