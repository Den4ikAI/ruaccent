# RUAccent

RUAccent - это библиотека для автоматической расстановки ударений на русском языке.

**Внимание!!! Смена лицензии на Apache 2.0**

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

    load(omograph_model_size='big_poetry', use_dictionary=True, custom_dict={})

 - На данный момент доступно 6 моделей. **big** (рекомендуется к использованию), **medium** и **small**. Рекомендуются к использованию модели версии **poetry**. Их названия **big_poetry**, **medium_poetry**, **small_poetry**.
 - Модель **big** имеет 178 миллионов параметров, **medium** 85 миллионов, а **small** 12 миллионов
 - Переменная **use_dictionary** отвечает за загрузку всего словаря (требуется больше ОЗУ), иначе все ударения расставляет нейросеть. 
 - Функция **custom_dict** отвечает за добавление своих вариантов ударений в словарь. Формат такой: `{'слово': 'сл+ово с удар+ением'}`

    **Для работы требуется 5 гигабайт ОЗУ**
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

Файлы моделей и словарей располагаются по [ссылке](https://huggingface.co/ruaccent/accentuator). Мы будем признательны фидбеку на [telegram аккаунт](https://t.me/chckdskeasfsd)