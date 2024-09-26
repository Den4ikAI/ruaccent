# RUAccent

RUAccent - это библиотека для автоматической расстановки ударений на русском языке.

**По вопросам коммерческого использования пишите на [telegram аккаунт](https://t.me/bceloss)**
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

    load(omograph_model_size='turbo2', use_dictionary=True, custom_dict={}, device="CPU", workdir=None)

 - На данный момент доступно 6 моделей - **tiny**, **tiny2**, **tiny2.1**, **turbo2**, **turbo3**, **turbo3.1**, **turbo**, **big_poetry**.
 - Переменная **use_dictionary** отвечает за загрузку всего словаря (требуется больше ОЗУ), иначе все ударения расставляет нейросеть. 
 - Функция **custom_dict** отвечает за добавление своих вариантов ударений в словарь. Формат такой: `{'слово': 'сл+ово с удар+ением'}`
- Выбор устройства CPU или CUDA. **Для работы с CUDA требуется установить onnxruntime-gpu и CUDA.**
- workdir - принимает строку. Является путём, куда скачиваются модели.
- tiny_mode - принимает True или False. При True отключает руловый пайплайн и часть моделей. Также не загружается словарь ударений.

    **Для стабильной работы требуется минимум 512 мегабайт ОЗУ (модель омографов - tiny)**

## Пример использования
```python
from ruaccent import RUAccent

accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, tiny_mode=False)

text = 'на двери висит замок.'
print(accentizer.process_all(text))
```

Файлы моделей и словарей располагаются по [ссылке](https://huggingface.co/ruaccent/accentuator). Мы будем признательны фидбеку на [telegram аккаунт](https://t.me/bceloss)

## Донат
Вы можете поддержать проект деньгами. Это поможет быстрее разрабатывать более качественные новые версии. 
CloudTips: https://pay.cloudtips.ru/p/b9d86686
