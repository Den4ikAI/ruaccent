import json
import pathlib
from huggingface_hub import HfFileSystem, hf_hub_download
import os
import gzip
from os.path import join as join_path
from .omograph_model import OmographModel
from .accent_model import AccentModel
from .stress_usage_model import StressUsagePredictorModel
from .yo_homograph_model import YoHomographModel
from .text_preprocessor import TextPreprocessor
from .text_postprocessor import fix_capital
from .morph_precitor import MorphPredictor
import re



class RUAccent:
    def __init__(self):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        self.stress_usage_predictor = StressUsagePredictorModel()
        self.yo_homograph_model = YoHomographModel()
        self.fs = HfFileSystem()
        self.normalize = re.compile(r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;\"\"''(){}\[\]«»„“”\-\+]")
        self.omograph_models_paths = {'big_poetry': '/nn/nn_omograph/big_poetry', 
                                      'medium_poetry': '/nn/nn_omograph/medium_poetry', 
                                      'small_poetry': '/nn/nn_omograph/small_poetry',
                                      'turbo': '/nn/nn_omograph/turbo',
                                      'turbo2': '/nn/nn_omograph/turbo2',
                                      'turbo3': '/nn/nn_omograph/turbo3',
                                      'turbo3.1': '/nn/nn_omograph/turbo3.1',
                                      'tiny': '/nn/nn_omograph/tiny',
                                      'tiny2': '/nn/nn_omograph/tiny2',
                                      'tiny2.1': '/nn/nn_omograph/tiny2.1',

                                      }
    
        self.accentuator_paths = ['/nn/nn_accent', '/nn/nn_stress_usage_predictor', '/nn/nn_yo_homograph_resolver', '/dictionary']
        self.letters_accent = {'о': '+о', 'О': '+О'}
        self.module_path = str(pathlib.Path(__file__).resolve().parent)
        self.workdir = self.module_path
        
    def load(
        self,
        omograph_model_size="turbo2",
        use_dictionary=True,
        custom_dict={},
        custom_homographs={},
        device="CPU",
        repo="ruaccent/accentuator",
        workdir=None,
        tiny_mode=False,
        stress_homographs_only=False,  # ставить ударение только в омографах
        ):
        self.tiny_mode = tiny_mode
        if workdir:
            self.workdir = workdir
        self.stress_homographs_only = stress_homographs_only
        self.custom_dict = custom_dict
        self.accents = {}

        if not os.path.exists(
            join_path(self.workdir, "dictionary")
        ):
            for path in self.accentuator_paths:
                files = self.fs.ls(repo + path)
                for file in files:
                    if file["type"] == "file":
                        hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=self.workdir, filename=file['name'].replace(repo+'/', ''))
    
        if not os.path.exists(join_path(self.workdir, "nn")):
            os.mkdir(join_path(self.workdir, "nn"))
        
        if not os.path.exists(join_path(self.workdir, "nn", "nn_omograph", omograph_model_size)):
            model_path = self.omograph_models_paths.get(omograph_model_size, None)
            if model_path:
                files = self.fs.ls(repo + model_path)
                for file in files:
                    if file["type"] == "file":
                        hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=self.workdir, filename=file['name'].replace(repo+'/', ''))
 
        if not os.path.exists(join_path(self.workdir, "nn", "nn_rumorph")):
            assert False, "Missing nn/nn_rumorph"
 
        self.omographs = json.load(
            gzip.open(join_path(self.workdir, "dictionary","omographs.json.gz"))
        )
        self.omographs.update(custom_homographs)
        self.omograph_model.load(join_path(self.workdir, self.omograph_models_paths[omograph_model_size][1:]), device=device)

        if not self.stress_homographs_only:
            if use_dictionary:
                if self.tiny_mode:
                    self.accents.update(json.load(
                        gzip.open(join_path(self.workdir, "dictionary","accents_nn.json.gz"))
                    ))
                else:
                    self.accents.update(json.load(
                        gzip.open(join_path(self.workdir, "dictionary","accents.json.gz"))
                    ))

            self.accents.update(self.custom_dict)
            self.accents.update(self.letters_accent)

            if not self.tiny_mode:
                self.stress_usage_predictor.load(join_path(self.workdir, "nn","nn_stress_usage_predictor/"), device=device)

            self.accent_model.load(join_path(self.workdir, "nn","nn_accent/"), device=device)

        # Ёфикатор
        self.load_yo(device=device)


    def load_yo(self, device="CPU"):
        if not self.workdir:
            self.workdir = str(pathlib.Path(__file__).resolve().parent)

        # Словарь для однозначной расстановки "ё"
        self.yo_words = json.load(
            gzip.open(join_path(self.workdir, "dictionary", "yo_words.json.gz"))
        )
        # Морфологический движок
        self.morph = MorphPredictor()
        self.morph.load(join_path(self.workdir, "nn", "nn_rumorph", "deberta_onnx"), device)

        self.yo_rules = json.load(
            gzip.open(join_path(self.workdir, "dictionary", "yo_rules.json.gz"))
        )

        if not self.tiny_mode:
            # Нейросеть для расстановки "ё"
            self.yo_homographs = json.load(
                    gzip.open(join_path(self.workdir, "dictionary", "yo_homographs.json.gz"))
                ) 
            self.yo_homograph_model.load(join_path(self.workdir, "nn", "nn_yo_homograph_resolver"), device=device)


    def count_vowels(self, text):
        vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
        return sum(1 for char in text if char in vowels)

    def has_punctuation(self, text):
        for char in text:
            if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                return True
        return False

    def delete_spaces_before_punc(self, text):
        text = re.sub(r'\s*-\s*', '-', text)
        text = re.sub(r'\s+([!\"#$%&\'()*,\./:;<=>?@\[\\\]^_`{|}])', r' \1', text)
        text = re.sub(r'([!\"#$%&\'()*,\./:;<=>?@\[\\\]^_`{|}])\s+', r'\1 ', text)
        return text.replace('~', '-')

    def extract_entities(self, data):
        entities = []
        for item in data:
            entity = item['entity']
            entities.append(entity)
        return entities

    # Используется ёфикатором
    def _align_morph_tokens(self, processed_data, original_text):
        """
        Выравнивает токены и удаляет артефакты из результатов обработки текста.
        
        Args:
            processed_data (list): Список словарей с результатами обработки
            original_text (str): Исходный текст
            
        Returns:
            list: Очищенный список токенов
        """
        # Функция для проверки валидности символа
        def is_valid_char(char):
            return ord(char) < 0xD800 or ord(char) > 0xDFFF
        
        # Фильтруем пустые токены и сортируем по позиции и длине
        valid_tokens = []
        for token in processed_data:
            # Пропускаем токены с пустым словом или нулевой длиной
            if not token['word'].strip() and token['start'] == token['end']:
                continue
                
            # Проверяем границы
            if token['start'] >= len(original_text) or token['end'] > len(original_text) or token['start'] < 0:
                continue
                
            # Получаем текст токена из оригинального текста
            token_text = original_text[token['start']:token['end']]
            
            # Проверяем на наличие невалидных символов
            if all(is_valid_char(char) for char in token_text):
                # Обновляем текст токена
                token['word'] = token_text
                valid_tokens.append(token)
        
        # Сортируем токены по начальной позиции и длине (более длинные первыми)
        sorted_tokens = sorted(valid_tokens, key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        # Объединяем перекрывающиеся токены с одинаковой начальной позицией
        final_tokens = []
        current_start = None
        
        for token in sorted_tokens:
            if current_start != token['start']:
                final_tokens.append(token)
                current_start = token['start']
        
        # Сортируем финальный результат по позиции
        result = sorted(final_tokens, key=lambda x: x['start'])

        result = [(_["word"].strip(), _["entity"]) for _ in result]

        return result

    def _choose_yo_rules(self, interpretations, tag):
        resultWord = None
        resultInterpretation = None
        sum_max = 0
        missing = 0
        tag = tag.split('|', 1) # отделяем част речи от остальных атрибутов
        tags = None
        for w, variants in interpretations.items():
            for var in variants:
                vtags = var['pt'].split('|', 1)  # отделяем част речи от остальных атрибутов
                if vtags[0] != tag[0]:  # сравниваем часть речи
                    continue
                # Парсим атрибуты, чтобы получить dict
                if len(vtags) > 1:
                    vtags = dict([_.split('=') for _ in vtags[1].split('|')])
                else:
                    vtags = {}
                if tags is None:
                    if len(tag) > 1:
                        tags = dict([_.split('=') for _ in tag[1].split('|')])
                    else:
                        tags = {}
                '''
                Алгоритм следующий: берём то правило, в котором больший процент совпадающих тегов.
                Если процент одинаковый, то берёт то правило, в котором меньше "лишних" тегов.
                Пример:
                 1: ADJ|Gender=Masc|Number=Sing|Variant=Short
                 2: ADJ|Gender=Masc|Number=Sing
                Для токена "ADJ|Gender=Masc|Number=Sing" будет выбрано правило 2, потому что в нём меньше "лишних" тегов.
                Если результат в итоге получается равный - не получается различить "е" и "ё" - тогда возвращаем None.
                '''
                n = 1  # число общих тегов
                pos_sum = 1 # число равных тегов  (считаем совпадение части речи, проверенное выше, за 1)
                for attr, val in vtags.items():
                    tag_val = tags.get(attr)
                    if tag_val is not None:
                        n += 1
                        if val == tag_val:
                            pos_sum += 1
                pos_sum = pos_sum / n
                if pos_sum >= sum_max:
                    # Считаем число лишних тегов
                    miss = 0
                    for attr in vtags.keys():
                        if attr not in tags:
                            miss += 1
                    if pos_sum > sum_max or miss < missing:
                        sum_max = pos_sum
                        missing = miss
                        resultWord = w
                        resultInterpretation = var
                    else:  # тот же "вес"
                        if resultWord != w:  # равнозначные варианты - возвращаем None
                            resultInterpretation = None
        if resultInterpretation is None:
            return None, None
        return resultWord, resultInterpretation

    def _process_yo(self, words, sentence):
        lower_sentence = sentence.lower()

        if 'е' in lower_sentence:
            morph_tags = None
            yo_predictions = None
            for i, word in enumerate(words):
                lower_word = word.lower()
                # Выбираем на основе словаря (для не омографов)
                yo_word = self.yo_words.get(lower_word)

                if yo_word is None:
                    # Выбираем на основе морфологии
                    interpretations = self.yo_rules.get(word.lower())
                    if interpretations is not None:
                        if morph_tags is None:  # инициализируем при первом использовании
                            split_sentence = " ".join(words)
                            morph_tags = self._align_morph_tokens(self.morph.predict_morph(split_sentence), split_sentence)
                            assert len(morph_tags) == len(words), words
                        assert morph_tags[i][0] == word, f"'{morph_tags[i][0]}' '{word}'"
                        w, choise = self._choose_yo_rules(interpretations, morph_tags[i][1])
                        if choise:
                            yo_word = choise.get('acc', w)

                if yo_word is None:
                    # Выбираем нейросетью
                    if not self.tiny_mode:
                        yo_word = self.yo_homographs.get(lower_word)
                        if yo_word is not None:
                            if yo_predictions is None:  # инициализируем при первом использовании
                                yo_predictions = self.extract_entities(self.yo_homograph_model.predict_yo_homographs(lower_sentence))
                                assert len(yo_predictions) == len(words), yo_predictions
                            if yo_predictions and yo_predictions[i] == "YO":
                                words[i] = fix_capital(word, yo_word)

                if yo_word is not None:
                    if self.stress_homographs_only:
                        # Убираем лишние ударения
                        if "ё" in yo_word or self.count_vowels(yo_word) == 1:
                            yo_word = re.sub(r'\+', '', yo_word)
                    words[i] = fix_capital(word, yo_word)

        return words


    def _process_omographs(self, text):
        text_lower = [w.lower() for w in text]
        
        founded_omographs = []
        texts = []
        hypotheses = []
    
        for i, word in enumerate(text_lower):
            variants = self.omographs.get(word)
            if variants:
                founded_omographs.append(
                    {"word": word, "variants": variants, "position": i}
                )
                texts.append(text_lower)
                hypotheses.append(variants)

        if len(founded_omographs) > 0:
            texts_batch = []
            hypotheses_batch = [val for sublist in hypotheses for val in sublist]
            num_hypotheses = [len(i) for i in hypotheses]
            
            for o, t in zip(founded_omographs, texts):
                position = o["position"]
                t_back = t[position]
                t[position] = ' <w>' + t[position] + '</w> '
                for _ in range(len(o["variants"])):
                    texts_batch.append(self.delete_spaces_before_punc(" ".join(t.copy())))
                t[position] = t_back
            cls_batch = self.omograph_model.classify(texts_batch, hypotheses_batch, num_hypotheses)
            for cls_index, omograph in enumerate(founded_omographs):
                position = omograph["position"]
                text[position] = fix_capital(text[position], cls_batch[cls_index])
        return text


    
    def _process_accent(self, text, stress_usages):
        splitted_text = text
        for i, word in enumerate(splitted_text):
            if '+' in word:
                continue
            if stress_usages[i] == "STRESS":
                lower_word = word.lower()
                if 'ё' in lower_word:
                    # Если есть ё, то просто ставим ей ударение
                    splitted_text[i] = re.sub(r'(?<!\+)(ё)', r'+\1', word, flags=re.IGNORECASE)
                    continue
                # Сначала смотрим по словарю
                stressed_word = self.accents.get(lower_word)
                if stressed_word is not None:
                    splitted_text[i] = fix_capital(word, stressed_word)
                    continue
                if self.has_punctuation(lower_word):
                    continue
                if self.count_vowels(lower_word) == 1:
                    splitted_text[i] = re.sub(r'(?<!\+)([аеиоуыэюя])', r'+\1', word, flags=re.IGNORECASE)
                    continue
                # Используем accent_model
                stressed_word = self.accent_model.put_accent(word)
                splitted_text[i] = fix_capital(word, stressed_word)
        return splitted_text


    def process_yo(self, text):
        sentences = TextPreprocessor.split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            words, remaining_text = TextPreprocessor.split_by_words(sentence)
            processed_words = self._process_yo(words, sentence)
            processed_text = "".join([l+r for l,r in zip(remaining_text, processed_words)])
            processed_text = self.delete_spaces_before_punc(processed_text)
            outputs.append(processed_text)
        return " ".join(outputs)


    def process_all_internal(self, text):
        text = re.sub(self.normalize, "", text)
        sentences = TextPreprocessor.split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            words, remaining_text = TextPreprocessor.split_by_words(sentence)
            if len(words) == 0:
                outputs.append("".join(remaining_text))
                continue
            # Ёфикация. Также выдаёт ударения в ё-омографах.
            processed_words = self._process_yo(words, sentence)
            # Расстановка ударений в омографах
            processed_words = self._process_omographs(processed_words)

            if not self.stress_homographs_only:
                # убираем ударения, т.к. stress_usage_predictor их считает пунктуацией
                sentence = " ".join([re.sub(r'\+(?=[а-яёa-z]+)', '', w, flags=re.IGNORECASE) for w in words])

                if not self.tiny_mode:
                    stress_usages = self.extract_entities(self.stress_usage_predictor.predict_stress_usage(sentence))
                else:
                    stress_usages = ["STRESS"] * len(text)
                processed_words = self._process_accent(processed_words, stress_usages)

            processed_sentence = "".join([l+r for l,r in zip(remaining_text, processed_words)] + [remaining_text[-1]])
            processed_sentence = self.delete_spaces_before_punc(processed_sentence)
            
            outputs.append(processed_sentence)
        return "".join(outputs)

    def process_all(self, text, skip_regex=None):
        if skip_regex:
            pattern = re.compile(skip_regex)
            matches = pattern.finditer(text)

            indices = [(match.start(), match.end()) for match in matches]
            skipped = [text[l:r] for l,r in indices]
            
            if len(indices) == 0:
                return self.process_all_internal(text)

            elems = []
            for l,r in zip(indices, indices[1:]):
                start = l[1]
                end = r[0]
                elem = text[start:end]
                elems.extend([elem])

            first_elem = text[:indices[0][0]]
            last_elem = text[indices[-1][1]:]

            elems = [first_elem] + elems + [last_elem]

            results = []
            for e in elems:
                if len(e) == 0:
                    results.append(e)
                    continue
                results.append(self.process_all_internal(e))
            return "".join([results[0]] + [l+r for l,r in zip(skipped, results[1:])])
        else:
            return self.process_all_internal(text)


