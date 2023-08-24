import json
import pathlib
from huggingface_hub import snapshot_download
import os
from .omograph_model import OmographModel
from .accent_model import AccentModel
import re

class RUAccent:
    def __init__(self):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        self.workdir = str(pathlib.Path(__file__).resolve().parent)

    def load(self, omograph_model_size='medium', dict_load_startup=False, repo="TeraTTS/accentuator"):
        if not os.path.exists(self.workdir + '/dictionary') or not os.path.exists(self.workdir + '/nn'):
            snapshot_download(repo_id=repo, ignore_patterns=["*.md", '*.gitattributes'], local_dir=self.workdir)
        self.omographs = json.load(open(self.workdir + '/dictionary/omographs.json'))
        self.yo_words = json.load(open(self.workdir + '/dictionary/yo_words.json'))
        self.dict_load_startup = dict_load_startup
        if dict_load_startup:
            self.accents = json.load(open(self.workdir + '/dictionary/accents.json'))
        if omograph_model_size not in ['small', 'medium']:
            raise NotImplementedError 
        self.omograph_model.load(self.workdir + f'/nn/nn_omograph/{omograph_model_size}/')
        self.accent_model.load(self.workdir + '/nn/nn_accent/')

    def split_by_words(self, text):
        text = text.lower()
        spec_chars = '!"#$%&\'()*,-./:;<=>?@[\\]^_`{|}~\r\n\xa0«»\t—…'
        text = re.sub('[' + spec_chars + ']', ' ', text)
        text = re.sub(' +', ' ', text)
        output = text.split()
        return output
    
    def extract_initial_letters(self, text):
        words = self.split_by_words(text)
        initial_letters = []
        for word in words:
            if len(word) > 2:
                initial_letters.append(word[0])

        return initial_letters
    
    def load_dict(self, text):
        chars = self.extract_initial_letters(text)
        out_dict = {}
        for char in chars:
            out_dict.update(json.load(open(f'{self.workdir}/dictionary/letter_accent/{char}.json')))
        return out_dict

    def process_punc(self, original_text, processed_text):
        original_text = self.split_by_words(original_text)
        processed_text = self.split_by_words(processed_text)
        for i, word_to_process in enumerate(original_text):
            spec_chars = 'абвгдеёжзийклмнопрстухфцчшщъыьэюя'
            word_to_append = re.sub('[' + spec_chars + ']', ' ', word_to_process)
            processed_text[i] = processed_text[i] + word_to_append.strip()
        return ' '.join(processed_text)

    def count_vowels(self, text):
        vowels = 'аеёиоуыэюяАЕЁИОУЫЭЮЯ'
        return sum(1 for char in text if char in vowels)

    def process_omographs(self, text):
        splitted_text = self.split_by_words(text)
        founded_omographs = []
        for i, word in enumerate(splitted_text):
            variants = self.omographs.get(word)
            if variants:
                founded_omographs.append({'word': word, 'variants': variants, 'position': i})
        for omograph in founded_omographs:
            splitted_text[omograph['position']] = f"<w>{splitted_text[omograph['position']]}</w>"
            cls = self.omograph_model.classify(' '.join(splitted_text), omograph['variants'])
            splitted_text[omograph['position']] = cls
        return ' '.join(splitted_text)

    def process_yo(self, text):
        splitted_text = self.split_by_words(text)
        for i, word in enumerate(splitted_text):
            splitted_text[i] = self.yo_words.get(word, word)
        return ' '.join(splitted_text)

    def process_accent(self, text):
        if not self.dict_load_startup:
            self.accents = self.load_dict(text)
        splitted_text = self.split_by_words(text)
        for i, word in enumerate(splitted_text):
            stressed_word = self.accents.get(word, word)
            if '+' not in stressed_word and self.count_vowels(word) > 1:
                splitted_text[i] = self.accent_model.put_accent(word)
            else:
                splitted_text[i] = stressed_word
        return ' '.join(splitted_text)

    def process_all(self, text):
        processed_text = self.process_yo(text)
        processed_text = self.process_omographs(processed_text)
        processed_text = self.process_accent(processed_text)
        return processed_text
