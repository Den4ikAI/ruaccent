import json
import pathlib
from huggingface_hub import snapshot_download
import os
from os.path import join as join_path
from .omograph_model import OmographModel
from .accent_model import AccentModel
from .text_split import split_by_sentences
import re


class RUAccent:
    def __init__(self, workdir=None):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        if not workdir:
            self.workdir = str(pathlib.Path(__file__).resolve().parent)
        else:
            self.workdir = workdir

    def load(
        self,
        omograph_model_size="medium",
        dict_load_startup=False,
        disable_accent_dict=False,
        custom_dict={},
        custom_homographs={},
        repo="TeraTTS/accentuator",
        ):

        self.custom_dict = custom_dict
        self.accents = {}
        if not os.path.exists(
            join_path(self.workdir, "dictionary")
        ) or not os.path.exists(join_path(self.workdir, "nn")):
            snapshot_download(
                repo_id=repo,
                ignore_patterns=["*.md", "*.gitattributes"],
                local_dir=self.workdir,
                local_dir_use_symlinks=False,
            )
        self.omographs = json.load(
            open(join_path(self.workdir, "dictionary/omographs.json"), encoding='utf-8')
        )
        self.yo_omographs = json.load(
            open(join_path(self.workdir, "dictionary/yo_omographs.json"), encoding='utf-8')
        )
        self.omographs.update(self.yo_omographs)
        self.omographs.update(custom_homographs)
        self.yo_words = json.load(
            open(join_path(self.workdir, "dictionary/yo_words.json"), encoding='utf-8')
        )
        self.dict_load_startup = dict_load_startup

        if dict_load_startup:
            self.accents.update(json.load(
                open(join_path(self.workdir, "dictionary/accents.json"), encoding='utf-8')
            ))
        if disable_accent_dict:
            self.disable_accent_dict = True
        else:
            self.disable_accent_dict = False

        self.accents.update(self.custom_dict)

        if omograph_model_size not in ["small", "medium"]:
            raise NotImplementedError

        self.omograph_model.load(
            join_path(self.workdir, f"nn/nn_omograph/{omograph_model_size}/")
            #"/media/denis/042CD5B7300C3479/stress_dataset/glycine/bert/rubert_base/onnx/"
        )
        self.accent_model.load(join_path(self.workdir, "nn/nn_accent/"))


    def split_by_words(self, string):
        result = re.findall(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower())
        return [res for res in result if res]

    def extract_initial_letters(self, text):
        words = text
        initial_letters = []
        for word in words:
            if len(word) > 2 and '+' not in word and not bool(re.search('[a-zA-Z]', word)):
                initial_letters.append(word[0])
        return initial_letters

    def load_dict(self, text):
        chars = self.extract_initial_letters(text)
        out_dict = {}
        for char in chars:
            out_dict.update(
                json.load(
                    open(
                        join_path(self.workdir, f"dictionary/letter_accent/{char}.json"),
                        encoding='utf-8'
                    )
                )
            )
        out_dict.update(self.custom_dict)
        return out_dict

    def count_vowels(self, text):
        vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
        return sum(1 for char in text if char in vowels)

    def has_punctuation(self, text):
        for char in text:
            if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                return True
        return False

    def delete_spaces_before_punc(self, text):
        punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}~"
        for char in punc:
            text = text.replace(" " + char, char)
        return text

    def process_yo(self, text):
        splitted_text = text

        for i, word in enumerate(splitted_text):
            splitted_text[i] = self.yo_words.get(word, word)
        return splitted_text

    def process_omographs(self, text):
        splitted_text = text

        founded_omographs = []
        for i, word in enumerate(splitted_text):
            variants = self.omographs.get(word)
            if variants:
                founded_omographs.append(
                    {"word": word, "variants": variants, "position": i}
                )
        for omograph in founded_omographs:
            splitted_text[
                omograph["position"]
            ] = f"<w>{splitted_text[omograph['position']]}</w>"
            cls = self.omograph_model.classify(
                " ".join(splitted_text), omograph["variants"]
            )
            splitted_text[omograph["position"]] = cls
        return splitted_text

    def process_accent(self, text):
        if not self.dict_load_startup and not self.disable_accent_dict:
            self.accents = self.load_dict(text)

        splitted_text = text

        for i, word in enumerate(splitted_text):
            stressed_word = self.accents.get(word, word)
            if stressed_word == word and not self.has_punctuation(word) and self.count_vowels(word) > 1:
                splitted_text[i] = self.accent_model.put_accent(word)
            else:
                splitted_text[i] = stressed_word
        return splitted_text

    def process_all(self, text):
        sentences = split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            text = self.split_by_words(sentence)
            processed_text = self.process_yo(text)
            processed_text = self.process_omographs(processed_text)
            processed_text = self.process_accent(processed_text)
            processed_text = " ".join(processed_text)
            processed_text = self.delete_spaces_before_punc(processed_text)
            outputs.append(processed_text)
        return " ".join(outputs)
