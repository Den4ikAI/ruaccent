import json
import pathlib
from huggingface_hub import HfFileSystem, hf_hub_download
import os
from os.path import join as join_path
from .omograph_model import OmographModel
from .accent_model import AccentModel
from .yo_omograph_model import YomographModel
from .text_split import split_by_sentences
import re


class RUAccent:
    def __init__(self, workdir=None):
        self.omograph_model = OmographModel()
        self.yo_omograph_model = YomographModel()
        self.accent_model = AccentModel()
        self.fs = HfFileSystem()
        self.omograph_models_paths = {'big': '/nn/nn_omograph/big', 'medium': '/nn/nn_omograph/medium', 'small': '/nn/nn_omograph/small'}
        self.accentuator_paths = ['/nn/nn_accent', '/dictionary']
        self.yo_omograph_path = ['/nn/nn_yo_omograph']
        if not workdir:
            self.workdir = str(pathlib.Path(__file__).resolve().parent)
        else:
            self.workdir = workdir


    def load(
        self,
        omograph_model_size="big",
        use_dictionary=False,
        custom_dict={},
        custom_homographs={},
        load_yo_homographs_model=False,
        repo="TeraTTS/accentuator",
        ):

        self.load_yo_homographs_model = load_yo_homographs_model
        self.custom_dict = custom_dict
        self.accents = {}
        if not os.path.exists(
            join_path(self.workdir, "dictionary")
        ):
            for path in self.accentuator_paths:
                files = self.fs.ls(repo + path)
                for file in files:
                    hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=self.workdir, filename=file['name'].replace(repo+'/', ''))
    
        if not os.path.exists(join_path(self.workdir, "nn")):
            os.mkdir(join_path(self.workdir, "nn"))
        
        if not os.path.exists(join_path(self.workdir, "nn", "nn_omograph", omograph_model_size)):
            model_path = self.omograph_models_paths.get(omograph_model_size, None)
            if model_path:
                files = self.fs.ls(repo + model_path)
                for file in files:
                    hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=self.workdir, filename=file['name'].replace(repo+'/', ''))
            else:
                raise FileNotFoundError
        
        self.omographs = json.load(
            open(join_path(self.workdir, "dictionary/omographs.json"), encoding='utf-8')
        )
        self.omographs.update(custom_homographs)

        if load_yo_homographs_model:
            if not os.path.exists(join_path(self.workdir, "nn", "nn_yo_omograph")):
                for path in self.yo_omograph_path:
                    files = self.fs.ls(repo + path)
                    for file in files:
                        hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=self.workdir, filename=file['name'].replace(repo+'/', ''))

            self.yo_omographs = json.load(
                open(join_path(self.workdir, "dictionary/yo_omographs.json"), encoding='utf-8')
            )
            self.yo_omograph_model.load(join_path(self.workdir, "nn/nn_yo_omograph/"))

        self.yo_words = json.load(
            open(join_path(self.workdir, "dictionary/yo_words.json"), encoding='utf-8')
        )

        if use_dictionary:
            self.accents.update(json.load(
                open(join_path(self.workdir, "dictionary/accents.json"), encoding='utf-8')
            ))

        self.accents.update(self.custom_dict)

        self.omograph_model.load(
            join_path(self.workdir, f"nn/nn_omograph/{omograph_model_size}/")
        )
        self.accent_model.load(join_path(self.workdir, "nn/nn_accent/"))



    def split_by_words(self, string):
        result = re.findall(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower())
        return [res for res in result if res]


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

    def _process_yo(self, text):
        splitted_text = text

        for i, word in enumerate(splitted_text):
            splitted_text[i] = self.yo_words.get(word, word)
        return splitted_text

    def _process_omographs(self, text):
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

    def _process_yo_omographs(self, text):
        splitted_text = text

        founded_omographs = []
        for i, word in enumerate(splitted_text):
            variants = self.yo_omographs.get(word)
            if variants:
                founded_omographs.append(
                    {"word": word, "variants": variants, "position": i}
                )
        for omograph in founded_omographs:
            splitted_text[
                omograph["position"]
            ] = f"<w>{splitted_text[omograph['position']]}</w>"
            cls = self.yo_omograph_model.classify(
                " ".join(splitted_text), omograph["variants"]
            )
            splitted_text[omograph["position"]] = cls
        return splitted_text
    
    def _process_accent(self, text):
        splitted_text = text

        for i, word in enumerate(splitted_text):
            stressed_word = self.accents.get(word, word)
            if stressed_word == word and not self.has_punctuation(word) and self.count_vowels(word) > 1:
                splitted_text[i] = self.accent_model.put_accent(word)
            else:
                splitted_text[i] = stressed_word
        return splitted_text

    def process_yo(self, text, process_yo_omographs=False):
        sentences = split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            text = self.split_by_words(sentence)
            processed_text = self._process_yo(text)
            if process_yo_omographs:
                processed_text = self._process_yo_omographs(processed_text)
            processed_text = " ".join(processed_text)
            processed_text = self.delete_spaces_before_punc(processed_text)
            outputs.append(processed_text)
        return " ".join(outputs)
    
    def process_all(self, text, process_yo_omographs=False):
        sentences = split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            text = self.split_by_words(sentence)
            processed_text = self._process_yo(text)
            if process_yo_omographs:
                processed_text = self._process_yo_omographs(processed_text)
            processed_text = self._process_omographs(processed_text)
            processed_text = self._process_accent(processed_text)
            processed_text = " ".join(processed_text)
            processed_text = self.delete_spaces_before_punc(processed_text)
            outputs.append(processed_text)
        return " ".join(outputs)
