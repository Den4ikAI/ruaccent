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
from razdel import sentenize
import re



class RUAccent:
    def __init__(self):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        self.stress_usage_predictor = StressUsagePredictorModel()
        self.yo_homograph_model = YoHomographModel()
        self.fs = HfFileSystem()
        self.normalize = re.compile(r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;""''(){}\[\]«»„“”-]")
        self.omograph_models_paths = {'big_poetry': '/nn/nn_omograph/big_poetry', 
                                      'medium_poetry': '/nn/nn_omograph/medium_poetry', 
                                      'small_poetry': '/nn/nn_omograph/small_poetry',
                                      'turbo': '/nn/nn_omograph/turbo',
                                      'turbo2': '/nn/nn_omograph/turbo2',
                                      'tiny': '/nn/nn_omograph/tiny',
                                      'test': "/media/denis/042CD5B7300C3479/stress_dataset/glycine/bert4/pretrain_ruaccent_turbo/onnx_deberta_10k"}
    
        self.accentuator_paths = ['/nn/nn_accent', '/nn/nn_stress_usage_predictor','/nn/nn_yo_homograph_resolver', '/dictionary', '/dictionary/rule_engine']
        self.letters_accent = {'о': '+о', 'О': '+О'}
        self.koziev_paths = ["/koziev/rulemma", "/koziev/rupostagger", "/koziev/rupostagger/database"]
        self.tiny_mode = False
        
    def load(
        self,
        omograph_model_size="turbo2",
        use_dictionary=False,
        custom_dict={},
        custom_homographs={},
        device="CPU",
        repo="ruaccent/accentuator",
        workdir=None,
        tiny_mode=False
        ):
        self.tiny_mode = tiny_mode
        if workdir:
            self.workdir = workdir
        else:
            self.workdir = str(pathlib.Path(__file__).resolve().parent)
        self.module_path = str(pathlib.Path(__file__).resolve().parent)
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
        if not os.path.exists(join_path(self.module_path, "koziev")):
          for path in self.koziev_paths:
               files = self.fs.ls(repo + path)
               for file in files:
                   if file["type"] == "file":
                       hf_hub_download(repo_id=repo, local_dir_use_symlinks=False, local_dir=self.module_path, filename=file['name'].replace(repo+'/', ''))
        if self.tiny_mode:
            self.accents.update(self.custom_dict)
            self.accents.update(self.letters_accent)

            self.omographs = json.load(
            gzip.open(join_path(self.workdir, "dictionary","omographs.json.gz"))
        )
            self.omographs.update(custom_homographs)
            self.yo_words = json.load(
            gzip.open(join_path(self.workdir, "dictionary","yo_words.json.gz"))
            )
            self.accents.update(json.load(
                gzip.open(join_path(self.workdir, "dictionary","accents_nn.json.gz"))
            ))
            self.omograph_model.load(join_path(self.workdir, self.omograph_models_paths[omograph_model_size][1:]), device=device)
            self.accent_model.load(join_path(self.workdir, "nn","nn_accent/"), device=device)
            self.yo_homographs = json.load(
                gzip.open(join_path(self.workdir, "dictionary","yo_homographs.json.gz"))
            )
            self.yo_homograph_model.load(join_path(self.workdir, "nn","nn_yo_homograph_resolver"), device=device)
        else:
            from .rule_accent_engine import RuleEngine
            self.rule_accent = RuleEngine()
            self.omographs = json.load(
                gzip.open(join_path(self.workdir, "dictionary","omographs.json.gz"))
            )
            self.omographs.update(custom_homographs)
    
            self.yo_words = json.load(
                gzip.open(join_path(self.workdir, "dictionary","yo_words.json.gz"))
            )
    
            if use_dictionary:
                self.accents.update(json.load(
                    gzip.open(join_path(self.workdir, "dictionary","accents.json.gz"))
                ))
            else:
                self.accents.update(json.load(
                    gzip.open(join_path(self.workdir, "dictionary","accents_nn.json.gz"))
                ))
            self.accents.update(self.custom_dict)
            self.accents.update(self.letters_accent)
            self.omograph_model.load(join_path(self.workdir, self.omograph_models_paths[omograph_model_size][1:]), device=device)        
            self.yo_homographs = json.load(
                gzip.open(join_path(self.workdir, "dictionary","yo_homographs.json.gz"))
            )
            self.accent_model.load(join_path(self.workdir, "nn","nn_accent/"), device=device)
            self.stress_usage_predictor.load("/media/denis/042CD5B7300C3479/stress_dataset/glycine/predict_need_stress/tiny_onnx", device=device)
            self.yo_homograph_model.load(join_path(self.workdir, "nn","nn_yo_homograph_resolver"), device=device)
            self.rule_accent.load(join_path(self.workdir, "dictionary","rule_engine"))

    def split_by_words(self, string):
        string = string.replace(" - ",' ~ ')
        result = re.findall(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower())
        return [res for res in result if res]

    def split_by_sentences(self, string):
        result = list(sentenize(string))
        result = [_.text for _ in result]
        return result

    def count_vowels(self, text):
        vowels = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"
        return sum(1 for char in text if char in vowels)

    def has_punctuation(self, text):
        for char in text:
            if char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                return True
        return False

    def delete_spaces_before_punc(self, text):
        punc = "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-"
        for char in punc:
            if char == '-':
                text = text.replace(" " + char, char).replace(char + " ", char)
            text = text.replace(" " + char, char)
        return text.replace('~', '-')

    def extract_entities(self, data):
        entities = []
        for item in data:
            entity = item['entity']
            entities.append(entity)
        return entities

    def _process_yo(self, text, sentence):
        splitted_text = text
        yo_predictions = self.extract_entities(self.yo_homograph_model.predict_yo_homographs(sentence))
        
        for i, word in enumerate(splitted_text):
            splitted_text[i] = self.yo_words.get(word, word)
            if yo_predictions[i] == "YO":
                splitted_text[i] = self.yo_homographs.get(word, word)
        return splitted_text

    def _process_omographs(self, text, sentence):
        splitted_text = text
    
        founded_omographs = []
        texts = []
        hypotheses = []
    
        for i, word in enumerate(splitted_text):
            variants = self.omographs.get(word)
            if variants:
                founded_omographs.append(
                    {"word": word, "variants": variants, "position": i}
                )
                texts.append(splitted_text)
                hypotheses.append(variants)
    
        if len(founded_omographs) > 0:
            texts_batch = []
            hypotheses_batch = [val for sublist in hypotheses for val in sublist]

            for o, t in zip(founded_omographs, texts):
                position = o["position"]
                t_back = t[position]
                t[position] = ' <w>' + t[position] + '</w> '
                for _ in range(len(o["variants"])):
                    texts_batch.append(self.delete_spaces_before_punc(" ".join(t.copy())))
                t[position] = t_back
            cls_batch = self.omograph_model.classify(texts_batch, hypotheses_batch)
    
            cls_index = 0
            for omograph in founded_omographs:
                position = omograph["position"]
                splitted_text[position] = cls_batch[cls_index]
                cls_index += 1
    
        return splitted_text


    
    def _process_accent(self, text, stress_usages):
        splitted_text = text
        for i, word in enumerate(splitted_text):
            if stress_usages[i] == "STRESS":
                stressed_word = self.accents.get(word, word)
                if stressed_word == word and not self.has_punctuation(word) and self.count_vowels(word) > 1:
                    splitted_text[i] = self.accent_model.put_accent(word)
                else:
                    splitted_text[i] = stressed_word
        return splitted_text

        
    def process_yo(self, text):
        sentences = self.self.split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            text = self.split_by_words(sentence)
            processed_text = self._process_yo(text)
            processed_text = " ".join(processed_text)
            processed_text = self.delete_spaces_before_punc(processed_text)
            outputs.append(processed_text)
        return " ".join(outputs)
    
    def process_all(self, text):
        text = re.sub(self.normalize, "", text)
        sentences = self.split_by_sentences(text)
        outputs = []
        for sentence in sentences:
            text = self.split_by_words(sentence)
            accented_tokens = self.rule_accent.accentuate(sentence) if not self.tiny_mode else text
            if len(accented_tokens) == len(text):
                for i in range(len(text)):
                    if '+' in accented_tokens[i]:
                        print(accented_tokens[i])
                        text[i] = accented_tokens[i]
            stress_usages = self.extract_entities(self.stress_usage_predictor.predict_stress_usage(sentence)) if not self.tiny_mode else ["STRESS"] * len(text)
            processed_text = self._process_yo(text, sentence)
            processed_text = self._process_omographs(processed_text, sentence)
            processed_text = self._process_accent(processed_text, stress_usages)
            processed_text = " ".join(processed_text)
            processed_text = self.delete_spaces_before_punc(processed_text)
            outputs.append(processed_text)
        return " ".join(outputs)
