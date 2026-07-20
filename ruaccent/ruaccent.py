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
import re



class RUAccent:
    def __init__(self):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        self.stress_usage_predictor = StressUsagePredictorModel()
        self.yo_homograph_model = YoHomographModel()
        self.fs = HfFileSystem()
        self.normalize = re.compile(r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;\"''(){}\[\]«»„“”-]")
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
        
        self.omographs = json.load(
            gzip.open(join_path(self.workdir, "dictionary","omographs.json.gz"))
        )
        self.omographs.update({"коса": ["к+оса", "кос+а"]})
        self.omographs.update(custom_homographs)
        self.omograph_model.load(join_path(self.workdir, self.omograph_models_paths[omograph_model_size][1:]), device=device)

        self.yo_words = json.load(
            gzip.open(join_path(self.workdir, "dictionary","yo_words.json.gz"))
        ) 
        self.accent_model.load(join_path(self.workdir, "nn","nn_accent/"), device=device)
        self.yo_homographs = json.load(
                gzip.open(join_path(self.workdir, "dictionary","yo_homographs.json.gz"))
            ) 
        self.yo_homograph_model.load(join_path(self.workdir, "nn","nn_yo_homograph_resolver"), device=device)

        if self.tiny_mode or not use_dictionary:
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
            from .rule_accent_engine import RuleEngine
            self.rule_accent = RuleEngine()

            self.stress_usage_predictor.load(join_path(self.workdir, "nn","nn_stress_usage_predictor/"), device=device)
            self.rule_accent.load(join_path(self.workdir, "dictionary","rule_engine"))


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

    def _process_yo(self, words, sentence):
        lower_sentence = sentence.lower()

        yo_predictions = None
        if 'е' in lower_sentence:
            yo_predictions = self.extract_entities(self.yo_homograph_model.predict_yo_homographs(lower_sentence))
        
        for i, word in enumerate(words):
            lower_word = word.lower()
            words[i] = fix_capital(word, self.yo_words.get(lower_word, word))
            if yo_predictions and yo_predictions[i] == "YO":
                words[i] = fix_capital(word, self.yo_homographs.get(lower_word, word))
        return words


    def _process_omographs(self, text):
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
                splitted_text[position] = cls_batch[cls_index]
        return splitted_text


    
    def _process_accent(self, text, stress_usages):
        splitted_text = text
        for i, word in enumerate(splitted_text):
            if '+' in word:
                continue
            if stress_usages[i] == "STRESS":
                lower_word = word.lower()
                stressed_word = self.accents.get(lower_word, lower_word)
                if stressed_word == lower_word and not self.has_punctuation(lower_word) and self.count_vowels(lower_word) > 1:
                    splitted_text[i] = self.accent_model.put_accent(word)
                else:
                    match = re.finditer(r'\+', stressed_word)
                    word_fixed = list(word)
                    for j, e in enumerate(list(match)):
                        word_fixed = word_fixed[:e.start() + j] + ["+"] + list(word)[e.end() - 1:]
                    splitted_text[i] = "".join(word_fixed)
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
            stress_usages = self.extract_entities(self.stress_usage_predictor.predict_stress_usage(sentence)) if not self.tiny_mode else ["STRESS"] * len(text)            
            processed_words = self._process_yo(words, sentence)
            processed_words = self._process_omographs(processed_words)
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


