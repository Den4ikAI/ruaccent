import json
import re
import gzip
from os.path import join as join_path

from .koziev.rupostagger.rupostagger import RuPosTagger
from .koziev.rulemma.rulemma import Lemmatizer
from collections import Counter




class RuleEngine:
    def __init__(self):
        self.wordforms = {}
        self.forms_dict = {}

    def load(self, path):
        with open(file=join_path(path, "accents.json"), mode='r') as f:
            self.wordforms = json.load(f)
        with open(file=join_path(path, "forms.json"), mode='r') as f:
            self.forms_dict = json.load(f)
        self.lemmatizer = Lemmatizer()
        self.lemmatizer.load()
        self.tagger = RuPosTagger()
        self.tagger.load()
    def split_by_words(self, string):
        string = string.replace(" - ",' ~ ')
        result = re.findall(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower())
        return [res for res in result if res]

    def all_elements_equal(self, lst):
        return all(x == lst[0] for x in lst)

    def calculate_similarity(self, set1, set2):
        intersection = set1 & set2
        union = set1 | set2
        return (len(intersection) / len(union)) * 100 if union else 100

    def check_lemmas(self, word):
        lemmas = [i['lemma'] for i in word["interpretations"]]
        if len(lemmas) > 2 or not self.all_elements_equal(lemmas):
            for interpretation in word['interpretations']:
                if interpretation['lemma'] == word['lemma']:
                    return interpretation["accentuated"]
        return None

    def compatible(self, interpretation, tag):
        forms = " ".join(self.forms_dict.get(form, "") for form in interpretation.split())
        tags = set(tag.replace("Voice=Act", "").replace("VerbForm=Fin", "").split("|"))
        return self.calculate_similarity(tags, set(forms.split()))
    '''
    def accentuate_word(self, word):
        accented_lemma = self.check_lemmas(word)
        if accented_lemma and word["token"] != "белье":
            return accented_lemma
        else:
            check_tag = [self.compatible(interpretation['form'], word['tag'])
                         for interpretation in word['interpretations']]
            max_compatible_index = check_tag.index(max(check_tag))
            return word["interpretations"][max_compatible_index]["accentuated"]
    '''
    def all_lemmas_unique(self, word):
        lemmas = [i["lemma"] for i in word['interpretations']]
        counter = Counter(lemmas)
        for count in counter.values():
            if count == 1:
                return True
        return False
    def accentuate_word(self, word):
        if self.all_lemmas_unique(word):
            accented_lemma = self.check_lemmas(word)
        else:
            check_tag = [self.compatible(interpretation['form'], word['tag'])
                            for interpretation in word['interpretations']]
            max_compatible_index = check_tag.index(max(check_tag))
            if word["interpretations"][max_compatible_index]["accentuated"]:
                return word["interpretations"][max_compatible_index]["accentuated"]
        return accented_lemma

#    def tokenize(self, text):
#        text_tokenized = self.split_by_words(text)
#        if any(word in self.wordforms for word in text_tokenized):
#            processed = self.morph.process(text)
#            return [
#                {"token": word, "tag": tags, "homograph": word.lower() in self.wordforms, "lemma": lemma, "interpretations": #self.wordforms.get(word.lower(), [])}
#                for word, tags, lemma in processed
#            ]
#        else:
#            return [{"token": token, "homograph": False} for token in text_tokenized]

    def tk(self, text):
        text_tokenized = self.split_by_words(text)
        if any(word in self.wordforms for word in text_tokenized):
            outs = []
            tags = self.tagger.tag(text_tokenized)
            lemmas = self.lemmatizer.lemmatize(tags)
            for word, tags, lemma, *_ in lemmas:
                outs.append({"token": word, "tag": tags,  "homograph": word.lower() in self.wordforms, "lemma": lemma, "interpretations": self.wordforms.get(word.lower(), [])})
            return outs 
        else:
            return [{"token": token, "homograph": False} for token in text_tokenized]

    def accentuate(self, text):
        words = self.tk(text)
        
        return [self.accentuate_word(word) if word["homograph"] else word['token'] for word in words]
