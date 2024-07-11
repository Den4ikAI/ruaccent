import re
from razdel import sentenize
from razdel.substring import Substring

class TextPreprocessor:
    @staticmethod
    def split_by_words(string):
        print("In", string)
        string = string.replace(" - ",' ~ ')
        match = list(re.finditer(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower()))
        if len(match) == 0:
            return []

        remaining_text =  [string[l.end():r.start()] for l,r in zip(match, match[1:])]
        words = [string[x.start():x.end()] for x in match]

        words_indices = [i for i, w  in enumerate(words) if w]
        
        valid_words = [words[i] for i in words_indices]

        remaining_text = ([string[:words_indices[0]]] if words_indices[0] > 0 else [""]) + ["".join(remaining_text[l+1:r]) for l, r in zip(words_indices, words_indices[1:])] 
        remaining_text.append("".join(remaining_text[words_indices[-1] + 1:]))
        
        return valid_words, remaining_text

    @staticmethod
    def split_by_sentences(string):
        result = list(sentenize(string))
        result = [string[l.stop:r.start] + r.text if l.stop != r.start else r.text for l,r in zip([Substring(0,0, "")] + result, result)]
        return result
