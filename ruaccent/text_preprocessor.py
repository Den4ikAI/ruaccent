import re
from razdel import sentenize
from razdel.substring import Substring

class TextPreprocessor:
    @staticmethod
    def split_by_words(string):
        string = string.replace(" - ",' ~ ')
        match = list(re.finditer(r"\w*(?:\+\w+)*|[^\w\s]+", string.lower()))
        remaining_text =  [string[l.end():r.start()] for l,r in zip(match, match[1:])]

        words = [string[x.start():x.end()] for x in match]
        words_mask = [i for i, w  in enumerate(words) if w]
        
        valid_words = [words[i] for i in words_mask]
        if len(words_mask) == 0:
            return valid_words, ["", ""]
        remaining_text_res = ["".join(remaining_text[:words_mask[0]])] + ["".join(remaining_text[l+1:r]) for l, r in zip(words_mask, words_mask[1:])]
        remaining_text_res.append("".join(remaining_text[words_mask[-1]+1:]))
        return valid_words, remaining_text_res

    @staticmethod
    def split_by_sentences(string):
        sentences = list(sentenize(string))
        if len(sentences) == 0:
            return []
        result = [string[l.stop:r.start] + r.text if l.stop != r.start else r.text for l,r in zip([Substring(0,0, "")] + sentences, sentences)]
        result[-1] = result[-1] + string[sentences[-1].stop:]
        return result
