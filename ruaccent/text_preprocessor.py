import re
from dataclasses import dataclass
from typing import List

from razdel import sentenize

# Characters the neural models were trained on. Everything else (emoji, %, /,
# user stress marks "+", ...) is hidden from the models but kept in the output.
MODEL_CHARS = re.compile(r"[a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;\"'(){}\[\]«»„“”-]")
CAPS_WORD = re.compile(r"\b[A-ZА-ЯЁ]{2,}\b")
TOKEN = re.compile(r"\w+|[^\w\s]+")


@dataclass
class Token:
    text: str          # token as the models see it
    start: int         # span in the model text
    end: int
    orig_start: int    # span in the original text
    orig_end: int
    original: str      # original text of the span

    @property
    def is_word(self):
        return self.text[0].isalnum() or self.text[0] == "_"

    @property
    def is_clean(self):
        """True if the original span is exactly what the models saw."""
        return self.original == self.text


@dataclass
class Sentence:
    original: str
    model_text: str
    tokens: List[Token]

    @property
    def case_normalized_text(self):
        """Model text with ALL-CAPS words lowercased (same length and offsets)."""
        return CAPS_WORD.sub(lambda m: m.group().lower(), self.model_text)


class TextPreprocessor:
    @staticmethod
    def split_by_sentences(string):
        """Split text into chunks whose concatenation is exactly ``string``."""
        sentences = list(sentenize(string))
        if not sentences:
            return [string] if string else []
        result, prev_stop = [], 0
        for s in sentences:
            result.append(string[prev_stop:s.stop])
            prev_stop = s.stop
        result[-1] += string[prev_stop:]
        return result

    @staticmethod
    def analyze(sentence):
        """Build the model view of a sentence and map its tokens back."""
        keep = [i for i, ch in enumerate(sentence) if MODEL_CHARS.match(ch)]
        model_text = "".join(sentence[i] for i in keep)
        # A free-standing dash is marked as "~" so the homograph model input
        # keeps the spaces around it (see join_for_model).
        tokens = []
        for m in TOKEN.finditer(model_text.replace(" - ", " ~ ")):
            orig_start, orig_end = keep[m.start()], keep[m.end() - 1] + 1
            tokens.append(Token(
                text=m.group(),
                start=m.start(),
                end=m.end(),
                orig_start=orig_start,
                orig_end=orig_end,
                original=sentence[orig_start:orig_end],
            ))
        return Sentence(sentence, model_text, tokens)

    @staticmethod
    def join_for_model(words):
        """Join tokens the way the homograph model saw text during training."""
        text = " ".join(words)
        for char in "!\"#$%&'()*,./:;<=>?@[\\]^_`{|}-":
            if char == "-":
                text = text.replace(" -", "-").replace("- ", "-")
            text = text.replace(" " + char, char)
        return text.replace("~", "-")
