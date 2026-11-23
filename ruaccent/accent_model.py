import json
from os.path import join as join_path

import numpy as np

from .char_tokenizer import CharTokenizer
from .session_helper import create_session
from .token_classifier import softmax


class AccentModel:
    """Character-level model that puts a stress mark into a single word."""

    threshold = 0.55

    def load(self, path, device="CPU"):
        self.session = create_session(join_path(path, "model.onnx"), device=device)
        with open(join_path(path, "config.json"), "r", encoding="utf-8") as f:
            self.id2label = json.load(f)["id2label"]
        self.primary_id = next(int(i) for i, label in self.id2label.items() if label == "STRESS_PRIMARY")
        self.tokenizer = CharTokenizer.from_pretrained(path)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def put_accent(self, word):
        inputs = self.tokenizer(word.lower(), return_tensors="np")
        feed = {k: v.astype(np.int64) for k, v in inputs.items() if k in self.input_names}
        if "token_type_ids" in self.input_names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        logits = self.session.run(["logits"], feed)[0][0]
        probs = softmax(logits)[1:-1]  # drop [bos]/[eos]

        # A word has exactly one primary stress: take the most confident
        # character among those the model labels as primary.
        candidates = [
            (p[self.primary_id], i)
            for i, p in enumerate(probs[: len(word)])
            if p.argmax() == self.primary_id and p[self.primary_id] >= self.threshold
        ]
        if not candidates:
            return word
        _, pos = max(candidates)
        return word[:pos] + "+" + word[pos:]
