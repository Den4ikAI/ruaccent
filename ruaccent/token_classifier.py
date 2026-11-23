import json
from os.path import join as join_path

import numpy as np
from transformers import AutoTokenizer

from .session_helper import create_session


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class TokenClassifier:
    """ONNX token-classification model that returns one label per word.

    Word boundaries follow the tokenizer: sub-word tokens are merged into the
    preceding word and their class probabilities are averaged. Every returned
    entity carries character offsets (``start``/``end``) into the input text,
    so callers can align predictions with their own tokenization.
    """

    def load(self, path, device="CPU"):
        self.session = create_session(join_path(path, "model.onnx"), device=device)
        with open(join_path(path, "config.json"), "r", encoding="utf-8") as f:
            self.id2label = json.load(f)["id2label"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self._has_subword_prefix = bool(
            getattr(self.tokenizer._tokenizer.model, "continuing_subword_prefix", None)
        )

    def _collect_tokens(self, text, input_ids, scores, offset_mapping, special_tokens_mask):
        tokens = []
        for idx, token_scores in enumerate(scores):
            if special_tokens_mask[idx]:
                continue
            token = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            start, end = int(offset_mapping[idx][0]), int(offset_mapping[idx][1])
            if self._has_subword_prefix:
                is_subword = len(token) != len(text[start:end])
            else:
                is_subword = start > 0 and text[start - 1:start] != " "
            if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                is_subword = False
            tokens.append({"scores": token_scores, "start": start, "end": end, "is_subword": is_subword})
        return tokens

    def _aggregate(self, group, text):
        average = np.nanmean(np.stack([t["scores"] for t in group]), axis=0)
        idx = int(average.argmax())
        start, end = group[0]["start"], group[-1]["end"]
        return {
            "entity": self.id2label[str(idx)],
            "score": float(average[idx]),
            "word": text[start:end],
            "start": start,
            "end": end,
        }

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_tensors="np",
            truncation=True,
            max_length=512,
        )
        offset_mapping = inputs.pop("offset_mapping")[0]
        special_tokens_mask = inputs.pop("special_tokens_mask")[0]
        input_ids = inputs["input_ids"][0]
        feed = {k: v.astype(np.int64) for k, v in inputs.items() if k in self.input_names}
        if "token_type_ids" in self.input_names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        logits = self.session.run(None, feed)[0]
        scores = softmax(logits)[0]

        tokens = self._collect_tokens(text, input_ids, scores, offset_mapping, special_tokens_mask)
        words, group = [], []
        for token in tokens:
            if group and not token["is_subword"]:
                words.append(self._aggregate(group, text))
                group = []
            group.append(token)
        if group:
            words.append(self._aggregate(group, text))
        return words
