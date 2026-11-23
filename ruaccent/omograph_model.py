import re

import numpy as np
from transformers import AutoTokenizer

from .session_helper import create_session
from .token_classifier import softmax

_SPACE_BEFORE_PUNCT = re.compile(r"\s+(?=[,.?!:;…])")


class OmographModel:
    """Cross-encoder that scores (context, stressed variant) pairs."""

    batch_size = 32

    def load(self, path, device="CPU"):
        self.session = create_session(f"{path}/model.onnx", device=device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _score(self, texts, hypotheses):
        """Probability that each hypothesis is the right reading of its text."""
        probs = []
        for i in range(0, len(texts), self.batch_size):
            inputs = self.tokenizer(
                texts[i:i + self.batch_size],
                hypotheses[i:i + self.batch_size],
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=512,
            )
            feed = {k: v.astype(np.int64) for k, v in inputs.items() if k in self.input_names}
            logits = self.session.run(None, feed)[0]
            probs.extend(softmax(logits)[:, 1].tolist())
        return probs

    def classify(self, texts, hypotheses, num_hypotheses):
        """Pick the best variant for every homograph.

        ``texts`` and ``hypotheses`` are flat, aligned lists; ``num_hypotheses``
        says how many consecutive entries belong to each homograph.
        """
        texts = [_SPACE_BEFORE_PUNCT.sub("", t) for t in texts]
        probs = self._score(texts, hypotheses)
        outs, start = [], 0
        for n in num_hypotheses:
            group = probs[start:start + n]
            outs.append(hypotheses[start + group.index(max(group))])
            start += n
        return outs
