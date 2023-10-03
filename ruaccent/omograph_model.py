import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
import re

class OmographModel:
    def __init__(self):
        pass

    def load(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.session = InferenceSession(f"{path}/model.onnx", providers=['CPUExecutionProvider'])

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def classify(self, text, hypotheses):
        hypotheses_probs = []
        text = re.sub(r'\s+(?=(?:[,.?!:;…]))', r'', text)
        for h in hypotheses:
            inputs = self.tokenizer(text, h, return_tensors="np")
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}

            outputs = self.session.run(None, inputs)[0]
            outputs = self.softmax(outputs)
            prob_label_is_true = [float(p[1]) for p in outputs][0]
            hypotheses_probs.append(prob_label_is_true)
        return hypotheses[hypotheses_probs.index(max(hypotheses_probs))]
