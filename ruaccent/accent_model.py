import numpy as np
import json
from onnxruntime import InferenceSession
from .char_tokenizer import CharTokenizer

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class AccentModel:
    def __init__(self) -> None:
        pass

    def load(self, path, device="CPU"):
        self.session = InferenceSession(f"{path}/model.onnx", providers=["CUDAExecutionProvider" if device == "CUDA" else "CPUExecutionProvider"])

        with open(f"{path}/config.json", "r") as f:
            self.id2label = json.load(f)["id2label"]
        self.tokenizer = CharTokenizer.from_pretrained(path)

    def render_stress(self, text, pred):
        text = list(text)
        i = 0
        for chunk in pred:
            if chunk['label'] != "NO" and chunk['label'] != "STRESS_SECONDARY" and chunk["score"] >= 0.55:
                text[i - 1] = "+" + text[i - 1]
            i += 1
        text = "".join(text)
        return text

    def put_accent(self, word):
        inputs = self.tokenizer(word, return_tensors="np")
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        outputs = self.session.run(None, inputs)
        output_names = {output_key.name: idx for idx, output_key in enumerate(self.session.get_outputs())}
        logits = outputs[output_names["logits"]]
        probabilities = softmax(logits)
        scores = np.max(probabilities, axis=-1)[0]
        labels = np.argmax(logits, axis=-1)[0]
        pred_with_scores = [{'label': self.id2label[str(label)], 'score': float(score)} 
                            for label, score in zip(labels, scores)]

        stressed_word = self.render_stress(word, pred_with_scores)

        return stressed_word