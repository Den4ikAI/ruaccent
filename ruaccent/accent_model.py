import numpy as np
import json
from onnxruntime import InferenceSession
from .char_tokenizer import CharTokenizer

class AccentModel:
    def __init__(self) -> None:
        pass

    def load(self, path):
        self.session = InferenceSession(f"{path}/model.onnx", providers=["CPUExecutionProvider"])

        with open(f"{path}/config.json", "r") as f:
            self.id2label = json.load(f)["id2label"]
        self.tokenizer = CharTokenizer.from_pretrained(path)
        self.tokenizer.model_input_names = ["input_ids", "attention_mask"]

    def render_stress(self, text, pred):
        text = list(text)
        i = 0
        for chunk in pred:
            if chunk != "NO":
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
        labels = np.argmax(logits, axis=-1)[0]
        labels = [self.id2label[str(label)] for label in labels]
        stressed_word = self.render_stress(word, labels)
        return stressed_word