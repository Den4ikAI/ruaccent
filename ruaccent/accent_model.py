import torch
from .char_tokenizer import CharTokenizer
from transformers import AutoModelForTokenClassification

class AccentModel:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def load(self, path):
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device)
        self.tokenizer = CharTokenizer.from_pretrained(path)
    
    def render_stress(self, word, token_classes):
        if 'STRESS' in token_classes:
            index = token_classes.index('STRESS')
            word = list(word)
            word[index-1] = '+' + word[index-1]
            return ''.join(word)
        else:
            return word
    
    def put_accent(self, word):
        inputs = self.tokenizer(word, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=2)
            predicted_token_class = [self.model.config.id2label[t.item()] for t in predictions[0]]
        return self.render_stress(word, predicted_token_class)
