from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class OmographModel:
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load(self, path):
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
    def classify(self, text, hypotheses):
        encodings = self.tokenizer.batch_encode_plus([(text, hyp) for hyp in hypotheses], return_tensors='pt', padding=True)
        input_ids = encodings['input_ids'].to(self.device)
        with torch.no_grad():
            logits = self.nli_model(input_ids)[0]
            entail_contradiction_logits = logits[:,[0,2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = [float(p[1]) for p in probs]

        return hypotheses[prob_label_is_true.index(max(prob_label_is_true))]