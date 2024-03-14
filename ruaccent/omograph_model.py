import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer
import re


class OmographModel:
    def __init__(self):
        self.special_words = ['балчуга', 'вертела', 'волоки', 'волоку', 'воронью', 'выбродите', 'вывозите', 'выносите', 'выноситесь', 'выходите', 'железы', 'начала', 'округа', 'перепела', 'развитая', 'развитого', 'развитое', 'развитой', 'развитом', 'развитому', 'развитою', 'развитую', 'развитые', 'развитым', 'развитыми', 'развитых', 'сторожа', 'сторожи', 'сторожу', 'удало', 'начался', 'началась', 'началось', 'бутиках', 'ожила', 'создало', 'коротки', 'проклята', 'роженица', 'роженицы', 'рожениц', 'роженице', 'роженицам', 'роженицу', 'роженицей', 'роженицею', 'роженицами', 'роженицах', 'пристава', 'приставов', 'приставам', 'приставами', 'приставах', 'пережитое', 'пережитого', 'пережитые', 'пережитых', 'пережитому', 'пережитым', 'пережитыми', 'пережитом', 'нипоняла']


    def load(self, path, device="CPU"):
        self.session = InferenceSession(f"{path}/model.onnx", providers=["CUDAExecutionProvider" if device == "CUDA" else "CPUExecutionProvider"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def group_words(self, words):
        groups = {}
        for word in words:
            parts = word.replace('+', '')
            key = parts
            group = groups.setdefault(key, [])
            group.append(word)
    
        result = []
        for group in groups.values():
            has_special_word = any(word.replace('+', '') in self.special_words for word in group)
            if has_special_word and len(group) > 3:
                subgroups = [group[i:i+3] for i in range(0, len(group), 3)]
                result.extend(subgroups)
            elif len(group) > 3 and len(group) % 2 == 0:
                subgroups = [group[i:i+2] for i in range(0, len(group), 2)]
                result.extend(subgroups)
            else:
                result.append(group)
    
        return result
        
    def transfer_grouping(self, grouped_list, target_list):
        new_grouped_list = []
        start_index = 0
        for group in grouped_list:
            group_length = len(group)
            new_group = target_list[start_index:start_index + group_length]
            new_grouped_list.append(new_group)
            start_index += group_length
        return new_grouped_list
        
    def classify(self, texts, hypotheses):
        hypotheses_probs = []
        preprocessed_texts = [re.sub(r'\s+(?=(?:[,.?!:;…]))', r'', text) for text in texts]
        if len(hypotheses) % 2 != 0:
            #print("NO_BATCH")
            outs = []
            grouped_h = self.group_words(hypotheses)
            grouped_t = self.transfer_grouping(grouped_h, preprocessed_texts)
            for h, t in zip(grouped_h, grouped_t):
                probs = []
                for hp in h:
                    inputs = self.tokenizer(t[0], hp, max_length=512, truncation=True, return_tensors="np")
                    inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
                    outputs = self.session.run(None, inputs)[0]
                    outputs = self.softmax(outputs)
                    prob_label_is_true = [float(p[1]) for p in outputs][0]
                    probs.append(prob_label_is_true)
                    #print(h, prob_label_is_true)
                outs.append(h[probs.index(max(probs))])
            return outs
        else:
            inputs = self.tokenizer(preprocessed_texts, hypotheses, return_tensors="np", padding=True, truncation=True, max_length=512)
            inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
    
            outputs = self.session.run(None, inputs)[0]
            outputs = self.softmax(outputs)
            #print(hypotheses)
            preprocessed_texts = [(preprocessed_texts[i], preprocessed_texts[i+1]) for i in range(0, len(preprocessed_texts), 2)]
            hypotheses =  [(hypotheses[i], hypotheses[i+1]) for i in range(0, len(hypotheses), 2)]
            
            for i in range(len(texts)):
                prob_label_is_true = float(outputs[i][1])
                hypotheses_probs.append(prob_label_is_true)
    
            hypotheses_probs = [(hypotheses_probs[i], hypotheses_probs[i+1]) for i in range(0, len(hypotheses_probs), 2)]
            outs = []
            for pair1, pair2 in zip(hypotheses, hypotheses_probs):
              outs.append(pair1[pair2.index(max(pair2))])
            return outs
