import numpy as np
import json
from onnxruntime import InferenceSession
from transformers import AutoTokenizer

class StressUsagePredictorModel:
    def __init__(self) -> None:
        pass

    def load(self, path, device="CPU"):
        self.session = InferenceSession(f"{path}/model.onnx", providers=["CUDAExecutionProvider" if device == "CUDA" else "CPUExecutionProvider"])

        with open(f"{path}/config.json", "r") as f:
            self.id2label = json.load(f)["id2label"]
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        #self.tokenizer.model_input_names = ["input_ids", "attention_mask"]

    def collect_pre_entities(self, sentence, input_ids, scores, offset_mapping, special_tokens_mask):
        pre_entities = []
        for idx, token_scores in enumerate(scores):
            if special_tokens_mask[idx]:
                continue
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
           
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                if not isinstance(start_ind, int):
                    start_ind = int(start_ind)
                    end_ind = int(end_ind)
                word_ref = sentence[start_ind:end_ind]
                if getattr(self.tokenizer._tokenizer.model, "continuing_subword_prefix", None):
                    is_subword = len(word) != len(word_ref)
                else:
                    is_subword = sentence[start_ind - 1 : start_ind] != " " if start_ind > 0 else False
    
                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                start_ind = None
                end_ind = None
                is_subword = False
    
            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            pre_entities.append(pre_entity)
        return pre_entities
    
    def aggregate_word(self, entities, aggregation_strategy):
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        if aggregation_strategy == "FIRST":
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.id2label[str(idx)]
        elif aggregation_strategy == "MAX":
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.id2label[str(idx)]
        elif aggregation_strategy == "AVERAGE":
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.id2label[str(entity_idx)]
            score = average_scores[entity_idx]
    
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity
    
    def aggregate_words(self, entities, aggregation_strategy):
        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities
    
    def predict_stress_usage(self, text):
        inputs = self.tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True, return_tensors="np")
        offset_mapping, special_tokens_mask, input_ids = inputs.pop('offset_mapping')[0], inputs.pop('special_tokens_mask')[0], inputs['input_ids'][0]
        inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
        outputs = self.session.run(None, inputs)
        logits = outputs[0]
        maxes = np.max(logits, axis=-1, keepdims=True)
        shifted_exp = np.exp(logits - maxes)
        scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)
        pre_entities = self.collect_pre_entities(text, input_ids, scores[0], offset_mapping, special_tokens_mask)
        #print(pre_entities)
        grouped_entities = self.aggregate_words(pre_entities, "AVERAGE")
        return grouped_entities