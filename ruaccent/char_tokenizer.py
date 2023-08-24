import os
from typing import Optional, Tuple, List
from collections import OrderedDict

from transformers import PreTrainedTokenizer


def load_vocab(vocab_file):
    vocab = OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class CharTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.txt"}

    def __init__(
        self,
        vocab_file=None,
        pad_token="[pad]",
        unk_token="[unk]",
        bos_token="[bos]",
        eos_token="[eos]",
        do_lower_case=False,
        *args,
        **kwargs
    ):
        super().__init__(
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            do_lower_case=do_lower_case,
            **kwargs
        )
        self.do_lower_case = do_lower_case

        if not vocab_file or not os.path.isfile(vocab_file):
            self.vocab = OrderedDict()
            self.ids_to_tokens = OrderedDict()
        else:
            self.vocab = load_vocab(vocab_file)
            self.ids_to_tokens = OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])

    @property
    def vocab_size(self):
        return len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def _convert_token_to_id(self, token):
        if self.do_lower_case:
            token = token.lower()
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index):
        return self.ids_to_tokens[index]

    def _tokenize(self, text):
        if self.do_lower_case:
            text = text.lower()
        return list(text)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        return bos + token_ids_0 + eos

    def get_special_tokens_mask(
         self,
         token_ids_0: List[int],
         token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        return (len(token_ids_0) + 2) * [0]

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        assert os.path.isdir(save_directory)
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") +
            self.vocab_files_names["vocab_file"]
        )
        index = 0
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                assert index == token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
