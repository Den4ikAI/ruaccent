import bisect
import gzip
import json
import os
import pathlib
import re
from os.path import join as join_path

from huggingface_hub import snapshot_download

from .accent_model import AccentModel
from .omograph_model import OmographModel
from .stress_usage_model import StressUsagePredictorModel
from .text_postprocessor import fix_capital, transfer_stress
from .text_preprocessor import CAPS_WORD, TextPreprocessor
from .yo_homograph_model import YoHomographModel

OMOGRAPH_MODELS = (
    "big_poetry", "medium_poetry", "small_poetry",
    "turbo", "turbo2", "turbo3", "turbo3.1",
    "tiny", "tiny2", "tiny2.1",
)
VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
PACKAGE_DIR = str(pathlib.Path(__file__).resolve().parent)


class _SpanLookup:
    """Find the model prediction covering a character offset."""

    def __init__(self, entities):
        self.entities = entities
        self.starts = [e["start"] for e in entities]

    def label(self, pos, default):
        i = bisect.bisect_right(self.starts, pos) - 1
        if i >= 0 and pos < self.entities[i]["end"]:
            return self.entities[i]["entity"]
        return default


class RUAccent:
    def __init__(self):
        self.omograph_model = OmographModel()
        self.accent_model = AccentModel()
        self.stress_usage_predictor = StressUsagePredictorModel()
        self.yo_homograph_model = YoHomographModel()
        self.omograph_models_paths = {name: f"/nn/nn_omograph/{name}" for name in OMOGRAPH_MODELS}
        self.letters_accent = {"о": "+о", "О": "+О"}
        self.tiny_mode = False

    # ------------------------------------------------------------------ loading

    def _accents_file(self, use_dictionary):
        # The full dictionary is big; without it the accent model covers the rest.
        return "accents.json.gz" if use_dictionary and not self.tiny_mode else "accents_nn.json.gz"

    def _required_files(self, omograph_model_size, use_dictionary):
        files = [
            f"dictionary/{name}"
            for name in ("omographs.json.gz", "yo_words.json.gz", "yo_homographs.json.gz",
                         self._accents_file(use_dictionary))
        ]
        dirs = ["nn/nn_accent", "nn/nn_yo_homograph_resolver", f"nn/nn_omograph/{omograph_model_size}"]
        if not self.tiny_mode:
            dirs.append("nn/nn_stress_usage_predictor")
        return files, dirs

    def _fetch(self, repo, workdir, files, dirs):
        def present(root):
            return all(os.path.exists(join_path(root, f)) for f in files) and all(
                os.path.exists(join_path(root, d, "model.onnx")) for d in dirs
            )

        # Models downloaded by older versions into the package directory.
        if workdir is None and present(PACKAGE_DIR):
            return PACKAGE_DIR
        if workdir is not None and present(workdir):
            return workdir
        return snapshot_download(
            repo_id=repo,
            local_dir=workdir,
            allow_patterns=files + [f"{d}/*" for d in dirs],
            ignore_patterns=["nn/nn_accent/big.onnx"],
        )

    def load(
        self,
        omograph_model_size="turbo2",
        use_dictionary=False,
        custom_dict=None,
        custom_homographs=None,
        device="CPU",
        repo="ruaccent/accentuator",
        workdir=None,
        tiny_mode=False,
    ):
        """Download (if needed) and load models and dictionaries.

        ``workdir``: where to keep the files. By default they go to the
        Hugging Face cache (``HF_HOME``).
        """
        if omograph_model_size not in self.omograph_models_paths:
            raise ValueError(
                f"Unknown omograph_model_size {omograph_model_size!r}; "
                f"choose one of: {', '.join(OMOGRAPH_MODELS)}"
            )
        self.tiny_mode = tiny_mode
        files, dirs = self._required_files(omograph_model_size, use_dictionary)
        self.workdir = self._fetch(repo, workdir, files, dirs)

        def load_json(name):
            with gzip.open(join_path(self.workdir, "dictionary", name)) as f:
                return json.load(f)

        self.omographs = load_json("omographs.json.gz")
        self.omographs["коса"] = ["к+оса", "кос+а"]
        self.omographs.update(custom_homographs or {})

        self.yo_words = load_json("yo_words.json.gz")
        self.yo_homographs = load_json("yo_homographs.json.gz")

        self.accents = load_json(self._accents_file(use_dictionary))
        self.accents.update(custom_dict or {})
        self.accents.update(self.letters_accent)

        nn = join_path(self.workdir, "nn")
        self.omograph_model.load(join_path(nn, "nn_omograph", omograph_model_size), device=device)
        self.accent_model.load(join_path(nn, "nn_accent"), device=device)
        self.yo_homograph_model.load(join_path(nn, "nn_yo_homograph_resolver"), device=device)
        if not self.tiny_mode:
            self.stress_usage_predictor.load(join_path(nn, "nn_stress_usage_predictor"), device=device)

    # --------------------------------------------------------------- pipeline

    @staticmethod
    def count_vowels(text):
        return sum(1 for ch in text if ch in VOWELS)

    # The sentence-level models are case-sensitive and get confused by ALL-CAPS
    # words ("БЕЛКИ ЕЛИ" -> "БЕЛК+И"), so those are lowercased before inference.

    def _apply_yo(self, sent, words):
        """words: {token index: current text}; updates it in place."""
        yo = None
        lower = sent.model_text.lower()
        if "е" in lower:
            yo = _SpanLookup(self.yo_homograph_model.predict_yo_homographs(lower))
        for i, token in enumerate(sent.tokens):
            if i not in words:
                continue
            lower = token.text.lower()
            words[i] = fix_capital(token.text, self.yo_words.get(lower, token.text))
            if yo is not None and lower in self.yo_homographs and yo.label(token.start, "NO_YO") == "YO":
                words[i] = fix_capital(token.text, self.yo_homographs[lower])

    def _apply_omographs(self, sent, words):
        context = [
            words.get(i, t.text).lower() if CAPS_WORD.fullmatch(t.text) else words.get(i, t.text)
            for i, t in enumerate(sent.tokens)
        ]
        found = []
        for i, word in words.items():
            variants = self.omographs.get(word.lower())
            if variants:
                found.append((i, variants))
        if not found:
            return
        texts, hypotheses = [], []
        for i, variants in found:
            marked = context.copy()
            # The homograph model is case-sensitive; it was trained on lowercase targets.
            marked[i] = " <w>" + marked[i].lower() + "</w> "
            text = TextPreprocessor.join_for_model(marked)
            texts.extend([text] * len(variants))
            hypotheses.extend(variants)
        chosen = self.omograph_model.classify(texts, hypotheses, [len(v) for _, v in found])
        for (i, _), variant in zip(found, chosen):
            words[i] = transfer_stress(words[i], variant)

    def _apply_accents(self, sent, words):
        if self.tiny_mode:
            usage = None
        else:
            usage = _SpanLookup(self.stress_usage_predictor.predict_stress_usage(sent.case_normalized_text))
        for i, word in words.items():
            if "+" in word:
                continue
            if usage is not None and usage.label(sent.tokens[i].start, "STRESS") != "STRESS":
                continue
            stressed = self.accents.get(word.lower())
            if stressed is not None:
                words[i] = transfer_stress(word, stressed)
            elif self.count_vowels(word) > 1:
                words[i] = self.accent_model.put_accent(word)

    def _process_sentence(self, sentence, yo_only=False):
        sent = TextPreprocessor.analyze(sentence)
        # Only touch words whose original text is exactly what the models saw;
        # anything with user stress marks or unusual characters stays as is.
        words = {i: t.text for i, t in enumerate(sent.tokens) if t.is_word and t.is_clean}
        if not words:
            return sentence
        self._apply_yo(sent, words)
        if not yo_only:
            self._apply_omographs(sent, words)
            self._apply_accents(sent, words)

        out, pos = [], 0
        for i, word in sorted(words.items()):
            token = sent.tokens[i]
            out.append(sentence[pos:token.orig_start])
            out.append(word)
            pos = token.orig_end
        out.append(sentence[pos:])
        return "".join(out)

    def _process(self, text, yo_only, skip_regex=None):
        def run(chunk):
            return "".join(
                self._process_sentence(s, yo_only) for s in TextPreprocessor.split_by_sentences(chunk)
            )

        if not skip_regex:
            return run(text)
        out, pos = [], 0
        for m in re.finditer(skip_regex, text):
            out.append(run(text[pos:m.start()]))
            out.append(m.group())
            pos = m.end()
        out.append(run(text[pos:]))
        return "".join(out)

    def process_yo(self, text, skip_regex=None):
        """Restore ё in the text."""
        return self._process(text, yo_only=True, skip_regex=skip_regex)

    def process_all(self, text, skip_regex=None):
        """Restore ё and put stress marks ("+" before the stressed vowel).

        Text matching ``skip_regex`` is left untouched.
        """
        return self._process(text, yo_only=False, skip_regex=skip_regex)
