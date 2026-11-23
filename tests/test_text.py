import pytest

from ruaccent.text_postprocessor import fix_capital, transfer_stress
from ruaccent.text_preprocessor import TextPreprocessor


@pytest.mark.parametrize("text", [
    "",
    "   ",
    "Одно предложение.",
    "Первое. Второе!  Третье?\n\nЧетвёртое…   ",
    "  ведущие пробелы. и т.д. и т.п. Конец",
])
def test_split_by_sentences_is_lossless(text):
    assert "".join(TextPreprocessor.split_by_sentences(text)) == text


def test_analyze_maps_tokens_back_to_original():
    sent = TextPreprocessor.analyze("Слово «пр+ивет» 😀 и 15% x_y.")
    by_text = {t.text: t for t in sent.tokens}
    assert "+" not in sent.model_text and "😀" not in sent.model_text
    assert by_text["привет"].original == "пр+ивет"
    assert not by_text["привет"].is_clean
    assert by_text["Слово"].is_clean
    for t in sent.tokens:
        assert sent.original[t.orig_start:t.orig_end] == t.original


def test_free_standing_dash_is_kept_for_model_input():
    sent = TextPreprocessor.analyze("Он - наш, кто-то")
    words = [t.text for t in sent.tokens]
    assert TextPreprocessor.join_for_model(words) == "Он - наш, кто-то"


@pytest.mark.parametrize("word, stressed, expected", [
    ("Замок", "зам+ок", "Зам+ок"),
    ("ЗАМОК", "з+амок", "З+АМОК"),
    ("ёлки", "+ёлки", "+ёлки"),
    ("слово", "другое+", "другое+"),  # letters differ: keep dictionary form
])
def test_transfer_stress(word, stressed, expected):
    assert transfer_stress(word, stressed) == expected


def test_fix_capital():
    assert fix_capital("Елка", "ёлка") == "Ёлка"
    assert fix_capital("abc", "abcd") == "abcd"
