"""End-to-end tests. They download ~500 MB of models on the first run:

    RUACCENT_TEST_MODELS=1 pytest
"""
import os
import re

import pytest

pytestmark = [
    pytest.mark.models,
    pytest.mark.skipif(not os.environ.get("RUACCENT_TEST_MODELS"), reason="set RUACCENT_TEST_MODELS=1"),
]

OMOGRAPH_MODEL = os.environ.get("RUACCENT_TEST_OMOGRAPH", "tiny2.1")


@pytest.fixture(scope="module")
def acc():
    from ruaccent import RUAccent

    a = RUAccent()
    a.load(omograph_model_size=OMOGRAPH_MODEL, use_dictionary=True,
           workdir=os.environ.get("RUACCENT_TEST_WORKDIR"))
    return a


def plain(text):
    return re.sub(r"\+(?=[аеёиоуыэюяАЕЁИОУЫЭЮЯ])", "", text)


@pytest.mark.parametrize("text", [
    "Цена: 1 000 руб.; скидка 10%… Звоните: +7 (999) 123-45-67.",
    "Привет 😀 как дела?",
    "№ 5 § 3 @user #тег a_b x*y=z <tag> 50°C 2−1",
    'Он сказал: "Привет" - и ушёл. Я  пойду – сказал он.',
    "   Пробелы   в начале и   в конце.   \n\n",
    "...",
    "",
])
def test_text_is_preserved(acc, text):
    assert plain(acc.process_all(text)).replace("ё", "е") == text.replace("ё", "е")
    assert acc.process_yo(text).replace("ё", "е") == text.replace("ё", "е")


def test_unknown_word_does_not_crash(acc):
    # https://github.com/Den4ikAI/ruaccent/issues/19
    assert acc.process_all("Пьерами.") == "Пь+ерами."


def test_homographs(acc):
    assert acc.process_all("на двери висит замок.") == "на двер+и вис+ит зам+ок."
    assert acc.process_all("Замок стоял на горе.").startswith("З+амок")


@pytest.mark.parametrize("text", ["белки ели орехи.", "замок стоял на горе.", "все елки зеленые."])
def test_caps_do_not_change_result(acc, text):
    expected = acc.process_all(text)
    assert acc.process_all(text.upper()) == expected.upper()
    assert acc.process_all(text.capitalize()) == expected[0].upper() + expected[1:]


def test_very_long_word(acc):
    assert acc.process_all("а" * 80 + " мама") == "а" * 80 + " м+ама"


def test_yo(acc):
    assert acc.process_yo("Все елки зеленые.") == "Все ёлки зелёные."


def test_user_stress_is_kept(acc):
    assert "«пр+ивет»" in acc.process_all("Слово «пр+ивет» тут.")


def test_one_stress_per_word(acc):
    out = acc.process_all("Табы между словами Байденом и Зеленским.")
    assert all(w.count("+") <= 1 for w in out.split())


def test_custom_dict_and_skip_regex():
    from ruaccent import RUAccent

    a = RUAccent()
    a.load(omograph_model_size=OMOGRAPH_MODEL, custom_dict={"привет": "пр+ивет"},
           workdir=os.environ.get("RUACCENT_TEST_WORKDIR"), tiny_mode=True)
    assert a.process_all("привет <b>мама</b>", skip_regex=r"<[^>]+>") == "пр+ивет <b>м+ама</b>"
