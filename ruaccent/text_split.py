import re
import logging
from typing import Set, Tuple, List

SENTENCE_SPLITTER = re.compile(r'[^\.?!…]+[\.?!…]*["»“]*')

LAST_WORD_PATTERN = re.compile(r'(?:\b|\d)([a-zа-я]+)\.$', re.IGNORECASE)
FIRST_WORD_PATTERN = re.compile(r'^\W*(\w+)')
ENDS_WITH_ONE_LETTER_LAT_AND_DOT_PATTERN = re.compile(r'(\d|\W|\b)([a-zA-Z])\.$')
HAS_DOT_INSIDE_PATTERN = re.compile(r'[\w]+\.[\w]+\.$', re.IGNORECASE)
INITIALS_PATTERN = re.compile(r'(\W|\b)([A-ZА-Я]{1})\.$')
ONLY_RUS_CONSONANTS_PATTERN = re.compile(r'^[бвгджзйклмнпрстфхцчшщ]{1,4}$', re.IGNORECASE)
STARTS_WITH_EMPTYNESS_PATTERN = re.compile(r'^\s+')
ENDS_WITH_EMOTION_PATTERN = re.compile(r'[!?…]|\.{2,}\s?[)"«»,“]?$')
STARTS_WITH_LOWER_PATTERN = re.compile(r'^\s*[–-—-("«]?\s*[a-zа-я]')
STARTS_WITH_DIGIT_PATTERN = re.compile(r'^\s*\d')
NUMERATION_PATTERN = re.compile(r'^\W*[IVXMCL\d]+\.$')
PAIRED_SHORTENING_IN_THE_END_PATTERN = re.compile(r'\b(\w+)\. (\w+)\.\W*$')

JOIN = 0
MAYBE = 1
SPLIT = 2

JOINING_SHORTENINGS = {
    'mr', 'mrs', 'ms', 'dr', 'vs', 'англ', 'итал', 'греч', 'евр', 'араб', 'яп', 'слав', 'кит',
    'тел', 'св', 'ул', 'устар', 'им', 'г', 'см', 'д', 'стр', 'корп', 'пл', 'пер', 'сокр', 'рис'
}

SHORTENINGS = {
    'co', 'corp', 'inc', 'авт', 'адм', 'барр', 'внутр', 'га', 'дифф', 'дол', 'долл', 'зав', 'зам', 'искл',
    'коп', 'корп', 'куб', 'лат', 'мин', 'о', 'обл', 'обр', 'прим', 'проц', 'р', 'ред', 'руб', 'рус', 'русск',
    'сан', 'сек', 'тыс', 'эт', 'яз', 'гос', 'мн', 'жен', 'муж', 'накл', 'повел', 'букв', 'шутл', 'ед'
}

PAIRED_SHORTENINGS = {('и', 'о'), ('т', 'е'), ('т', 'п'), ('у', 'е'), ('н', 'э')}


def split_sentences(text: str) -> List[str]:
    return [x.strip() for x in SENTENCE_SPLITTER.findall(text)]


def is_sentence_end(left: str, right: str,
                    shortenings: Set[str],
                    joining_shortenings: Set[str],
                    paired_shortenings: Set[Tuple[str, str]]) -> int:
    if not STARTS_WITH_EMPTYNESS_PATTERN.match(right):
        return JOIN

    if HAS_DOT_INSIDE_PATTERN.search(left):
        return JOIN

    left_last_word = LAST_WORD_PATTERN.search(left)
    lw = ' '
    if left_last_word:
        lw = left_last_word.group(1)

        if lw.lower() in joining_shortenings:
            return JOIN

        if ONLY_RUS_CONSONANTS_PATTERN.search(lw) and lw[-1].islower():
            return MAYBE

    pse = PAIRED_SHORTENING_IN_THE_END_PATTERN.search(left)
    if pse:
        s1, s2 = pse.groups()
        if (s1, s2) in paired_shortenings:
            return MAYBE

    right_first_word = FIRST_WORD_PATTERN.match(right)
    if right_first_word:
        rw = right_first_word.group(1)
        if (lw, rw) in paired_shortenings:
            return MAYBE

    if ENDS_WITH_EMOTION_PATTERN.search(left) and STARTS_WITH_LOWER_PATTERN.match(right):
        return JOIN

    initials = INITIALS_PATTERN.search(left)
    if initials:
        border, _ = initials.groups()
        if (border or ' ') not in "°'":
            return JOIN

    if lw.lower() in shortenings:
        return MAYBE

    last_letter = ENDS_WITH_ONE_LETTER_LAT_AND_DOT_PATTERN.search(left)
    if last_letter:
        border, _ = last_letter.groups()
        if (border or ' ') not in "°'":
            return MAYBE
    if NUMERATION_PATTERN.match(left):
        return JOIN
    return SPLIT


def split_by_sentences(text: str,
                      shortenings: Set[str] = SHORTENINGS,
                      joining_shortenings: Set[str] = JOINING_SHORTENINGS,
                      paired_shortenings: Set[Tuple[str, str]] = PAIRED_SHORTENINGS) -> List[str]:
    sentences = []
    sents = split_sentences(text)
    si = 0
    processed_index = 0
    sent_start = 0
    while si < len(sents):
        s = sents[si]
        span_start = text[processed_index:].index(s) + processed_index
        span_end = span_start + len(s)
        processed_index += len(s)

        si += 1

        send = is_sentence_end(text[sent_start: span_end], text[span_end:],
                               shortenings, joining_shortenings, paired_shortenings)
        if send == JOIN:
            continue

        if send == MAYBE:
            if STARTS_WITH_LOWER_PATTERN.match(text[span_end:]):
                continue
            if STARTS_WITH_DIGIT_PATTERN.match(text[span_end:]):
                continue

        if not text[sent_start: span_end].strip():
            print(text)
        sentences.append(text[sent_start: span_end].strip())
        sent_start = span_end
        processed_index = span_end

    if sent_start != len(text):
        if text[sent_start:].strip():
            sentences.append(text[sent_start:].strip())
    return sentences

