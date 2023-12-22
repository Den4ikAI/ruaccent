# -*- coding: utf-8 -*-
import json
import gzip
DEBUG = False

def decode_pos(pos):
    if pos in ['ДЕЕПРИЧАСТИЕ', 'ГЛАГОЛ', 'ИНФИНИТИВ']:
        return 'ГЛАГОЛ'
    else:
        return pos


class Lemmatizer(object):
    def __init__(self):
        pass

    def load(self, dict_path):
        self.forms = json.load(gzip.open(dict_path+"/forms.json.gz","r"))
        self.forms2 = json.load(gzip.open(dict_path+"/forms2.json.gz","r"))
        self.special_lemmas = json.load(gzip.open(dict_path+"/spec.json.gz","r"))
        self.key2transducer = json.load(gzip.open(dict_path+"/key2transducer.json.gz","r"))

    def get_lemma(self, word):
        if word in self.forms:
            return self.forms[word]
        elif word in self.forms2:
            return self.forms2[word][0]
        elif word in self.special_lemmas:
            return self.special_lemmas[word]
        else:
            return word

    def decode_pos_tags(self, pos_tags):
        stags1 = []
        part_of_speech = 'unk'
        short_tag_index = -1
        for tag in pos_tags.split('|'):
            if tag == 'NOUN':
                part_of_speech = 'СУЩЕСТВИТЕЛЬНОЕ'
            elif tag == 'VERB':
                part_of_speech = 'ГЛАГОЛ'
            elif tag == 'ADJ':
                part_of_speech = 'ПРИЛАГАТЕЛЬНОЕ'
                stags1.append(('КРАТКИЙ', '0'))
                short_tag_index = 0
            elif tag == 'ADV':
                part_of_speech = 'НАРЕЧИЕ'
            elif tag == 'PRON':
                part_of_speech = 'МЕСТОИМЕНИЕ'
            elif tag == 'ADP':
                part_of_speech = 'ПРЕДЛОГ'
            elif tag == 'CONJ':
                part_of_speech = 'СОЮЗ'
            elif tag == 'PART':
                part_of_speech = 'ЧАСТИЦА'
            elif tag == 'PUNCT':
                part_of_speech = 'ПУНКТУАТОР'
            elif '=' in tag:
                if part_of_speech == 'СУЩЕСТВИТЕЛЬНОЕ':
                    if tag == 'Case=Nom':
                        stags1.append(('ПАДЕЖ', 'ИМ'))
                    elif tag == 'Case=Acc':
                        stags1.append(('ПАДЕЖ', 'ВИН'))
                    elif tag == 'Case=Dat':
                        stags1.append(('ПАДЕЖ', 'ДАТ'))
                    elif tag == 'Case=Ins':
                        stags1.append(('ПАДЕЖ', 'ТВОР'))
                    elif tag == 'Case=Prep':
                        stags1.append(('ПАДЕЖ', 'ПРЕДЛ'))
                    elif tag == 'Case=Loc':
                        stags1.append(('ПАДЕЖ', 'ПРЕДЛ'))  # 03-02-2020 'МЕСТ'
                    elif tag == 'Case=Gen':
                        stags1.append(('ПАДЕЖ', 'РОД'))
                    elif tag == 'Case=Voc':
                        stags1.append(('ПАДЕЖ', 'ЗВАТ'))
                    elif tag == 'Number=Sing':
                        stags1.append(('ЧИСЛО', 'ЕД'))
                    elif tag == 'Number=Plur':
                        stags1.append(('ЧИСЛО', 'МН'))
                    elif tag == 'Gender=Masc':
                        stags1.append(('РОД', 'МУЖ'))
                    elif tag == 'Gender=Fem':
                        stags1.append(('РОД', 'ЖЕН'))
                    elif tag == 'Gender=Neut':
                        stags1.append(('РОД', 'СР'))
                    else:
                        if DEBUG:
                            print('неизвестный тэг "{}"'.format(tag))
                        #raise NotImplementedError()
                elif part_of_speech == 'ПРИЛАГАТЕЛЬНОЕ':
                    if tag == 'Case=Nom':
                        stags1.append(('ПАДЕЖ', 'ИМ'))
                    elif tag == 'Case=Acc':
                        stags1.append(('ПАДЕЖ', 'ВИН'))
                    elif tag == 'Case=Dat':
                        stags1.append(('ПАДЕЖ', 'ДАТ'))
                    elif tag == 'Case=Ins':
                        stags1.append(('ПАДЕЖ', 'ТВОР'))
                    elif tag == 'Case=Prep':
                        stags1.append(('ПАДЕЖ', 'ПРЕДЛ'))
                    elif tag == 'Case=Loc':
                        stags1.append(('ПАДЕЖ', 'ПРЕДЛ')) # 03-02-2020 'МЕСТ'
                    elif tag == 'Case=Gen':
                        stags1.append(('ПАДЕЖ', 'РОД'))
                    elif tag == 'Number=Sing':
                        stags1.append(('ЧИСЛО', 'ЕД'))
                    elif tag == 'Number=Plur':
                        stags1.append(('ЧИСЛО', 'МН'))
                    elif tag == 'Gender=Masc':
                        stags1.append(('РОД', 'МУЖ'))
                    elif tag == 'Gender=Fem':
                        stags1.append(('РОД', 'ЖЕН'))
                    elif tag == 'Gender=Neut':
                        stags1.append(('РОД', 'СР'))
                    elif tag == 'Degree=Cmp':
                        stags1.append(('СТЕПЕНЬ', 'СРАВН'))
                    elif tag == 'Degree=Pos':
                        stags1.append(('СТЕПЕНЬ', 'АТРИБ'))
                    elif tag in ('Variant=Short', 'Variant=Brev'):
                        stags1[short_tag_index] = ('КРАТКИЙ', '1')
                    else:
                        if DEBUG:
                            print('неизвестный тэг "{}"'.format(tag))
                            #raise NotImplementedError()
                elif part_of_speech == 'ГЛАГОЛ':
                    #print("#"+tag+"#")
                    if tag == 'Number=Sing':
                        stags1.append(('ЧИСЛО', 'ЕД'))
                    elif tag == 'Number=Plur':
                        stags1.append(('ЧИСЛО', 'МН'))
                    elif tag == 'Gender=Masc':
                        stags1.append(('РОД', 'МУЖ'))
                    elif tag == 'Gender=Fem':
                        stags1.append(('РОД', 'ЖЕН'))
                    elif tag == 'Gender=Neut':
                        stags1.append(('РОД', 'СР'))
                    elif tag == 'Mood=Ind':
                        stags1.append(('НАКЛОНЕНИЕ', 'ИЗЪЯВ'))
                    elif tag == 'Mood=Imp':
                        stags1.append(('НАКЛОНЕНИЕ', 'ПОБУД'))
                    elif tag == 'Tense=Past':
                        stags1.append(('ВРЕМЯ', 'ПРОШЕДШЕЕ'))
                    elif tag == 'Tense=Fut':
                        stags1.append(('ВРЕМЯ', 'БУДУЩЕЕ'))
                    elif tag == 'Tense=Notpast':
                        stags1.append(('ВРЕМЯ', 'НАСТОЯЩЕЕ'))
                    elif tag == 'Tense=Pres':
                        stags1.append(('ВРЕМЯ', 'НАСТОЯЩЕЕ'))
                    elif tag == 'Person=1':
                        stags1.append(('ЛИЦО', '1'))
                    elif tag == 'Person=2':
                        stags1.append(('ЛИЦО', '2'))
                    elif tag == 'Person=3':
                        stags1.append(('ЛИЦО', '3'))
                    elif tag == 'VerbForm=Fin':
                        pass
                    elif tag == 'VerbForm=Inf':
                        pass
                    elif tag == 'VerbForm=Conv':
                        pass
                    else:
                        if DEBUG:
                            print('неизвестный тэг "{}"'.format(tag))
                        #raise RuntimeError(msg)
                elif part_of_speech == 'НАРЕЧИЕ':
                    if tag == 'Degree=Pos':
                        stags1.append(('СТЕПЕНЬ', 'АТРИБ'))
                    elif tag == 'Degree=Cmp':
                        stags1.append(('СТЕПЕНЬ', 'СРАВН'))
                    else:
                        if DEBUG:
                            print('неизвестный тэг "{}"'.format(tag))
                    pass

        return part_of_speech, stags1

    def get_lemma2(self, word, pos_tags):
        part_of_speech, decoded_tags = self.decode_pos_tags(pos_tags)

        nword = word.lower().replace('ё', 'е')

        if nword in self.special_lemmas:
            return self.special_lemmas[nword], part_of_speech, decoded_tags

        if nword in self.forms:
            lemma = self.forms[nword]
            return lemma, part_of_speech, decoded_tags
        elif nword in self.forms2:
            if part_of_speech == 'СУЩЕСТВИТЕЛЬНОЕ':
                # Для существительных учитываем падеж.
                required_case = None
                for tag in decoded_tags:
                    if tag[0] == 'ПАДЕЖ':
                        required_case = tag[1]
                        break

                for lemma, lemma_part_of_speech, tag in self.forms2[nword]:
                    if lemma_part_of_speech == part_of_speech and tag == required_case:
                        return lemma, part_of_speech, decoded_tags
            else:
                for lemma, lemma_part_of_speech, tags in self.forms2[nword]:
                    if lemma_part_of_speech == part_of_speech:
                        return lemma, part_of_speech, decoded_tags
        elif len(word) > 4:
            ending = nword[-4:]
            key = ending + '|' + part_of_speech
            if key in self.key2transducer:
                transducer = self.key2transducer[key]
                if transducer[0] > 0:
                    lemma = word[:-transducer[0]] + transducer[1]
                else:
                    lemma = word + transducer[1]

                return lemma.lower(), part_of_speech, decoded_tags

        return nword, part_of_speech, decoded_tags

    def lemmatize(self, tagged_words):
        return [(word, tags,)+tuple(self.get_lemma2(word, tags)) for (word, tags) in tagged_words]
