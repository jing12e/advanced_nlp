import ast
from feature_extract import load_glove, map_embedding


def split_instance_into_words(instance):
    words_instances = []

    tokens = instance['word']
    predicates = instance['predicate']
    predicate_positions = instance['predicate_position']
    arguments = instance['arguments']
    lemmas = instance['lemma']
    ners = instance['ner']
    poss = instance['pos']
    path_len = instance['path_len']
    pos_heads = instance['pos_head']
    dependency_labels = instance['dependency_label']
    next_lemmas = instance['next_lemma']
    previous_lemmas = instance['previous_lemma']
    suffix_3 = instance['suffix_3']
    suffix_2 = instance['suffix_2']


    for i in range(len(tokens)):
        word_instance = {
            'word': tokens[i],
            'path_len':path_len[i],
            'predicate': predicates if i == int(predicate_positions) - 1 else None,
            'predicate_position': predicate_positions,
            'argument': arguments[i],
            'lemma': lemmas[i],
            'ner': ners[i],
            'pos': poss[i],
            'pos_head': pos_heads[i],
            'dependency_label': dependency_labels[i],
            'next_lemma': next_lemmas[i],
            'previous_lemma': previous_lemmas[i],
            'suffix_3': suffix_3[i],
            'suffix_2': suffix_2[i]
        }
        words_instances.append(word_instance)

    return words_instances



instance = {
    "word": "['What', 'if', 'Google', 'Morphed', 'Into', 'GoogleOS', '?']",
    "predicate": "morph.01",
    "predicate_position": "4",
    "arguments": "['_', '_', 'ARG1', 'V', '_', 'ARG2', '_']",
    "lemma": "['what', 'if', 'Google', 'morph', 'into', 'GoogleOS', '?']",
    "ner": "['', '', '', '', '', '', '']",
    "pos": "['PRON', 'SCONJ', 'PROPN', 'VERB', 'ADP', 'PROPN', 'PUNCT']",
    "pos_head": "['PRON', 'VERB', 'VERB', 'PRON', 'VERB', 'ADP', 'PRON']",
    "dependency_label": "['ROOT', 'mark', 'nsubj', 'advcl', 'prep', 'pobj', 'punct']",
    "next_lemma": "['if', 'Google', 'morph', 'into', 'GoogleOS', '?', 'None']",
    "previous_lemma": "['None', 'what', 'if', 'Google', 'morph', 'into', 'GoogleOS']",
    "suffix_3": "['hat', 'None', 'gle', 'hed', 'nto', 'eOS', 'None']",
    "suffix_2": "['at', 'None', 'le', 'ed', 'to', 'OS', 'None']",
    "hypernym": "['None', 'None', nan, 'change', 'None', 'None', 'None']"
}

#glove = load_glove('glove.6B.300d.txt')
#words_instances = split_instance_into_words(instance)
#for word_instance in words_instances:
#    print(word_instance)


