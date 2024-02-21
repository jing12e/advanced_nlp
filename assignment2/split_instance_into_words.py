import ast
import pandas as pd
import pickle

def picklify(pickle_file,not_so_pickle):
    with open(pickle_file, 'wb') as p:
        pickle.dump(not_so_pickle, p)


def split_instance_into_words(instance):
    words_instances = []

    tokens = ast.literal_eval(instance['word'])
    predicates = instance['predicate']
    predicate_positions = instance['predicate_position']
    arguments = ast.literal_eval(instance['arguments'])
    lemmas = ast.literal_eval(instance['lemma'])
    ners = ast.literal_eval(instance['ner'])
    poss = ast.literal_eval(instance['pos'])
    pos_heads = ast.literal_eval(instance['pos_head'])
    dependency_labels = ast.literal_eval(instance['dependency_label'])
    next_lemmas = ast.literal_eval(instance['next_lemma'])
    previous_lemmas = ast.literal_eval(instance['previous_lemma'])
    suffix_3 = ast.literal_eval(instance['suffix_3'])
    suffix_2 = ast.literal_eval(instance['suffix_2'])


    for i in range(len(tokens)):
        word_instance = {
            'word': tokens[i],
            'predicate': predicates,
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
instances = pd.read_csv('test_dataset.csv')
results = []
for i, instance in instances.iterrows():
    results.append(dict(instance))
picklify('test_dataset_by_words.pickle',results)



