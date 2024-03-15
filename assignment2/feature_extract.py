import spacy
import pandas as pd
from nltk.corpus import wordnet as wn
import numpy as np

nlp = spacy.load('en_core_web_md')

def load_glove(path):
    """
    Loads GloVe model of size --glove_dimension from directory.
    """
    with open(path, encoding = "utf-8") as f:
        lines = f.readlines()
    return {line.split()[0]:np.asarray(line.split()[1:]).astype(float) for line in lines}

def map_embedding(glove, word):
    """
    Maps a GloVe embedding to an input word
    """
    if word in glove:    
        return glove[word.lower()]
    else: #Out-of-vocab all zeros
        return np.zeros_like(np.array(glove['the']))

def extract_features_OLD(text):
    doc = nlp(text)
    features_list = []

    for token in doc:
        feature = {}
        
        # Path length
        path_length = 0
        current_token = token
        while current_token.dep_ != 'ROOT':
            current_token = current_token.head
            path_length += 1
        feature['path_len'] = path_length

        # Word and lemma
        #feature['word'] = map_embedding(glove,token.text)
        feature['word'] = token.text
        feature['lemma'] = token.lemma_

        # Named entity recognition (NER)
        feature['ner'] = token.ent_type_

        # Part-of-speech (POS) tagging
        feature['pos'] = token.pos_

        # Head POS
        feature['pos_head'] = token.head.pos_

        # Dependency label
        feature['dependency_label'] = token.dep_

        # Next lemma
        if token.i + 1 < len(doc):
            next_token = doc[token.i + 1]
            feature['next_lemma'] = next_token.lemma_
        else:
            feature['next_lemma'] = None

        # Previous lemma
        if token.i - 1 >= 0:
            previous_token = doc[token.i - 1]
            feature['previous_lemma'] = previous_token.lemma_
        else:
            feature['previous_lemma'] = None

        # Morphological features
        if len(token.text) > 3:
            feature['suffix_3'] = token.text[-3:]
        else:
            feature['suffix_3'] = None

        if len(token.text) > 2:
            feature['suffix_2'] = token.text[-2:]
        else:
            feature['suffix_2'] = None

        # Semantic features
        synsets = wn.synsets(token.text)
        if synsets:
            first_synset = synsets[0]
            hypernyms = first_synset.hypernyms()
            if hypernyms:
                first_hypernym = hypernyms[0]
                feature['hypernym'] = first_hypernym.name().split('.')[0]
        else:
            feature['hypernym'] = None

        # Append this feature to the list
        features_list.append(feature)

    return pd.DataFrame(features_list)

def extract_features(text, max_len):
    doc = nlp(text)
    features_list = []
    pred_mask = [0] * max_len
    for token in doc:
        feature = {}

        # Word and lemma
        #feature['word'] = map_embedding(glove,token.text)
        feature['word'] = token.text
        feature['lemma'] = token.lemma_
        
        # Part-of-speech (POS) tagging
        feature['pos'] = token.pos_
        
        # Dependency label
        feature['basic_dep'] = token.dep_
        
        # Head 
        feature['head'] = token.head.text
        
        # Children 
        mask  = [0] * max_len
        for child in token.children:
            mask[child.i] = 1
        feature['children'] = mask
        
        # Predicate subtree
        if token.pos_ == 'VERB':
            for subtree_token in token.subtree:
                pred_mask[subtree_token.i] = 1
        feature['pred_subtree'] = pred_mask
        
        
        # shape
        feature['shape'] = token.shape_

        # Named entity recognition (NER)
        feature['ner'] = token.ent_type_

        # Next pos
        if token.i + 1 < len(doc):
            next_token = doc[token.i + 1]
            feature['next_pos'] = next_token.lemma_
        else:
            feature['next_pos'] = None

        # Previous pos
        if token.i - 1 >= 0:
            previous_token = doc[token.i - 1]
            feature['previous_pos'] = previous_token.lemma_
        else:
            feature['previous_pos'] = None

        # suffix
        if len(token.text) > 3:
            feature['suffix_3'] = token.text[-3:]
        else:
            feature['suffix_3'] = None

        # Semantic features
        synsets = wn.synsets(token.text)
        if synsets:
            first_synset = synsets[0]
            hypernyms = first_synset.hypernyms()
            if hypernyms:
                first_hypernym = hypernyms[0]
                feature['hypernym'] = first_hypernym.name().split('.')[0]
        else:
            feature['hypernym'] = None

        # Append this feature to the list
        features_list.append(feature)

    return pd.DataFrame(features_list)

# Example usage
#glove = load_glove('glove.6B.300d.txt')

text = "Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of 0aim, near the Syrian border. "
if __name__ == 'main':
    features_df = extract_features(text)
#[print(features_df[col]) for col in features_df.columns]
#[print(features_df[col].astype) for col in features_df.columns]

    print(features_df)
    print(features_df['suffix_3'].tolist())
