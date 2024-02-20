import spacy
import pandas as pd
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_sm')
def extract_features(text):
    doc = nlp(text)

    features_list = []

    for token in doc:
        feature = {
            'word': token.text,
            'lemma': token.lemma_,
            'ner': token.ent_type_,
            'pos': token.pos_,
            'pos_head': token.head.pos_,
            'dependency_label': token.dep_,
            'next_lemma': 'None',
            'previous_lemma': 'None',
            'suffix_3': 'None',
            'suffix_2': 'None',
            'hypernym': 'None'
        }

        # Next lemma feature
        if token.i + 1 < len(doc):
            next_token = doc[token.i + 1]
            feature['next_lemma'] = next_token.lemma_

        # Previous lemma feature
        if token.i - 1 >= 0:
            previous_token = doc[token.i - 1]
            feature['previous_lemma'] = previous_token.lemma_

        # Morphological features
        if len(token.text) > 3:
            feature['suffix_3'] = token.text[-3:]
        if len(token.text) > 2:
            feature['suffix_2'] = token.text[-2:]

        # Semantic features
        synsets = wn.synsets(token.text)
        if synsets:
            first_synset = synsets[0]
            hypernyms = first_synset.hypernyms()
            if hypernyms:
                first_hypernym = hypernyms[0]
                feature['hypernym'] = first_hypernym.name().split('.')[0]


        features_list.append(feature)

    return pd.DataFrame(features_list)


# Example usage
text = "Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of 0aim, near the Syrian border. "
features_df = extract_features(text)
print(features_df)