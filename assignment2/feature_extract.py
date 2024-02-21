import spacy
import pandas as pd
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_md')
def extract_features(text):
    doc = nlp(text)

    features_list = []

    for token in doc:
        feature = {}

        # Word and lemma
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
            feature['next_lemma'] = 'None'

        # Previous lemma
        if token.i - 1 >= 0:
            previous_token = doc[token.i - 1]
            feature['previous_lemma'] = previous_token.lemma_
        else:
            feature['previous_lemma'] = 'None'

        # Morphological features
        if len(token.text) > 3:
            feature['suffix_3'] = token.text[-3:]
        else:
            feature['suffix_3'] = 'None'

        if len(token.text) > 2:
            feature['suffix_2'] = token.text[-2:]
        else:
            feature['suffix_2'] = 'None'

        # Semantic features
        synsets = wn.synsets(token.text)
        if synsets:
            first_synset = synsets[0]
            hypernyms = first_synset.hypernyms()
            if hypernyms:
                first_hypernym = hypernyms[0]
                feature['hypernym'] = first_hypernym.name().split('.')[0]
        else:
            feature['hypernym'] = 'None'

        # Append this feature to the list
        features_list.append(feature)

    return pd.DataFrame(features_list)


# Example usage
text = "Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of 0aim, near the Syrian border. "
features_df = extract_features(text)
print(features_df)
print(features_df['suffix_3'].tolist())