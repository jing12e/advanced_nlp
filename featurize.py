
from sklearn.preprocessing import  OneHotEncoder
import numpy as np
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
silent = False

#todo:  Morphological features
def preprocess(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    processed_features = []
    for i,token in enumerate(doc):
        path_length = 0
        current_token = token
        while current_token.dep_ != 'ROOT':
            current_token = current_token.head
            path_length += 1
        processed_features.append({
            "TEXT": token.text,
            "LEMMA": token.lemma_,
            "POS": token.tag_,
            "DEP": token.dep_,
            "SHAPE": token.shape_,
            "ALPHA": token.is_alpha,
            "STOP": token.is_stop,
            "HEAD": token.head.text,
            "CHILD": [child.text for child in token.children],
            "PATH_LEN": path_length,
            "PREV_POS" : doc[i-1].tag_ if i > 0 else '',
            "PREV_DEP" : doc[i-1].dep_ if i > 0 else '',
            "CONST" : [t.text for t in token.head.subtree],
            "MORPH" : token.morph
        })
    return processed_features, [chunk.text for chunk in doc.noun_chunks]

def visualize(txt, n, plot_list=['POS', 'PATH_LEN', 'STOP', 'SHAPE']):
    plt.figure(figsize=(12, 20))
    current_plot = 1
    for feature_name, feature in txt[0].items():
        print(f'::{feature_name}::\n\n')
        text = [tok['TEXT'] for tok in txt]
        print(f'Plain text:\n\n"{" ".join(text[:n])}"')
        features = [str(token[feature_name]) for token in txt]
        print(f'Text {feature_name}:\n\n"{features[:n]}"')
        if feature_name in plot_list:
            ax = plt.subplot(2, 2, current_plot)
            ax.hist(features, bins=len(set(features)), color='skyblue')
            ax.set_title(f'{feature_name} Distribution')
            ax.set_ylabel('Frequency')
            current_plot += 1
    plt.tight_layout(pad=5.0)
    plt.show()

training_string = """The sun rose slowly over the horizon, casting its golden glow across the tranquil landscape. Birds chirped melodiously, welcoming the new day with their joyful songs. In the distance, a gentle breeze rustled through the leaves of the trees, carrying with it the promise of adventure."""
training_features = training_string
training_features,chunks = preprocess(training_features)


visualize(training_features, 10)
