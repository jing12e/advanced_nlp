
from sklearn.preprocessing import  OneHotEncoder
import numpy as np
import spacy
from tqdm import tqdm
silent = False


def preprocess(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    processed_features = []
    for token in doc:
        path_length = 0
        current_token = token
        while current_token.dep_ != 'ROOT':
            current_token = current_token.head
            path_length += 1
        processed_features.append({
            "TEXT": token.text,
            "LEMMA": token.lemma_,
            "POS": token.pos_,
            "TAG": token.tag_,
            "DEP": token.dep_,
            "SHAPE": token.shape_,
            "ALPHA": token.is_alpha,
            "STOP": token.is_stop,
            "HEAD": token.head.text,
            "CHILD": [child.text for child in token.children],
            "PATH_LEN": path_length
        })
    return processed_features

training_string = """The sun rose slowly over the horizon, casting its golden glow across the tranquil landscape. Birds chirped melodiously, welcoming the new day with their joyful songs. In the distance, a gentle breeze rustled through the leaves of the trees, carrying with it the promise of adventure."""
training_features = training_string
training_features = preprocess(training_features)
print(training_features)
#print(f"training features:\n{[(token['HEAD'],token['CHILD'],token['PATH_LEN']) for token in training_features]}")
