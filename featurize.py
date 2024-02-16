#requires GloVe.6B.{n}D.txt to be present in path

from sklearn.preprocessing import  OneHotEncoder
import numpy as np
import spacy
from tqdm import tqdm
silent = False

def preprocess(features, labels, load_pickle=False, pickle_name='training'):
    """
    Preprocesses tokenized data to generate more features using the spacy library or saves/loads previously 
    featurized data from a pickle file.
    """
    #use load pickle=True to load a previously created file.
    
    nlp = spacy.load("en_core_web_sm")
    processed_features = []
    new_labels = []
    token_texts = [feature['token'] for feature in features]
    docs = nlp.pipe(token_texts)
    i = 0
    #labels are changed here as well to match the slightly different tokenization method the featurization model uses
    for doc in tqdm(docs, disable=silent):
        if i < len(features):
            for token in doc:
                current_token = token
                path_length = 0
                print(token.dep_)
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
            new_labels.append(labels[i])
            i += 1
    return processed_features, new_labels

training_string = """The sun rose slowly over the horizon, casting its golden glow across the tranquil landscape. Birds chirped melodiously, welcoming the new day with their joyful songs. In the distance, a gentle breeze rustled through the leaves of the trees, carrying with it the promise of adventure."""
label_string = training_string
training_features = [{'token' : i} for i in training_string.split(' ')]
gold_labels = [{'token' : i} for i in label_string.split(' ')]
training_features,gold_labels = preprocess(training_features, gold_labels, load_pickle=False)
print(f"training features:\n{[(token['HEAD'],token['CHILD'],token['PATH_LEN']) for token in training_features]}")
