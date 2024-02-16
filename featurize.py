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
            token = doc[0]
            processed_features.append({
                "TEXT": token.text,
                "LEMMA": token.lemma_,
                "POS": token.pos_,
                "TAG": token.tag_,
                "DEP": token.dep_,
                "SHAPE": token.shape_,
                "ALPHA": token.is_alpha,
                "STOP": token.is_stop
            })
            new_labels.append(labels[i])
            i += 1
    return processed_features, new_labels

training_string = 'This is my favourite test string.'
label_string = 'Until I saw this fabulous string!'
training_features = [{'token' : i} for i in training_string.split(' ')]
gold_labels = [{'token' : i} for i in label_string.split(' ')]
training_features,gold_labels = preprocess(training_features, gold_labels, load_pickle=False)
print(f"training features:{training_features}")
