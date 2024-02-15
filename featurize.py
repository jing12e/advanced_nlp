#requires GloVe.6B.{n}D.txt to be present in path

from sklearn.preprocessing import  OneHotEncoder
import numpy as np
import spacy
from tqdm import tqdm
silent = False
def load_glove(path):
    """
    Loads GloVe model of size --glove_dimension from directory.
    """
    if not silent:
        print('Loading GloVe')
    with open(path, encoding = "utf-8") as f:
        lines = f.readlines()
    return {line.split()[0]:np.asarray(line.split()[1:]).astype(float) for line in lines}

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


def one_hotify(x_data, features=['POS', 'SHAPE', 'DEP'],encoder = False):
    """
    Takes in sklearn format data and returns dense one-hot arrays.
    It returns dense arrays to make integration with other data pipelines 
    easier and to ensure that neural networks can be trained on it.
    """
    data_by_category = [[feature[cat] for cat in features] for feature in x_data]
    if not encoder:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(data_by_category)
    else:
        encoder = encoder
    one_hot_data = encoder.transform(data_by_category).toarray()
    return one_hot_data, encoder

def map_embeddings_glove(glove, data, feature_list, ohv=False):
    """
    Takes in a GloVe model from load_glove and returns embedding vectors of data
    """
    #changes data[i]['TEXT'] to corresponding GloVe embedding of data[i]['TEXT'] or an array of zeroes for data[i]['TEXT'] in data
    # for x in data:
    #     x['TEXT'] = [glove[token] if token in glove else np.zeros_like(glove['the']) for token in x['TEXT']]
    data_out = []
    #Try to fit text from best fit to non-capital lemma. Otherwise all zeros.
    for i, x in enumerate(data):
        if x['TEXT'].lower() in glove:
            data_out.append(glove[x['TEXT'].lower()])
        elif x['LEMMA'] in glove:
            data_out.append(glove[x['LEMMA'].lower()])
        else: #Out-of-vocab all zeros
            data_out.append( np.zeros_like(np.array(glove['the'])))
    feature_list = feature_list.replace('LEMMA,','').split(',')
    to_concat, one_hot_vec = one_hotify(data,features=feature_list,encoder=ohv)
    output = [np.concatenate((np.array(data_out[i]).astype(float), to_concat[i]), axis=0) 
              for i, x in enumerate(data_out)]
    return output, one_hot_vec

training_string = 'This is my favourite test string.'
label_string = 'Until I saw this fabulous string!'
training_features = [{'token' : i} for i in training_string.split(' ')]
gold_labels = [{'token' : i} for i in label_string.split(' ')]
path = 'C:/Users/katko/NER/glove.6B/glove.6B.50d.txt'
glove = load_glove(path)
training_features,gold_labels = preprocess(training_features, gold_labels, load_pickle=False)
feature_list = 'POS', 'SHAPE', 'DEP'
embedded = map_embeddings_glove(glove, training_features, feature_list)
embedded
