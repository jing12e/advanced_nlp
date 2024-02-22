import pandas as pd
import numpy as np
import argparse


def load_glove(path):
    """
    Loads and returns GloVe model
    """
    with open(path, encoding = "utf-8") as f:
        lines = f.readlines()
    return {line.split()[0]:np.asarray(line.split()[1:]).astype(float) for line in lines}


def transform_csv(input_file):
    """
    Loads csv  of features and labels, splits them into x & y sets and applies embeddings to tokens
    """
    data =  pd.read_csv(input_file)
    glove = load_glove('glove.6B.50d.txt')
    def map_embedding(word):
        """
        Maps a GloVe embedding to an input word
        """
        if word in glove:    
            return glove[word.lower()]
        else: #Out-of-vocab all zeros
            return np.zeros_like(np.array(glove['the']))
    data['word'] = data['word'].apply(map_embedding)
    x = data.drop('argument', axis='columns')
    y = data['argument']
    #x = np.array(data.drop(['argument'],axis='columns').astype(float))
    #y = np.array(data['argument'].astype(float))
    return x, y

#example usage:
x,y = transform_csv('test_dataset.csv')
x.to_csv('test_dataset_x.csv')
y.to_csv('test_dataset_y.csv')
    
