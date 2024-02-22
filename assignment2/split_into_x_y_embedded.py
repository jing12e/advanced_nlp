import pandas as pd
import numpy as np
import argparse

def process_args():
    """
    Processes user arguments from the command prompt. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, 
                        default='test_dataset.csv',
                        help="Input the relative path in the folder to the csv containing featurized non-embedded training/test data")
    
    parser.add_argument('--output_file', type=str, 
                        default='test_dataset',
                        help="Input the path of the output files")
    return parser.parse_args()

def load_glove(path):
    """
    Loads GloVe model of size --glove_dimension from directory.
    """
    with open(path, encoding = "utf-8") as f:
        lines = f.readlines()
    return {line.split()[0]:np.asarray(line.split()[1:]).astype(float) for line in lines}


def transform_csv(input_file, output_file):
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
    x = data.drop('argument', axis='columns').to_csv(f'{output_file}_x.csv')
    y = data['argument'].to_csv(f'{output_file}_y.csv')
    #x = np.array(data.drop(['argument'],axis='columns').astype(float))
    #y = np.array(data['argument'].astype(float))
    return x, y

#example usage:
args = process_args()
x,y = transform_csv(args.input_file, args.output_file)

    
