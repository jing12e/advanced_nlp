from sklearn.linear_model import LogisticRegression
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import pickle

def load_model(pickle):
    '''
    Input: string name with .pickle extension. Must be present in working directory. Check with 'pwd' command
    Loads pretrained model variable from pickle file at filepath(pickle).
    '''
    with open(pickle, 'rb') as p:
        return pickle.load(p)

def picklify(pickles,not_so_pickles):
    '''
    Saves a model or variable to a pickle file 
    '''
    with open(pickles, 'wb') as p:
        pickle.dump(not_so_pickles, p)

def fit_encoder(data, encoder=None):
    data = data.reshape(-1, 1)
    if not encoder:
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(data)
    return encoder

def encode(encoder, data):
    data = data.reshape(-1, 1)
    return encoder.transform(data).toarray()

def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train', type=str,
                        default='train_dataset_x.csv',
                        help="Input the relative path in the folder to the csv containing featurizes for the training set")

    parser.add_argument('--y_train', type=str,
                        default='train_dataset_y.csv',
                        help="Input the relative path in the folder to the csv containing the y labels for the training set")

    parser.add_argument('--x_test', type=str,
                        default='test_dataset_x.csv',
                        help="Input the relative path in the folder to the csv containing featurizes for the test set")

    parser.add_argument('--y_test', type=str,
                        default='test_dataset_y.csv',
                        help="Input the relative path in the folder to the csv containing the y labels for the test set")

    parser.add_argument('--output_file', type=str,
                        default='output/inference.txt',
                        help="Input the path of the output files")
    return parser.parse_args()

def create_classifier(x_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    return model

def inference(model, x_test, output_file):
    predictions = model.predict(x_test)
    with open(output_file, 'w') as file:
        [file.write(f"{prediction}\n") for prediction in predictions]
    return predictions

def evaluate(predictions, true_y):
    report = classification_report(true_y, predictions, output_dict=True)
    conf_matrix = confusion_matrix(true_y, predictions)
    


# Example usage:
args = process_args()
x_train, y_train = pd.read_csv(args.x_train), pd.read_csv(args.y_train)
skiplist = ['word','predicate_position', 'next_lemma','previous_lemma', 'lemma','path_len']
t = x_train.at[0,'predicate']

x_train['predicate'] = pd.Series([1 if isinstance(pred, str) else 0 for pred in x_train['predicate']])

encoder_dict = {feature: fit_encoder(x_train[feature].values) for feature in x_train.columns[1:] if feature not in skiplist}
encoded_x_train = []
dict_vec = DictVectorizer()
for column in x_train.columns[1:]:
    if column not in skiplist:
        encoded_x_train.append(encode(encoder_dict[column], x_train[column].values))
    if column in ['lemma']:

        pass
    elif column in ['next_lemma', 'previous_lemma']:
        pass
        
x_train_encoded = np.concatenate(encoded_x_train, axis=1)
y_train = y_train['argument']
model = create_classifier(x_train_encoded, y_train)


x_test, y_test = pd.read_csv(args.x_test), pd.read_csv(args.y_test)
encoded_x_test = []
for column in x_test.columns[1:]:
    if column not in skiplist:
        encoded_x_test.append(encode(encoder_dict[column], x_test[column].values))

x_test_encoded = np.concatenate(encoded_x_test, axis=1)
y_test = y_test['argument']

predictions = inference(model, x_test_encoded, args.output_file)

picklify(model)

evaluate(predictions, y_test)
