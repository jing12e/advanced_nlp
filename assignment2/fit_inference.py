from sklearn.linear_model import LogisticRegression, SGDClassifier
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from tqdm import tqdm

def load_model(pickles):
    '''
    Input: string name with .pickle extension. Must be present in working directory. Check with 'pwd' command
    Loads pretrained model variable from pickle file at filepath(pickle).
    '''
    with open(pickles, 'rb') as p:
        return pickle.load(p)

def unpicklify(pickles):
    '''
    Loads a variable from a pickle file
    '''
    with open(pickles, 'rb') as p:
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
    
    parser.add_argument('--load', type=bool,
                        default=False,
                        help="Load from generator folder? (needs to be prepopulated)")
    
    parser.add_argument('--eval_only', type=bool,
                        default=False,
                        help="Only evaluate a the stored model.")
    
    parser.add_argument('--load_model', type=bool,
                        default=False,
                        help="Load a stored model instead of creating one to train from scratch.")
    return parser.parse_args()

def create_classifier():
    model = LogisticRegression(max_iter=1000, verbose=True)
    model = SGDClassifier(loss='log_loss')

    return model

def fit_classifier(x_train, y_train, model, classes):
    return model.partial_fit(x_train, y_train, classes=classes)

    

def inference(model, x_test, output_file):
    predictions = model.predict(x_test)
    with open(output_file, 'w') as file:
        [file.write(f"{prediction}\n") for prediction in predictions]
    return predictions

def evaluate(predictions, true_y):
    report = classification_report(true_y, predictions, output_dict=True)
    conf_matrix = confusion_matrix(true_y, predictions)
    print(report)
    print(conf_matrix)

# Example usage:

args = process_args()
skiplist = ['word','predicate_position', 'next_lemma','previous_lemma', 'lemma','path_len']
x_train, y_train = pd.read_csv(args.x_train), pd.read_csv(args.y_train) #we have one million words to process. It's too long
x_train['predicate'] = pd.Series([1 if isinstance(pred, str) else 0 for pred in x_train['predicate']])

if not args.load:
    t = x_train.at[0,'predicate']
    x_train,y_train = x_train, y_train
encoder_dict = {feature: fit_encoder(x_train[feature].values) for feature in x_train.columns[1:] if feature not in skiplist}
encoded_x_train = []
dict_vec = DictVectorizer()
norm_factor = x_train['path_len'].max()
print(norm_factor)
if not args.load:
    for column in x_train.columns[1:]:
        if column not in skiplist:
            encoded_x_train.append(encode(encoder_dict[column], x_train[column].values))
        if column == 'path_len':
            path_len = np.array(x_train['path_len']).reshape(-1, 1)
            norm_factor = max(path_len)
            encoded_x_train.append(path_len/norm_factor)

        elif column in ['next_lemma', 'previous_lemma']:
            pass

    x_train = np.concatenate(encoded_x_train, axis=1).astype(float)
y_train = y_train['argument']
print(f'### Populating Generator folder ###\n')
if not args.load:
    for i in tqdm(range(100)):
        picklify(f'generator_folder/x_train_{i}.pickle',x_train[int(len(x_train)/100)*(i-1):int(len(x_train)/100)*i])
x_train = 0
if not args.eval_only:
    if not args.load_model:
        model = create_classifier()
    else:
        model = load_model('logreg.pickle')
    print(f'### Training LogReg Model ###\n')
    classes = np.unique(y_train)
    for i in tqdm(range(100)):
        x_train = np.array(unpicklify(f'generator_folder/x_train_{i}.pickle'))
        y_train_chunk = y_train[int(len(y_train)/100)*(i-1):int(len(y_train)/100)*i]
    
        if len(x_train) > 5:
            model = fit_classifier(x_train,y_train_chunk,model, classes)
            picklify('logreg.pickle',model)
else:
    model = load_model('logreg.pickle')



x_test, y_test = pd.read_csv(args.x_test), pd.read_csv(args.y_test)
encoded_x_test = []
for column in x_test.columns[1:]:
    if column not in skiplist:
        encoded_x_test.append(encode(encoder_dict[column], x_test[column].values))
    if column == 'path_len':
        path_len = np.array(x_test['path_len']).reshape(-1, 1)
        encoded_x_test.append(path_len/norm_factor)

x_test_encoded = np.concatenate(encoded_x_test, axis=1)
y_test = y_test['argument']

predictions = inference(model, x_test_encoded, args.output_file)


evaluate(predictions, y_test)
