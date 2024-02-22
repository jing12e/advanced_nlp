from sklearn.linear_model import LogisticRegression

def process_args():
    """
    Processes user arguments from the command prompt. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_train', type=str, 
                        default='training_dataset_x.csv',
                        help="Input the relative path in the folder to the csv containing featurizes for the training set")
    
    parser.add_argument('--y_train', type=str, 
                        default='training_dataset_y.csv',
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
    model = LogisticRegression(max_iter=1000) #can remove max_iter if the model converges properly
    x_train, y_train = x_train.to_dict(orient='records'), y_train.to_dict(orient='records') 
    model.fit(x_train, y_train)
    return model

def inference(model, x_test, y_test, output_file):
    predictions = model.predict(x_test)
    with open(output_file, 'w') as file:
        [file.write(f"{prediction}\n") for prediction in predictions]
    return predictions

def evaluate(prediction, true_y):
    pass
    # some evaluation

# Example usage:
args = proces_args()
x_train, y_train, x_test, y_test = pd.read_csv(args.x_train), pd.read_csv(args.y_train), pd.read_csv(args.x_test), pd.read_csv(args.y_test)
model = create_classifier(x_train, y_train)
inference = mode, (x_test, y_test)
