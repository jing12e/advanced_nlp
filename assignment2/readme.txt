## Prerequisites

Before running the scripts, ensure you have the following prerequisites installed:

- pandas
- numpy
- argparse
- scikit-learn
- spacy
- nltk

You can install these packages using pip:

```bash
pip install pandas numpy scikit-learn spacy nltk
```

After installing `nltk` and `spacy`, you'll need to download additional data:

```bash
python -m spacy download en_core_web_md
python -m nltk.downloader wordnet
```

## Setting Up Your Environment

Make sure you have the GloVe embeddings available locally. The code uses 'glove.6B.50d.txt'. You can download GloVe embeddings from [the GloVe website](https://nlp.stanford.edu/projects/glove/). It should be placed inside the  `Assignment2` folder.

## Running the Scripts

### 1. Data Preparation

First, you should generate or obtain your data file in ConLL-U format ('en_ewt-up-train.conllu', 'en_ewt-up-test.conllu'). The script 'generate_instances.py' will process this data to generate instances suitable for training and evaluation.

To run `generate_instances.py`, execute:

```bash
python generate_instances.py --dataset en_ewt-up-train.conllu --output_file train_dataset
```

This will create a JSON file with processed instances (`train_dataset.json`) and a corresponding CSV file (`train_dataset.csv`).

### 2. Embedding

Next, use `split_into_x_y_embedded.py` to split the data into features (X) and labels (Y), and apply GloVe embeddings.

```bash
python split_into_x_y_embedded.py --input_file train_dataset.csv --output_file train_dataset
```

This script will produce two CSV files: one for features (`train_dataset_x.csv`) and one for labels (`train_dataset_y.csv`).

### 3. Model Training and Evaluation

Finally, run the `fit_inference.py` script to train a logistic regression model and evaluate it on your test data. Ensure you have prepared your test data similarly to your training data.

```bash
python fit_inference.py --x_train train_dataset_x.csv --y_train train_dataset_y.csv --x_test test_dataset_x.csv --y_test test_dataset_y.csv --output_file output/inference.txt
```

This command trains the model using the given training data and evaluates it on the test data, saving the predictions to `output/inference.txt`. 

fit_inference.py has extra boolean arguments --eval_only, --load and --load_model to run the script in different modes. 
The example below loads a saved pickle model and trains it further on the files in the `generator_folder` folder.

```bash
python fit_inference.py --load_model --load
```

## Notes

- Ensure all scripts and necessary files are in the same directory or modify the paths in the commands accordingly.
- The default parameters (such as file names) in each script can be changed according to your specific needs. Use the `-h` option with any script to see all available options. For example:

```bash
python generate_instances.py -h
```

- The scripts are designed to work sequentially. Ensure you follow the steps in order for successful execution.
