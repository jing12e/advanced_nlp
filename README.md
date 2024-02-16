How to run minumum_complete_tree.ipynb
- download Standford CoreNLP
- start Standford CoreNLP Server  
    `java -mx4g -cp "D:\Desktop\vu-courses\wdps22\stanford-corenlp-4.5.5\*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000`
    change the filepath to where the standford corenlp located in your pc
- install nltk and dependencies


implement.py include features of tokenization, POS taggings, segment and dependency parse
How to run implement.py
- install spaCy
- input the text you want to implement in the end of python file


How to run featurize.py

To run the featurize.py, you need to download the `en_core_web_sm` model from spaCy. This can be done using the following command:

```
python -m spacy download en_core_web_sm
```

Usage

1. Ensure all dependencies are installed and the required spaCy model is downloaded.

- scikit-learn
- numpy
- spaCy
- tqdm
- matplotlib

2. The script can be run via command line or terminal with optional arguments. Example:

```
python featurize.py --eval_num=10
```

- `--eval_num` specifies how many tokens to print features for and defaults to 10.
