import spacy

nlp = spacy.load("en_core_web_sm")

def tokenize(text):
    """
    Tokenizes the input text.

    """
    doc = nlp(text)
    print("Tokenization:")
    for token in doc:
        print(token.text)
    print("\n")


def pos_tagging(text):
    """
    Performs POS tagging on the input text.

    """
    doc = nlp(text)
    print("POS Tagging:")
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)
    print("\n")

def dependency_parse(text):
    """
    Performs dependency parsing on the input text.

    """
    doc = nlp(text)
    print("Dependency Parse:")
    for chunk in doc.noun_chunks:
        print(chunk.text, chunk.root.text, chunk.root.dep_, chunk.root.head.text)
    print("\n")

def sentence_segmentation(text):
    """
    Segments the input text into sentences.

    """
    doc = nlp(text)
    print("Sentence Segmentation:")
    assert doc.has_annotation("SENT_START")
    for sent in doc.sents:
        print(sent.text)
    print("\n")

text = "Apple is looking at buying U.K. startup for $1 billion"

tokenize(text)
pos_tagging(text)
dependency_parse(text)
sentence_segmentation(text)