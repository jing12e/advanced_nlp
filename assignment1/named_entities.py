import nltk
from nltk import word_tokenize, pos_tag, ne_chunk


text = "Barack Obama was born in Hawaii. He was the 44th President of the United States."


tokens = word_tokenize(text)
tagged_tokens = pos_tag(tokens)


named_entities = ne_chunk(tagged_tokens)


for entity in named_entities:
    if isinstance(entity, nltk.tree.Tree):
        entity_label = entity.label()
        entity_text = " ".join([word for word, tag in entity.leaves()])
        print(f"Entity: {entity_text}, Label: {entity_label}")
