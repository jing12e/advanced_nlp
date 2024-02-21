import json
import feature_extract
from split_instance_into_words import split_instance_into_words
import pandas as pd
def extract_data_from_conll_extended(conll_file):
    sentences = []
    with open(conll_file, 'r', encoding='utf-8') as file:
        sentence_data = {"tokens": [], "features": [], "predicate_positions": [], "argument_labels": []}
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                if sentence_data["tokens"]:
                    sentences.append(sentence_data.copy())
                    sentence_data = {"tokens": [], "features": [], "predicate_positions": [], "argument_labels": []}
                continue
            if line == "":
                continue

            parts = line.split(" ")
            parts = parts[0].split("\t")
            token_id = parts[0]
            token = parts[1]
            lemma = parts[2]
            pos_tag = parts[3]
            predicate_label = parts[10]
            argument_labels = parts[11:]

            sentence_data["tokens"].append(token)
            sentence_data["features"].append({"token": token, "lemma": lemma, "pos_tag": pos_tag})
            if predicate_label != "_":
                sentence_data["predicate_positions"].append([predicate_label,int(token_id)])

            sentence_data["argument_labels"].append(argument_labels)

        # Append the last sentence if not empty
        if sentence_data["tokens"]:
            sentences.append(sentence_data)

    return sentences

def generate_instances(sentences):
    instances = []
    for sentence in sentences:
        tokens = sentence["tokens"]
        predicates = sentence["predicate_positions"]
        n = 0
        for predicate_data in predicates:
            predicate = predicate_data[0]
            predicate_position = predicate_data[1]

            argument_labels = sentence["argument_labels"]
            arguments = [labels[n] for labels in argument_labels]
            n = n + 1

            text = " ".join(tokens)
            features_df = feature_extract.extract_features(text)


            word_instances = split_instance_into_words({
                "word": str(tokens),
                "predicate": predicate,
                "predicate_position": predicate_position,
                "arguments": str(arguments),
                "lemma": str(features_df['lemma'].tolist()),
                "ner": str(features_df['ner'].tolist()),
                "pos": str(features_df['pos'].tolist()),
                "pos_head": str(features_df['pos_head'].tolist()),
                "dependency_label": str(features_df['dependency_label'].tolist()),

                "next_lemma": str(features_df['next_lemma'].tolist()),
                "previous_lemma": str(features_df['previous_lemma'].tolist()),

                "suffix_3": str(features_df['suffix_3'].tolist()),
                "suffix_2": str(features_df['suffix_2'].tolist()),
                "hypernym": str(features_df['hypernym'].tolist())
            })

            instances.extend(word_instances)

    return instances



# Example usage
data_file = "../data/en_ewt-up-test.conllu"
sentences = extract_data_from_conll_extended(data_file)
instances = generate_instances(sentences)


# Save instances
output_file = "test_dataset.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(instances, f, ensure_ascii=False, indent=4)

print("Instances saved to", output_file)

with open('test_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)


df = pd.DataFrame(data)
print(df)
output_file_csv = "test_dataset.csv"
df.to_csv(output_file_csv, index=False)
print("Instances saved to", output_file_csv)