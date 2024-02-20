import json
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
            pos_tag = parts[3]
            predicate_label = parts[10]
            argument_labels = parts[11:]

            sentence_data["tokens"].append(token)
            sentence_data["features"].append({"token": token, "lemma": "_", "pos_tag": pos_tag})
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
            instance = {
                "tokens": tokens,
                "predicate": predicate,
                "predicate_position": predicate_position,
                "arguments": arguments
            }
            instances.append(instance)

    return instances

# Example usage

# Example usage
data_file = "../data/en_ewt-up-test.conllu"
sentences = extract_data_from_conll_extended(data_file)
instances = generate_instances(sentences)

# Save instances as JSON file
output_file = "instances.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(instances, f, ensure_ascii=False, indent=4)

print("Instances saved to", output_file)