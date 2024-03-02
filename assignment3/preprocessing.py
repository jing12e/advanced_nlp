import json
import pandas as pd
import argparse
import csv
import ast
import pandas as pd
def extract_data_from_conll_extended(conll_file, output_csv):
    sentences = []
    with open(conll_file, 'r', encoding='utf-8') as file:
        sentence_data = {"tokens": [], "predicate_positions": [], "argument_labels": []}
        for line in file:
            line = line.strip()

            if line.startswith("#"):
                if sentence_data["tokens"]:
                    sentences.append(sentence_data.copy())
                    sentence_data = {"tokens": [], "predicate_positions": [], "argument_labels": []}
                continue
            if line == "":
                continue

            parts = line.split(" ")
            parts = parts[0].split("\t")
            token_id = parts[0]
            token = parts[1]
            predicate_label = parts[10]
            argument_labels = parts[11:]

            sentence_data["tokens"].append(token)

            if predicate_label != "_":
                sentence_data["predicate_positions"].append([predicate_label,int(token_id)])

            sentence_data["argument_labels"].append(argument_labels)


        if sentence_data["tokens"]:
            sentences.append(sentence_data)

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


                sentence_instance = {

                    "word": tokens,
                    "predicate": predicate,
                    "arguments": arguments,
                    "predicate_position": int(predicate_position)

                }
                instances.append(sentence_instance)

    with open(output_csv, 'w', encoding='utf-8') as f:
        json.dump(instances, f, ensure_ascii=False, indent=4)




#extract_data_from_conll_extended("../data/en_ewt-up-test.conllu", "test.json")

