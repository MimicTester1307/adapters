from collections import defaultdict
import random
from random import choice, choices
import secrets
import string
import json

LEN_DATASET = 24000
RANGE_OF_MAPPINGS = 100
OPERATORS = [' + ']


# CREATING KNOWLEDGE ARTIFACT (MAPPINGS)
def generate_string_to_number_mappings(count):
    unique_values = list(range(1, count + 1))
    random.shuffle(unique_values)
    unique_strings = set()
    characters = string.ascii_lowercase 
    
    while len(unique_strings) < count:
        random_string = ''.join(secrets.choice(characters) for _ in range(choice([2,3])))
        unique_strings.add(random_string)

    knowledge_artifact = list(zip(list(unique_strings), unique_values))
    return knowledge_artifact

knowledge_artifact_list = generate_string_to_number_mappings(RANGE_OF_MAPPINGS)
knowledge_artifact = dict(knowledge_artifact_list)

def generate_random_key(knowledge_artifact):
    return choice(list(knowledge_artifact.keys()))

# GENERATING DATASET 1 
def generate_key_pairs_dataset(size):
    key_expressions = []

    for _ in range(size):
        collection = defaultdict(list)

        for sample_length in range(choice(list(range(3, 6)))):
            collection["pairs"].append(choices(knowledge_artifact_list))

        collection["query"] = choices(collection["pairs"])
        unpacked_examples = [item[0] for item in collection['pairs']]
        query, value = collection['query'][0][0]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        transformed_dict = {
            'pairs': mapped_examples,
            'query': query,
            'value': value,
        }
        key_expressions.append(transformed_dict)
        # key_expressions.append(collection)
        # random.shuffle(unpacked_examples)
        # shuffled_mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        # collection["queries"] = list(shuffled_mapped_examples.keys())
        # collection["values"] = list(shuffled_mapped_examples.values())

        # key_expressions.append(collection)

    return (key_expressions)

key_to_value_mappings = generate_key_pairs_dataset(LEN_DATASET)


# GENERATING DATASET 2
def create_numeric_arithmetic_expressions(knowledge_base, expression_length):
    arithmetic_value_expression = ''

    for _ in range(choice(list(range(1, expression_length)))):
        operator = choice(OPERATORS)
        key = generate_random_key(knowledge_base)
        arithmetic_value_expression += str(knowledge_base[key]) + operator

    last_key = generate_random_key(knowledge_base)
    arithmetic_value_expression += str(knowledge_base[last_key])
    arithmetic_value = eval(arithmetic_value_expression)    

    return arithmetic_value_expression, arithmetic_value

def generate_arithmetic_training_dataset(length):
    arithmetic_value_expressions = []
    arithmetic_values = []

    for _ in range(length):
        arithmetic_value_expression, arithmetic_value = create_numeric_arithmetic_expressions(knowledge_artifact, 5)
        arithmetic_value_expressions.append(arithmetic_value_expression)
        arithmetic_values.append(arithmetic_value)
    
    # print(arithmetic_key_expressions, arithmetic_value_expressions, arithmetic_values)

    val_expr_to_arithmetic_val = list(zip(arithmetic_value_expressions, arithmetic_values))

    return val_expr_to_arithmetic_val

val_expr_to_arithmetic_val = generate_arithmetic_training_dataset(LEN_DATASET)


# GENERATING DATASET 3
def create_arithmetic_expressions_from_keys(map, expression_length):
    keys = list(map.keys())
    arithmetic_key_expression = ''
    arithmetic_value_expression = ''

    for _ in range(choice(list(range(1, expression_length)))):
        operator = choice(OPERATORS)
        key_operand = choices(keys)[0]
        numeric_operand = map[key_operand]
        arithmetic_key_expression += key_operand + operator
        arithmetic_value_expression += str(numeric_operand) + operator

    last_key_operand = choices(keys)[0]
    last_numeric_operand = map[key_operand]
    arithmetic_key_expression += last_key_operand
    arithmetic_value_expression += str(last_numeric_operand)

    total_value = eval(arithmetic_value_expression)
    return arithmetic_key_expression, total_value

def generate_merged_dataset(size):
    merged_dataset = []
    for _ in range(size):
        collection = defaultdict(list)
        for sample_length in range(choice(list(range(3, 6)))):
            collection["pairs"].append(choices(knowledge_artifact_list))
        unpacked_examples = [item[0] for item in collection['pairs']]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        transformed_collection = {
            'pairs': mapped_examples,
            'expression': '',
            'value': 0,
        }
        # print(transformed_dict["pairs"].keys())
        arithmetic_key_expression, total_value = create_arithmetic_expressions_from_keys(mapped_examples, 5)
        transformed_collection["expression"] = arithmetic_key_expression
        transformed_collection['value'] = total_value
        merged_dataset.append(transformed_collection)

    return merged_dataset

key_value_pairs_to_key_expressions = generate_merged_dataset(LEN_DATASET)
# print("last dataset: \n\n", key_value_pairs_to_key_expressions)


# WRITING TO JSON
D_KV_SUBS, D_SUBS_VAL, D_KV_VAL =  key_to_value_mappings, [], key_value_pairs_to_key_expressions

def create_arithmetic_dataset_list(mapping, dataset_list):
    for key, val in mapping:
        record = defaultdict()
        record["key"] = key
        record["value"] = val
        dataset_list.append(record)

create_arithmetic_dataset_list(val_expr_to_arithmetic_val, D_SUBS_VAL)

# adapter 1
with open("datasets/D_KV_SUBS.json", "w") as outfile:
    json.dump(D_KV_SUBS, outfile)

# adapter 2
with open("datasets/D_SUBS_VAL.json", "w") as outfile:
    json.dump(D_SUBS_VAL, outfile)

# merged
with open("datasets/D_KV_VAL.json", "w") as outfile:
    json.dump(D_KV_VAL, outfile)


# Creating small dataset for inference
def inference_dataset_for_adapter_1(size):
    f = open("inference_inputs/inference_for_dataset_1.txt", 'w')

    for _ in range(size):
        collection = defaultdict(list)

        for sample_length in range(choice(list(range(3, 6)))):
            collection["pairs"].append(choices(knowledge_artifact_list))

        collection["query"] = choices(collection["pairs"])
        unpacked_examples = [item[0] for item in collection['pairs']]
        query, value = collection['query'][0][0]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        # random.shuffle(unpacked_examples)
        # shuffled_mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        # queries, values = list(shuffled_mapped_examples.keys()), list(shuffled_mapped_examples.values())
        prompt = f"{mapped_examples}, {query} = "

        f.write(prompt)
        f.write('\n')

    f.close()

inference_knowledge_artifact = generate_string_to_number_mappings(30)
print(inference_knowledge_artifact, dict(inference_knowledge_artifact))

def inference_dataset_for_adapter_2(size):
    f = open("inference_inputs/inference_for_dataset_2.txt", 'w')

    for _ in range(size):
        arithmetic_value_expression, arithmetic_value = create_numeric_arithmetic_expressions(dict(inference_knowledge_artifact), 2)
        prompt = f"{arithmetic_value_expression} = "
    
        f.write(prompt)
        f.write('\n')

    f.close()


def inference_dataset_for_merged_adapter(size):
    f = open("inference_inputs/inference_for_merged_adapter.txt", 'w')
    icl = open("inference_inputs/inference_for_base_model_merged_adapter.txt", 'w')

    for _ in range(size):
        collection = defaultdict(list)

        for sample_length in range(choice(list(range(3, 6)))):
            collection["pairs"].append(choices(inference_knowledge_artifact))

        unpacked_examples = [item[0] for item in collection['pairs']]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        arithmetic_key_expression, total_value = create_arithmetic_expressions_from_keys(mapped_examples, 2)
        prompt = f"{mapped_examples}, {arithmetic_key_expression} = "
        f.write(prompt)
        f.write('\n')

    for _ in range(size):
        collection = defaultdict(list)

        for sample_length in range(choice(list(range(3, 6)))):
            collection["pairs"].append(choices(inference_knowledge_artifact))

        unpacked_examples = [item[0] for item in collection['pairs']]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        arithmetic_key_expression, total_value = create_arithmetic_expressions_from_keys(mapped_examples, 2)
        prompt = f"{mapped_examples}, {arithmetic_key_expression} = "
        icl.write(prompt)
        icl.write('\n')

    f.close()
    icl.close()

inference_dataset_for_adapter_1(20)
inference_dataset_for_adapter_2(20)
inference_dataset_for_merged_adapter(20)