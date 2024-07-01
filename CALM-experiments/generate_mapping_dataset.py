from collections import defaultdict
import random
from random import choice, choices
import secrets
import string
import json

LEN_DATASET = 48000
RANGE_OF_MAPPINGS = 100
OPERATORS = [' + ', ' - ']


# CREATING KNOWLEDGE ARTIFACT (MAPPINGS)
def generate_string_to_number_mappings(count):
    unique_values = list(range(1, count + 1))
    random.shuffle(unique_values)
    unique_strings = set()
    characters = string.ascii_lowercase 
    
    while len(unique_strings) < count:
        random_string = ''.join(secrets.choice(characters) for _ in range(choice([2,3,4,5])))
        unique_strings.add(random_string)

    knowledge_artifact = list(zip(list(unique_strings), unique_values))
    return knowledge_artifact

knowledge_artifact_list = generate_string_to_number_mappings(RANGE_OF_MAPPINGS)
knowledge_artifact = dict(knowledge_artifact_list)

def generate_random_key():
    return choice(list(knowledge_artifact.keys()))


# GENERATING DATASET 1 
def generate_key_pairs_dataset(size):
    key_expressions = []

    for _ in range(size):
        collection = defaultdict(list)

        for sample_length in range(choice(list(range(3, 6)))):
            collection["examples"].append(choices(knowledge_artifact_list))

        collection["query"] = choices(collection["examples"])
        unpacked_examples = [item[0] for item in collection['examples']]
        query, value = collection['query'][0][0]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        transformed_dict = {
            'examples': mapped_examples,
            'query': query,
            'value': value,
        }

        key_expressions.append(transformed_dict)

    return (key_expressions)

key_to_value_mappings = generate_key_pairs_dataset(LEN_DATASET)


# GENERATING DATASET 2
def create_numeric_arithmetic_expressions():
    arithmetic_value_expression = ''

    for _ in range(choice(list(range(1, 5)))):
        operator = choice(OPERATORS)
        key = generate_random_key()
        arithmetic_value_expression += str(knowledge_artifact[key]) + operator

    last_key = generate_random_key()
    arithmetic_value_expression += str(knowledge_artifact[last_key])
    arithmetic_value = eval(arithmetic_value_expression)    

    return arithmetic_value_expression, arithmetic_value

def generate_arithmetic_training_dataset(length):
    arithmetic_value_expressions = []
    arithmetic_values = []

    for _ in range(length):
        arithmetic_value_expression, arithmetic_value = create_numeric_arithmetic_expressions()
        arithmetic_value_expressions.append(arithmetic_value_expression)
        arithmetic_values.append(arithmetic_value)
    
    # print(arithmetic_key_expressions, arithmetic_value_expressions, arithmetic_values)

    val_expr_to_arithmetic_val = list(zip(arithmetic_value_expressions, arithmetic_values))

    return val_expr_to_arithmetic_val

val_expr_to_arithmetic_val = generate_arithmetic_training_dataset(LEN_DATASET)


# GENERATING DATASET 3
def create_arithmetic_expressions_from_keys(map):
    keys = list(map.keys())
    # print(map)
    # print(keys)
    arithmetic_key_expression = ''
    arithmetic_value_expression = ''
    for _ in range(choice(list(range(1, 5)))):
        operator = choice(OPERATORS)
        key_operand = choices(keys)[0]
        # print(key_operand)
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
        # print(collection["pairs"])
        unpacked_examples = [item[0] for item in collection['pairs']]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        transformed_collection = {
            'pairs': mapped_examples,
            'expression': '',
            'value': 0,
        }
        # print(transformed_dict["pairs"].keys())
        arithmetic_key_expression, total_value = create_arithmetic_expressions_from_keys(transformed_collection["pairs"])
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
with open("D_KV_SUBS.json", "w") as outfile:
    json.dump(D_KV_SUBS, outfile)

# adapter 2
with open("D_SUBS_VAL.json", "w") as outfile:
    json.dump(D_SUBS_VAL, outfile)

# merged
with open("D_KV_VAL.json", "w") as outfile:
    json.dump(D_KV_VAL, outfile)


# Creating small dataset for inference
def inference_dataset_for_adapter_1(size):
    f = open("inference_for_dataset_1.txt", 'w')

    for _ in range(size):
        collection = defaultdict(list)

        for sample_length in range(choice(list(range(3, 6)))):
            collection["examples"].append(choices(knowledge_artifact_list))

        collection["query"] = choices(collection["examples"])
        unpacked_examples = [item[0] for item in collection['examples']]
        query, value = collection['query'][0][0]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        prompt = f"# Arithmetic Expression: {mapped_examples}, # Query: {query} # Value: "

        f.write(prompt)
        f.write('\n')

    f.close()

inference_dataset_for_adapter_1(10)

def inference_dataset_for_merged_adapter(size):
    f = open("inference_for_merged_adapter.txt", 'w')

    for _ in range(size):
        collection = defaultdict(list)

        for sample_length in range(choice(list(range(3, 6)))):
            collection["pairs"].append(choices(knowledge_artifact_list))

        unpacked_examples = [item[0] for item in collection['pairs']]
        mapped_examples = {string_key : value for string_key, value in unpacked_examples}
        arithmetic_key_expression, total_value = create_arithmetic_expressions_from_keys(mapped_examples)
        prompt = f"# Pairs: {mapped_examples}, # Arithmetic Expression: {arithmetic_key_expression}, # Value: "
        f.write(prompt)
        f.write('\n')

    f.close()

inference_dataset_for_merged_adapter(20)