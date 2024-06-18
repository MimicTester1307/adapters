from collections import default_dict
import random
from random import choice
import secrets
import string
import json

LEN_DATASET = 3

def generate_string_to_number_mappings(count):
    unique_values = list(range(1, count + 1))
    random.shuffle(unique_values)
    unique_strings = set()
    characters = string.ascii_lowercase 
    
    while len(unique_strings) < count:
        random_string = ''.join(secrets.choice(characters) for _ in range(choice([2,3,4,5])))
        unique_strings.add(random_string)

    knowledge_artifact = dict(zip(list(unique_strings), unique_values))
    return knowledge_artifact

knowledge_artifact = generate_string_to_number_mappings(LEN_DATASET)
# print(knowledge_artifact)

# realized a namedtuple is not needed here
# arithmetic_expression = namedtuple("expression_properties", "key_expression value_expression value")

def generate_random_key():
    return choice(list(knowledge_artifact.keys()))

def create_arithmetic_expressions():
    arithmetic_key_expression = ''
    arithmetic_value_expression = ''

    for _ in range(choice(list(range(1, 5)))):
        operator = choice([' + ', ' - ', ' * '])
        key = generate_random_key()
        arithmetic_key_expression += key + operator
        arithmetic_value_expression += str(knowledge_artifact[key]) + operator

    last_key = generate_random_key()
    arithmetic_key_expression += last_key
    arithmetic_value_expression += str(knowledge_artifact[last_key])
    arithmetic_value = eval(arithmetic_value_expression)    

    return arithmetic_key_expression, arithmetic_value_expression, arithmetic_value

arithmetic_key_expressions = []
arithmetic_value_expressions = []
arithmetic_values = []

def generate_datasets(length):

    for _ in range(length):
        arithmetic_key_expression, arithmetic_value_expression, arithmetic_value = create_arithmetic_expressions()
        arithmetic_key_expressions.append(arithmetic_key_expression)
        arithmetic_value_expressions.append(arithmetic_value_expression)
        arithmetic_values.append(arithmetic_value)
    
    # print(arithmetic_key_expressions, arithmetic_value_expressions, arithmetic_values)

    key_expr_to_val_expr = list(zip(arithmetic_key_expressions, arithmetic_value_expressions))
    key_expr_to_arithmetic_val = list(zip(arithmetic_key_expressions, arithmetic_values))
    val_expr_to_arithmetic_val = list(zip(arithmetic_value_expressions, arithmetic_values))

    return key_expr_to_val_expr, key_expr_to_arithmetic_val, val_expr_to_arithmetic_val

key_expr_to_val_expr, key_expr_to_arithmetic_val, val_expr_to_arithmetic_val = generate_datasets(LEN_DATASET)

# METHOD 1 to create datasets:
D_KV_SUBS, D_KV_VAL, D_SUBS_VAL =  default_dict(), default_dict(), default_dict()

for key, val in key_expr_to_arithmetic_val:
    D_KV_SUBS["instruction"] = "Given an arithmetic expression between strings. Find the corresponding arithmetic expression in numeric terms between the numeric mappings of those strings."
    D_KV_SUBS["key"] = key
    D_KV_SUBS["value"] = val

with open("D_KV_SUBS.json", "w") as outfile:
    json.dump(D_KV_SUBS, outfile)

# METHOD 2 to create datasets:
# D_KV_SUBS, D_KV_VAL, D_SUBS_VAL =  [], [], []

# for key, subs in key_expr_to_val_expr:
#     row = "<s><INST> Given an arithmetic expression between strings: '" \
#          + key + "'. Find the corresponding arithmetic expression in numeric terms between the numeric mappings of those strings.</INST> " \
#              + "The corresponding numeric expression is: '" + subs + "'.</s>"
#     D_KV_SUBS.append(row)

# for key, val in key_expr_to_arithmetic_val:
#     row = "<s><INST> Given an arithmetic expression of the following type: '" \
#          + key + "'. Find the value corresponding to the solution of this arithmetic expression.</INST> " \
#              + "The numeric solution to this arithmetic expression is: '" + str(val) + "'.</s>"
#     D_KV_VAL.append(row)

# for key, subs in val_expr_to_arithmetic_val:
#     row = "<s><INST> Given an arithmetic expression: '" \
#          + key + "'. Find the value corresponding to the solution of this arithmetic expression.</INST> " \
#              + "The numeric solution to this arithmetic expression is: '" + str(val) + "'.</s>"
#     D_SUBS_VAL.append(row)


# print(D_KV_SUBS, D_KV_VAL, D_SUBS_VAL)