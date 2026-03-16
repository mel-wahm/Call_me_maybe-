from llm_sdk import Small_LLM_Model
import json
from os import system
from time import sleep

model = Small_LLM_Model()
encode = model.encode
decode = model.decode
log = model.get_logits_from_input_ids

fun = open('data/input/functions_definition.json', 'r')
functions = json.load(fun)

function_names = [f['name'] for f in functions]
print(function_names)
user_request = "what is the sum of 2 and 3?"
prompt = f"""
Available functions: {function_names}

For this request: {user_request}
The function to call is: """
token_path = model.get_path_to_vocab_file()
f = open(token_path, 'r')
tokens = json.load(f)

def get_function_names(prompt):
    for _ in range(30):
        logits = log(list(encode(prompt)[0]))
        # for key, value in tokens.items():
        #     if not key.isdigit():
        #         logits[value] = float('-inf')
        # for i in range(len(tokens), len(logits)):
        #         logits[value] = float('-inf')
        prompt += decode([logits.index(max(logits))])
        sleep(0.01)
        system('clear')
        print(prompt)

get_function_names(prompt)

"""
looking at this list ['fn_add_numbers', 'fn_greet', 'fn_reverse_string', 'fn_get_square_root', 'fn_substitute_string_with_regex'] the available functions are: fn
fn_add_numbers
fn_greet
fn_reverse_string
fn_get_square_root
fn_substitute_string_with_regex
"""