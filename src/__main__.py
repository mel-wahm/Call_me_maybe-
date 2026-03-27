from llm_sdk import Small_LLM_Model
import json
model = Small_LLM_Model()

def get_sceleton():
    definitions = open('data/input/functions_definition.json', 'r')
    defs = json.load(definitions)
    
    functions = {}
    for function in defs:
        functions[function['name']] = (function['parameters'])
    return functions

sceleton = get_sceleton()

parameter  = list(sceleton['fn_add_numbers'].keys())[0]

type_of_parameter = sceleton['fn_add_numbers'][parameter]['type']

def json_prefix(prompt, function_name, sceleton):
    para = '"' + list(sceleton[function_name].keys())[0] + '"' + ':' 
    return f"""
    {{
    "prompt": "{prompt}",
    "name": "{function_name}",
    "parameters": {{{para}
    """

path_to_tokens = model.get_path_to_vocab_file()
with open(path_to_tokens, 'r') as f:
    tokens = json.load(f)

def strip_space(token):
    resulted_token = ""
    for char in token:
        if char != 'Ġ':
            resulted_token += char
    return resulted_token

def one_num(token):
    digits = "0123456789"
    for char in token:
        if char in digits:
            return True
    return False

def get_valid_numbers():
    valid_numbers = []
    for k in tokens:
        if one_num(k):
            valid_numbers.append(k)
    return valid_numbers

print(get_valid_numbers())

"""

p src/__main__.py
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 311/311 [00:00<00:00, 558.63it/s]
['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '²', '³', '¹']

"""