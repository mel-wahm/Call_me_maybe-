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

dec = {f['name']: f['description'] for f in functions}

desc = "\n".join(f"Function name: {key} ---> Function description: {value}" for key, value in dec.items())

user_request = "Say hi to Mark"



def strip(string):
    result = ""
    for char in string:
        if char != 'Ġ':
            result += char
    return result

from os import system
token_path = model.get_path_to_vocab_file()
f = open(token_path, 'r')
function_ids = [encode(text[3:])[0] for text in function_names]
tokens = json.load(f)

def get_function_names(user_request):

    prompt = f"""

    <|im_start|>system
    You are a function calling assistant. Given a user request, select the correct function from the list below.

    AVAILABLE FUNCTIONS
    {desc}
    <|im_end|>
    <|im_start|>user
    {user_request}
    <|im_end|>
    <|im_start|>assistant
    fn_
    """
    newly_generated = ""
    while True:
        logits = model.get_logits_from_input_ids((encode(prompt + newly_generated)[0]).tolist())
        for k, v in tokens.items():
            if all(not f.startswith(newly_generated + k) for f in function_names):
                logits[v] = float("-inf")
            
        for i in range(len(tokens), len(logits)):
            logits[i] = float('-inf')
        
        valid = [i for i in range(len(logits)) if logits[i]!= float('-inf')]
        valid = sorted(valid, key=lambda x: logits[x], reverse=True)
        newly_generated += decode([logits.index(max(logits))])
        
        
        if newly_generated in function_names:
            return newly_generated


def get_valid_number(user_request):
    prompt = f"""
    <|im_start|>system
    your job is to extract arguments from a user prompt like this:
    for nergative number do this
    "prompt": "What is the sum of -2 and -3?",
    "parameters": {{"a": -2.0, "b": -3.0}}
    "sum of -5 and 3" → {{"a": -5.0, "b": 3.0}}
    "prompt": "What is the sum of 2 and 3?",
    "parameters": {{"a": 2.0, "b": 3.0}}
    <|im_end|>
    <|im_start|>user
    {user_request}
    <|im_end|>
    <|im_start|>assistant

    "parameters": {{"a": """
    prompt_len = len(prompt)
    #get first parameter
    for i in range(20):
        enc = encode(prompt)
        logits = model.get_logits_from_input_ids(enc[0].tolist())
        if '-' not in prompt[prompt_len:]:

            logits[tokens.get('Ġ-')] += 10

        valid = ['Ġ-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '"', ',', '-']
        for k, v in tokens.items():
            if not k in valid:
                logits[v] = float('-inf')
        for log in range(len(tokens), len(logits)):
            logits[log] = float('-inf')

        # valid = {}
        # for k, v in tokens.items():
        #     if  logits[v] != float('-inf'):
        #         valid[k] = v
        # valid = dict(sorted(valid.items(), key = lambda elem : logits[elem[1]], reverse=True))
        # for k, v in valid.items():
        #     print(f'key: {k} ----> score: {logits[v]}')


        dec = decode([logits.index(max(logits))])
        prompt += dec
        system('clear')
        print(prompt)
        if ',' in dec:
            break
        if i == 19:
            prompt += ","
            break

    prompt+=  ' "b": '
    prompt_len = len(prompt)
    for i in range(20):
        enc = encode(prompt)
        logits = model.get_logits_from_input_ids(enc[0].tolist())
        if not '-' in prompt[prompt_len:]:
            logits[tokens.get('Ġ-')] += 10

        valid = ['Ġ-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '}', '-']
        for k, v in tokens.items():
            if not k in valid:
                logits[v] = float('-inf')
        for log in range(len(tokens), len(logits)):
            logits[log] = float('-inf')

        dec = decode([logits.index(max(logits))])
        prompt += dec
        system('clear')
        print(prompt)
        if '}' in dec:
            break
        if i == 19:
            prompt += "}"
            break

def get_function_parameters(function_name):
    for func in functions:
        if func.get('name') == function_name:
            return func.get('parameters')

# def get_next_number(user_request):
#     prompt = f"""<|im_start|>system
#     You are a number extractor. Extract the relevant number from the user request.
#     Only output the number, nothing else.
#     <|im_end|>
#     <|im_start|>user
#     {user_request}
#     <|im_end|>
#     <|im_start|>assistant
#     """
#     p_len = len(prompt)
#     while True:
#         enc = encode(prompt)
#         logits = log(enc[0].tolist())

#         avaiable_logits = [str(i) for i in range(10)]
#         avaiable_logits.extend(['.', '"', ',', '-', 'Ġ-', '}'])

#         if '-' not in prompt[p_len:]:
#             logits[tokens.get('Ġ-')] += 10
        
#         if '.' in prompt[p_len:]:
#             logits[tokens.get('.')] = float('-inf')
#         if '-' in prompt[p_len:]:
#             logits[tokens.get('Ġ-')] = float('-inf')
#         if 'Ġ-' in prompt[p_len:]:
#             logits[tokens.get('Ġ-')] = float('-inf')
        
#         for k, v in tokens.items():
#             if not k in avaiable_logits:
#                 logits[v] = float('-inf')

        
#         prompt += decode([logits.index(max(logits))])
#         system('clear')
#         print(prompt)
#         if '}' in prompt[p_len:]:
#             break


test_prompt = f"""<|im_start|>system
You are a number extractor. Extract the relevant number from the user request.
Only output the number, nothing else.
<|im_end|>
<|im_start|>user
whats 5 + 9?
<|im_end|>
<|im_start|>assistant
"""



def get_next_number(prompt, terminator):
    generated = ""
    digit_ids = set(range(15, 25))  # 0-9
    dot_id = 13
    sign_id = 481  # Ġ-
    
    term_id = tokens.get(terminator)
    while True:
        original_logits = model.get_logits_from_input_ids(encode(prompt + generated)[0].tolist())
        logits = original_logits[:]  # copy
        
        has_digit = any(c.isdigit() for c in generated)
        has_dot = '.' in generated
        has_sign = '-' in generated
        
        # block everything
        for i in range(len(logits)):
            logits[i] = float('-inf')
        
        # allow digits — restore original values
        for i in digit_ids:
            logits[i] = original_logits[i]
        
        # allow sign only at start
        if not has_digit and not has_sign:
            logits[sign_id] = original_logits[sign_id]
        
        # allow dot only after a digit, only once
        if has_digit and not has_dot:
            logits[dot_id] = original_logits[dot_id]
        
        # allow terminator only after a digit
        if has_digit:
            logits[term_id] = original_logits[term_id]
        
        best = logits.index(max(logits))
        
        if best == term_id:
            break
        
        generated += decode([best])
        system('clear')
        print(generated)
    return generated

get_next_number(test_prompt, '}')
get_next_number(test_prompt, ',')

"""

14.00000000000000000000000000000000000000000
^CTraceback

"""