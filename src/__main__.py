from llm_sdk import Small_LLM_Model
from os import system
from json import load

model = Small_LLM_Model()
encode = model.encode
decode = model.decode
log = model.get_logits_from_input_ids


with open(model.get_path_to_vocab_file(), 'r') as f:
    tokens = load(f)

with open('data/data/input/functions_definition.json') as f:
    functions = load(f)

system('clear')
desc = {}
function_names = []

for fun in functions:
    desc[fun['name']] = fun['description']
    function_names.append(fun['name'])


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


        # print()
        # exit()
        if newly_generated in function_names:
            return(newly_generated)

def get_params(function_name):
    for fun in functions:
        if fun['name'] == function_name:
            return (list(fun['parameters'].keys()))



def build_number_json(user_request):
    chosen_function = get_function_names(user_request)
    j = f'''{{\n "prompt": "{user_request}",\n"name": "{chosen_function}",\n"parameters": {{"{get_params(chosen_function)[0]}": '''
    return j

def build_string_json(user_request):
    chosen_function = get_function_names(user_request)
    j = f'''{{\n "prompt": "{user_request}",\n"name": "{chosen_function}",\n"parameters": {{"{get_params(chosen_function)[0]}": "'''
    return j

# def extract_next_number(user_request):
#     partial_json = build_number_json(user_request)
#     prompt = f"""<|im_start|>system
#     You are a function calling assistant.
#     <|im_end|>
#     <|im_start|>user
#     {user_request}
#     <|im_end|>
#     <|im_start|>assistant
#     {partial_json}"""
#     to_print = to_print = len(prompt) - len(partial_json)

#     newly_generated = ""
#     function_selected = get_function_names(user_request)
#     parameters = get_params(function_selected)
#     for i in range(len(parameters)):
#         allowed = ""
#         if i == len(parameters) - 1:
#             allowed = '0123456789.-}'
#         else:
#             allowed = '0123456789.,-'
#         if i != 0:
#             newly_generated += f' "{parameters[i]}": '
#         count = 0
#         start_of_param = len(newly_generated)
#         while True:
#             logits = log(encode(prompt + newly_generated)[0].tolist())
#             for v in range(len(tokens), len(logits)):
#                 logits[v] == float('-inf')
#             for k, v in tokens.items():
#                 if not (k in allowed or k == 'Ġ-'):
#                     logits[v] = float('-inf')
#                 if count == 0 and i == 0 and k == 'Ġ-':
#                     logits[v] += 7
#                 if count == 0 and i != 0 and k == 'Ġ-':
#                     if '-' in  newly_generated:
#                         logits[v] += 7
#                     else:
#                         logits[v] += 12

#             count += 1
#             newly_generated += decode([logits.index(max(logits))])
#             if len(newly_generated[start_of_param:]) >= 20:
#                 if i != len(parameters) - 1:
#                     newly_generated += ","
#                     break
#                 if i == len(parameters) - 1:
#                     newly_generated += "}"
#                     break
#             if newly_generated[-1] in ',}':
                
#                 if i == len(parameters) - 1:
#                     newly_generated = newly_generated[:-1] + '}'
#                 break
#     return (prompt + newly_generated)[to_print:] + '\n}'

# def extract_string(user_request):
#     partial_json = build_string_json(user_request)
#     prompt = f"""<|im_start|>system
#     You are a function calling assistant.
#     <|im_end|>
#     <|im_start|>user
#     {user_request}
#     <|im_end|>
#     <|im_start|>assistant
#     {partial_json}"""
#     to_print = len(prompt) - len(partial_json)

#     newly_generated = ""
#     function_selected = get_function_names(user_request)
#     parameters = get_params(function_selected)
#     for i in range(len(parameters)):
#         para_start = len(newly_generated)
#         if i != 0:
#             newly_generated += f' "{parameters[i]}": "'
#         while True:
#             logits = log(encode(prompt + newly_generated)[0].tolist())
#             for v in range(len(tokens), len(logits)):
#                 logits[v] = float('-inf')
#             newly_generated += decode([logits.index(max(logits))])
#             if len(newly_generated[para_start:]) >= 50:
#                 if '}' not in newly_generated:
#                     newly_generated += '"}'
#                 break
#             if newly_generated.rstrip()[-1] in ',}':
#                 if i == len(parameters) - 1:
#                     newly_generated = newly_generated.rstrip()[:-1] + '}'
#                 break
#     return (prompt + newly_generated)[to_print:] + '\n}'

# def extract(user_request):
#     function_selected = get_function_names(user_request)
#     parameters = get_params(function_selected)

#     # determine type from first parameter
#     for fun in functions:
#         if fun['name'] == function_selected:
#             first_param_type = fun['parameters'][parameters[0]]['type']
#             break

#     if first_param_type == 'number':
#         partial_json = build_number_json(user_request)
#     else:
#         partial_json = build_string_json(user_request)

#     prompt = f"""<|im_start|>system
#     You are a function calling assistant.
#     <|im_end|>
#     <|im_start|>user
#     {user_request}
#     <|im_end|>
#     <|im_start|>assistant
#     {partial_json}"""
#     to_print = len(prompt) - len(partial_json)
#     newly_generated = ""

#     for i in range(len(parameters)):
#         para_start = len(newly_generated)
#         if i != 0:
#             if first_param_type == 'number':
#                 newly_generated += f' "{parameters[i]}": '
#             else:
#                 newly_generated += f' "{parameters[i]}": "'

#         count = 0
#         while True:
#             logits = log(encode(prompt + newly_generated)[0].tolist())
#             for v in range(len(tokens), len(logits)):
#                 logits[v] = float('-inf')

#             if first_param_type == 'number':
#                 allowed = '0123456789.-}' if i == len(parameters) - 1 else '0123456789.,-'
#                 for k, v in tokens.items():
#                     if not (k in allowed or k == 'Ġ-'):
#                         logits[v] = float('-inf')
#                     if count == 0 and k == 'Ġ-':
#                         if i == 0:
#                             logits[v] += 7
#                         else:
#                             logits[v] += 7 if '-' in newly_generated else 12

#             count += 1
#             newly_generated += decode([logits.index(max(logits))])

#             if first_param_type == 'number':
#                 if len(newly_generated[para_start:]) >= 20:
#                     newly_generated += ',' if i != len(parameters) - 1 else '}'
#                     break
#                 if newly_generated[-1] in ',}':
#                     if i == len(parameters) - 1:
#                         newly_generated = newly_generated[:-1] + '}'
#                     break
#             else:
#                 if len(newly_generated[para_start:]) >= 50:
#                     if '}' not in newly_generated:
#                         newly_generated += '"}'
#                     break
#                 if newly_generated.rstrip()[-1] in ',}':
#                     if i == len(parameters) - 1:
#                         newly_generated = newly_generated.rstrip()[:-1] + '}'
#                     break

#     return (prompt + newly_generated)[to_print:] + '\n}'

def generate_dynamic_json(user_request):
    function_selected = get_function_names(user_request)
    
    # Find the parameter definitions for the selected function
    param_defs = {}
    for fun in functions:
        if fun['name'] == function_selected:
            param_defs = fun.get('parameters', {})
            break
            
    keys = list(param_defs.keys())
    
    # Start building the JSON structure
    partial_json = f'{{\n "prompt": "{user_request}",\n"name": "{function_selected}",\n"parameters": {{'
    
    prompt = f"""<|im_start|>system
    You are a function calling assistant.
    <|im_end|>
    <|im_start|>user
    {user_request}
    <|im_end|>
    <|im_start|>assistant
    {partial_json}"""
    
    to_print = len(prompt) - len(partial_json)
    newly_generated = ""
    
    # Loop dynamically through the Pydantic/JSON keys
    for i, key in enumerate(keys):
        param_type = param_defs[key]['type']
        
        # Decide if the current parameter needs a comma or a closing brace
        terminator = "}" if i == len(keys) - 1 else ","
        
        if i != 0:
            newly_generated += " "
            
        newly_generated += f'"{key}": '
        
        # If it's a string, force the opening quote
        if param_type == "string":
            newly_generated += '"'
            
        tokens_generated = 0
        
        while True:
            logits = log(encode(prompt + newly_generated)[0].tolist())
            
            # Mask out-of-bounds tokens
            for v in range(len(tokens), len(logits)):
                logits[v] = float('-inf')
                
            # Dynamic Masking based on type
            for k, v in tokens.items():
                if param_type == "number":
                    # Remove tokenizers space/newline markers for clean checking
                    clean_k = k.replace('Ġ', '').replace('Ċ', '')
                    allowed = '0123456789.-' + terminator
                    
                    if not all(c in allowed for c in clean_k):
                        logits[v] = float('-inf')
                        
                elif param_type == "string":
                    # Mask newlines to prevent breaking the JSON structure
                    if 'Ċ' in k or '\n' in k:
                        logits[v] = float('-inf')

            # Pick the highest allowed token
            best_idx = logits.index(max(logits))
            chosen_token_str = decode([best_idx])
            newly_generated += chosen_token_str
            tokens_generated += 1
            
            # Dynamic Stop Conditions
            if param_type == "number":
                if terminator in chosen_token_str:
                    # Clean up if it generated garbage attached to the terminator
                    newly_generated = newly_generated.split(terminator)[0] + terminator
                    break
                
                # Failsafe: force terminator after 10 tokens
                if tokens_generated >= 10:
                    newly_generated += terminator
                    break
                    
            elif param_type == "string":
                # Look for the closing quote naturally generated by the model
                if '"' in chosen_token_str:
                    # Strip anything after the quote and force the terminator
                    parts = newly_generated.split('"')
                    newly_generated = '"'.join(parts[:-1]) + '"' + terminator
                    break
                
                # Failsafe: force closing quote and terminator after 25 tokens
                if tokens_generated >= 25:
                    newly_generated += '"' + terminator
                    break

    return (prompt + newly_generated)[to_print:] + '\n}'

with open('data/data/input/function_calling_tests.json') as f:
    tests = load(f)

for test in tests:
    for v in test.values():
        print(generate_dynamic_json(v))
