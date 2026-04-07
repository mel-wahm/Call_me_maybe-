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

def extract_next_number(prompt):
    new = f"""
    <im_start|>system
    you are a tool to extract number parameters, for example "prompt": "What is the sum of 2 and 3?",
    "parameters": {{"a": 2.0, "b": 3.0}}
    <im_end|>
    <im_start|>user
    you will extract these numbers based on this prompt: {prompt}
    <im_end|>
    <im_start|>assistant
    {prompt}
    """
    length_of_original = len(prompt)
    start_of_answer = 0
    for i in range(20, len(new)):
        if new[i - 9: i] == 'assistant':
            start_of_answer = i
            break
    print(start_of_answer)
    exit()
    while True:
        pass



def build_answer(user_request):
    selected_function = get_function_names(user_request)
    function_parameters = get_params(selected_function)
    answer =  f'"prompt": "{user_request}",\n"name": "{selected_function}","'
    

extract_next_number(('"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a"'))

print(build_answer("Say hi to the simo"))


