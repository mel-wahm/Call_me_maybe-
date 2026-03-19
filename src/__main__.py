 cat src/__main__.py
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



from os import system
token_path = model.get_path_to_vocab_file()
f = open(token_path, 'r')
function_ids = [encode(text[3:]) for text in function_names]
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
        
        
        # print()
        # exit()
        if newly_generated in function_names:
            print(newly_generated)
            break


tests = [
    # Clear cases
    "What is 5 plus 3?",
    "Greet Alice",
    "Reverse the string 'python'",
    "What is the square root of 25?",
    "Replace all vowels with * in 'hello world'",
    
    # Synonyms
    "Add 100 and 200",
    "Say hello to Bob",
    "Flip the string 'banana'",
    "Calculate sqrt of 81",
    "Substitute 'foo' with 'bar' in 'foo fighters'",
    
    # Indirect phrasing
    "I need to sum 7 and 8",
    "Can you greet my friend Carlos?",
    "Invert the word 'racecar'",
    "Find the square root of 9",
    "Swap every digit with X in 'phone: 0612345678'",
    
    # Tricky — sounds like wrong function
    "What is 4 squared?",           # NOT add, NOT sqrt — actually substitute or none
    "Say the reverse of 'hello'",   # reverse, not greet
    "Greet the number 42",          # greet, not add
    "Add exclamation marks to 'hello'",  # substitute, not add
    "What's the root of evil?",     # tricky — square root?
    
    # Complex phrasing
    "Could you please add the numbers 15 and 30 together?",
    "Would you mind reversing 'OpenAI' for me?",
    "I want you to say hi to my colleague Sarah",
    "Please compute the square root of 256",
    "Change all spaces to underscores in 'hello world foo'",
    
    # Very short
    "5 + 3",
    "hi Mark",
    "reverse 'abc'",
    "sqrt 64",
    "replace a with b in 'banana'",
    
    # Ambiguous
    "Process the string 'hello'",
    "Do something with 4 and 9",
    "Handle 'cat' and 'dog'",
    "Transform 'hello world'",
    "Compute 144",
    
    # Multi-step sounding
    "Take 'hello' and give it back reversed",
    "Take 5 and 10 and tell me their total",
    "Introduce yourself to Jake",
    "Find what number times itself gives 49",
    "Remove all digits from 'abc123def456'",
    
    # Foreign phrasing
    "Bonjour to Marie",
    "Quanto fa 3 più 4?",
    "مرحبا بـ Ahmed",
    "Invertir la cadena 'mundo'",
    "Racine carrée de 100",
    
    # Edge cases
    "Reverse greet John",           # tricky — reverse or greet?
    "Add the string 'hello' and 'world'",  # tricky — add or substitute?
    "Square root of the sum of 9 and 16",  # two operations — which wins?
    "Greet everyone by reversing 'hello'", # tricky — greet or reverse?
    "Replace the number 0 with zero in '10 20 30'",  # substitute
]

for t in tests:
    print(f"{t}", end="--->")
    get_function_names(t)

