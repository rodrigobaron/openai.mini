
from fastapi import HTTPException

def raise_if_invalid_model(model, type):
    if model is None or not isinstance(model, type):
        raise HTTPException(status_code=400, detail="Invalid request model")


def parse_chat_kwargs(**kwargs):
    allowed_keys = ['max_tokens', 'presence_penalty', 'seed', 'stop', 'temperature', 'top_p']

    not_implemented = ['frequency_penalty', 'logit_bias', 'logprobs', 'top_logprobs',]
    keep_same = ['logit_bias', 'logprobs', 'top_logprobs', 'max_tokens', 'presence_penalty', 'seed', 'temperature', 'top_p']
    
    return_kwargs = {}
    for k in kwargs.keys():
        if k not in allowed_keys:
            continue

        if k in not_implemented:
            continue # :(

        if k in keep_same:
            return_kwargs[k] = kwargs[k]

        if k == 'stop':
            return_kwargs['stopping_criteria'] = kwargs[k]
        
        if k == 'frequency_penalty':
            return_kwargs['repetition_penalty'] = kwargs[k]
    
    do_sample = False
    if 'stopping_criteria' in return_kwargs.keys() and return_kwargs['stopping_criteria'] == True:
        do_sample = True

    if 'temperature' in return_kwargs.keys() and return_kwargs['temperature'] > 0:
        do_sample = True
    
    return_kwargs['do_sample'] = do_sample

    return return_kwargs