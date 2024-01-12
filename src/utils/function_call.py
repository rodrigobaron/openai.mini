import json
from typing import Tuple, Union

from ..type import ChatMessage, DeltaMessage, FunctionCallResponse

OBSERVATION = "Observation"

TOOL_DESC_OLD = """{name}: Call this tool to interact with the {name} API. \
What is the {name} API useful for? \
{description} \
Parameters: {parameters}. Format the arguments as a JSON object."""


# Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
# REACT_PROMPT_OLD = """Answer the following questions as best you can. You have access to the following tools:

# {tool_descs}

# Use the following format:

# Question: the input question you must answer
# Thought: you should always think about what to do
# Action: the action to take, should be one of [{tool_names}]
# Action Input: the input to the action
# {OBSERVATION}: the result of the action
# ... (this Thought/Action/Action Input/{OBSERVATION} can be repeated zero or more times)
# Thought: I now know the final answer
# Final Answer: the final answer to the original input question, markdown format is preferred in final answer.

# Begin!

# Question: {query}"""

TOOL_PARAM_DESC = """{param_name}: {param_type}
        {param_description}
"""

# name: str
# type: str
# required: Optional[bool] = False
# description: str

def _build_tool_param(params):
    props = [{'name': k, 'type': v['type'], 'description': v['description']} for k, v in params["properties"].items()]

    params_fmt = [TOOL_PARAM_DESC.format(param_name=p['name'], param_type=p['type'], param_description=p['description']) for p in props]
    return "".join(params_fmt)


TOOL_DESC = """
* {name}: 
{description}
Parameters
------------
{parameters}
"""

REACT_PROMPT = """Imagine there are experts to help you to answering user questions as best they can.
The experts have functions which they can use to help to get the right answer. The user cannot see or use the functions themselves, nor can they know the
process of your function usage. Provide all necessary information in the
"Final Answer" field. If function parameters are missing, use the "getDetails" function to ask the user for them.

### You have access to the following functions (use json format as input):

* getDetails: 
Call this function to get parameter information.
Parameters
------------
    query: string
        The question to to ask. Format the arguments as a JSON object.
{tool_descs}

### The experts use the following format:
Thought: think step by step about what to do to get the right answer
Before Action: think before act, if need more thinking go back to Thought
Action: the action to take, should be one of [getDetails, {tool_names}]
Action Input: the input to the action
{OBSERVATION}: the result of the action
Verify: Check if all steps so far are enough to get the answer
... (this Thought/Before Action/Action/Action Input/{OBSERVATION}/Verify can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, markdown format is preferred in final answer.

Just use experts when needed otherwise just answer the user.

"""

def need_function_call(messages, functions):
    if functions is not None and len(functions) > 0:
        return True
    if messages is not None and len(messages) > 0 and messages[-1].role == "function":
        return True
    return False

# def build_function_call_messages(messages, functions, function_call="auto"):
#     if messages is None or len(messages) == 0:
#         return None
#     if functions is None or function_call == 'none':
#         return messages[-1]
#     if function_call != "auto" and isinstance(function_call, dict):
#         functions = [f for f in functions if f.name in [function_call.name]]

#     tool_descs, tool_names = [], []
#     for f in functions:
#         tool_descs.append( TOOL_DESC.format( name=f.name, description=f.description, parameters=json.dumps(f.parameters, ensure_ascii=False)))
#         tool_names.append(f.name)
#     tool_descs = "\n\n".join(tool_descs)
#     tool_names = ", ".join(tool_names)

def build_function_call_messages(messages, functions, function_call="auto"):
    if messages is None or len(messages) == 0:
        return None
    if functions is None or function_call == 'none':
        return messages[-1]
    if function_call != "auto" and isinstance(function_call, dict):
        functions = [f for f in functions if f.name in [function_call.name]]

    tool_descs, tool_names = [], []
    for f in functions:
        tool_descs.append( TOOL_DESC.format( name=f.name, description=f.description, parameters=_build_tool_param(f.parameters)) )
        tool_names.append(f.name)
    tool_descs = "\n\n".join(tool_descs)
    tool_names = ", ".join(tool_names)
    
    last = ""
    for index, message in enumerate(reversed(messages)):
        if message.role == "user":
            # last = _build_react_message(message, tool_descs, tool_names, OBSERVATION) + last
            last = message.content + last
            break
        elif message.role == "assistant":
            if message.function_call:
                last = _build_function_call_message(message) + last
        elif message.role == "function":
            last = _build_function_message(message, OBSERVATION) + last

    converted = [ChatMessage(role="user", content=last)]
    # FIXME: just filter out the other messages
    for i, message in enumerate(reversed(messages[:-(index+1)])):
        if message.role == 'user' or (message.role == 'assistant' and message.function_call == None):
            converted.append(message)

    converted.append(ChatMessage(role="system", content=_build_react_message(message, tool_descs, tool_names, OBSERVATION)))
    return [x for x in reversed(converted)]


# def _build_react_message(message, tool_descs, tool_names, OBSERVATION):
#     return REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, query=message.content, OBSERVATION=OBSERVATION)


def _build_react_message(tool_descs, tool_names, OBSERVATION):
    return REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names, OBSERVATION=OBSERVATION)

def _build_function_message(message, OBSERVATION):
    return f"\n{OBSERVATION}: {str(message.content).strip()}"

def _build_function_call_message(message):
    function_name = message.function_call.name
    arguments = message.function_call.arguments
    this_part = f"\nThought: I should call {function_name} with {arguments.strip()}"
    this_part += f"\nAction: {function_name.strip()}"
    this_part += f"\nAction Input: {arguments.strip()}"
    return this_part


def build_chat_message(response: str) -> ChatMessage:
    parsed = _parse_qwen_plugin_call(response)
    if parsed is None:
        return ChatMessage(role="assistant", content=response), "stop"
    else:
        name, args, final = parsed
        if final:
            return ChatMessage(role="assistant", content=final), "stop"
        else:
            function_call = FunctionCallResponse(name=name, arguments=args)
            return ChatMessage(role="assistant", content=None, function_call=function_call), "function_call"

def build_fc_name_message(text: str) -> DeltaMessage:
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    name = text[i + len('\nAction:'): j].strip()
    return DeltaMessage(function_call=FunctionCallResponse(name=name, arguments=""))

def build_fc_args_message(delta: str) -> DeltaMessage:
    return DeltaMessage(function_call=FunctionCallResponse(arguments=delta))

def _parse_qwen_plugin_call(text: str) -> Union[Tuple[str, str], None]:
    i = text.rfind('\nAction:')
    j = text.rfind('\nAction Input:')
    k = text.rfind('\nObservation:')
    l = text.rfind('\nFinal Answer:')

    if l >= 0:
        final = text[l + len('\nFinal Answer:'):].strip()
        return None, None, final

    if 0 <= i < j:  # If the text has `Action` and `Action input`,
        if k < j:  # but does not contain `Observation`,
            # then it is likely that `Observation` is ommited by the LLM,
            # because the output text may have discarded the stop word.
            text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')

    if 0 <= i < j < k:
        name = text[i + len('\nAction:'): j].strip()
        args = text[j + len('\nAction Input:'): k].strip()
        return name, args, None
    return None
