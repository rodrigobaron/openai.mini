from pydantic import BaseModel

# ["vicuna","alpaca","chatml","llama2-chat","oasst"]

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

# TokenFormatConfig
class TokenFormatConfig(BaseModel):
    SYSTEM_PROMPT: str = DEFAULT_SYSTEM_PROMPT
    B_SYS: str = ""
    E_SYS: str = ""
    B_INST: str = ""
    E_INST: str = ""
    B_AI: str = ""
    E_AI: str = ""

class Llama2TokenFormatConfig(TokenFormatConfig):
    B_SYS: str = "<<SYS>>\n"
    E_SYS: str = "\n<</SYS>>\n\n"
    B_INST: str = "[INST]"
    E_INST: str = "[/INST]"
    B_AI: str = ""
    E_AI: str = ""


class VicunaTokenFormatConfig(TokenFormatConfig):
    B_SYS: str = ""
    E_SYS: str = "\n\n"
    B_INST: str = "USER: "
    E_INST: str = "\n"
    B_AI: str = "ASSISTANT: "
    E_AI: str = "</s>\n\n"


class AlpacaTokenFormatConfig(TokenFormatConfig):
    B_SYS: str = ""
    E_SYS: str = "\n\n"
    B_INST: str = "user:\n"
    E_INST: str = "</s>\n"
    B_AI: str = "assistant:"
    E_AI: str = "</s>\n"


class ChatMLTokenFormatConfig(TokenFormatConfig):
    B_SYS: str = "<|im_start|>system\n"
    E_SYS: str = "\n<|im_end|>\n"
    B_INST: str = "<|im_start|>user\n"
    E_INST: str = "\n<|im_end|>\n"
    B_AI: str = "<|im_start|>assistant\n"
    E_AI: str = "\n<|im_end|>\n"


class FreeTokenFormatConfig(TokenFormatConfig):
    B_INST: str = "Instruct: " 
    E_INST: str ="\n"
    B_AI: str = "Output: "
    E_AI: str = "\n"

class OpenChatTokenFormatConfig(TokenFormatConfig):
    B_INST: str = "GPT4 Correct User: " 
    E_INST: str ="<|end_of_turn|>"
    B_AI: str = "GPT4 Correct Assistant: "
    E_AI: str = "<|end_of_turn|>"


def build_chat_template(token_format_config):
    chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{%if message['role'] == 'system' %}{{'" + token_format_config.B_SYS + "' + message['content'] + '" + token_format_config.E_SYS + "'}}{% endif %}{%if message['role'] == 'user' %}{{'" + token_format_config.B_INST + "' + message['content'] + '" + token_format_config.E_INST + "'}}{% endif %}{%if message['role'] == 'assistant' %}{{'" + token_format_config.B_AI + "' + message['content'] + '" + token_format_config.E_AI + "'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    return chat_template


def format_tokens(messages, tokenizer, config):
    prompt = ""
    for h in  messages:
        if h.role == 'system':
            prompt = f"{config.B_SYS}{h.content}{config.E_SYS}"
        if h.role == 'user':
            prompt += f"{config.B_INST}{h.content}{config.E_INST}"
        if h.role == 'assistant':
            prompt += f"{config.B_AI}{h.content}{config.E_AI}"
    prompt += f"\n{config.B_AI}"
    return tokenizer.encode(prompt)

