
from typing import List

from .llama import LLaMA
from src.utils.token import TokenFormatConfig

_SYSTEM_PROMPT = ""

_token_format_config = TokenFormatConfig(
    SYSTEM_PROMPT=_SYSTEM_PROMPT,
    B_INST="Instruct: ", E_INST="\n",
    B_AI="Output: ", E_AI="\n")


class Phi(LLaMA):
    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        return super().chat(messages, stream, token_format_config=_token_format_config) #, **kwargs)