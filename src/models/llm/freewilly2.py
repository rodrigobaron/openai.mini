
from typing import List

from .llama import LLaMA
from src.utils.token import TokenFormatConfig


class FreeWilly2(LLaMA):
    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        return super().chat(messages, stream)