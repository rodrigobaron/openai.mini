
from typing import List

from .llama import LLaMA


class Phi(LLaMA):
    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        return super().chat(messages, stream)