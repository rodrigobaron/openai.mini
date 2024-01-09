from src.models.llm.llama import LLaMA
from typing import List


class DeepSeek(LLaMA):
    def load(self):
        super().load()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self
    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        return super().chat(messages, stream, **kwargs)