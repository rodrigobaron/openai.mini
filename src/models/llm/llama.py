import torch
from threading import Thread
from transformers import TextIteratorStreamer
from typing import List, Optional
from src.utils.request import parse_chat_kwargs

from .base import LlmModel
from src.type import ChatMessage
from src.utils.chat_template import TokenFormatConfig, format_tokens


class LLaMA(LlmModel):
    def load(self):
        super().load()
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        return self
    
    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        streamer = _stream_chat(self.model, self.tokenizer, messages, self.token_format_config, **kwargs)
        if stream:
            return streamer, "delta"
        else:
            chunks = []
            for chunk in streamer:
                chunks.append(chunk)

            return "".join(chunks).strip(), None


def _stream_chat(model, tokenizer, messages: List[ChatMessage], config: TokenFormatConfig, **kwargs):
    gen_kwargs = _compose_args(tokenizer, messages, config, **kwargs)
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    return gen_kwargs["streamer"]

def _compose_args(tokenizer, messages: List[ChatMessage], config: TokenFormatConfig, **kwargs):
    gen_kwargs = {"max_length": 1024, "eos_token_id": tokenizer.eos_token_id}
    chat_kwargs = parse_chat_kwargs(**kwargs)
    if "max_length" in chat_kwargs.keys() and chat_kwargs["max_length"] is None:
        chat_kwargs.pop("max_length") # FIXME: Have an best way.. just bumping
    
    gen_kwargs.update(chat_kwargs)

    if config is None:
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        input_ids = torch.tensor(input_ids).long()
    else:
        input_ids = format_tokens(messages, tokenizer, config)
        input_ids = torch.tensor(input_ids).long()
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to("cuda")
    gen_kwargs["input_ids"] = input_ids

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer

    return gen_kwargs
