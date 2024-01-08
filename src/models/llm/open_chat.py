import torch
from threading import Thread
from transformers import TextIteratorStreamer
from typing import List, Optional

from .base import LlmModel
from src.type import ChatMessage

class OpenChat(LlmModel):
    def load(self):
        super().load()
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        return self
    
    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        streamer = _stream_chat(self.model, self.tokenizer, messages) #, **kwargs)
        if stream:
            return streamer, "delta"
        else:
            chunks = []
            for chunk in streamer:
                chunks.append(chunk)

            return "".join(chunks).strip(), None


def _stream_chat(model, tokenizer, messages: List[ChatMessage], **kwargs):
    gen_kwargs = _compose_args(tokenizer, messages)

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    return gen_kwargs["streamer"]

def _compose_args(tokenizer, messages: List[ChatMessage]):
    gen_kwargs = {"do_sample": True, "max_length": 1024, "temperature": 0.3,
                  "repetition_penalty": 1.2, "top_p": 0.95, "eos_token_id": tokenizer.eos_token_id,
                  "pad_token_id": tokenizer.eos_token_id}

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

    input_ids = torch.tensor(input_ids).long()
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to("cuda")
    gen_kwargs["input_ids"] = input_ids

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer

    return gen_kwargs