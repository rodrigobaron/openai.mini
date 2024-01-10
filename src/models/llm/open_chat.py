import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

from transformers import TextIteratorStreamer
from typing import List, Optional
from src.utils.env import compose_model_id
from src.models.llm.base import get_quant_config
from src.utils.request import parse_chat_kwargs

from .base import LlmModel
from src.type import ChatMessage
from src.utils.chat_template import TokenFormatConfig, format_tokens


class OpenChat(LlmModel):
    def load(self):
        super().load()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
    gen_kwargs = {"max_length": 1024, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}

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

class CodeNinjaOpenChat(OpenChat):
    def load(self):
        model_id = compose_model_id(self.id, prefix=self.org)
        print(f"Loading model {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained("openchat/openchat-3.5-1210", use_fast=True, **self.tokenizer_args)

        quantization_config = None
        if self.apply_quant:
            quantization_config = get_quant_config()

        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, **self.model_args)
        self.model.eval()
        print(f"Model {model_id} loaded!")

        return self
