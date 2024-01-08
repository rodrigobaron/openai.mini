from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Optional
import torch

from src.utils.env import compose_model_id
from src.utils.chat_template import build_chat_template
from .base import LlmModel, split_messages

class Qwen(LlmModel):
    def load(self):
        model_id = compose_model_id(self.id, prefix=self.org)
        print(f"Loading model {model_id}")
        quantization_config = get_quant_config()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, **self.tokenizer_args)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, **self.model_args)
        if 'fp16' in self.model_args:
            self.model.bfloat16().eval()
        else:
            self.model.eval()
        if self.token_format_config is not None:
            self.tokenizer.chat_template = build_chat_template(self.token_format_config)
        print(f"Model {model_id} loaded!")

        return self

    def chat(self, messages: List[str], stream: Optional[bool] = False, **kwargs):
        query, history = split_messages(messages)
        if stream:
            response = self.model.chat_stream(self.tokenizer, query, history, **kwargs)
            return response, self.stream_type
        else:
            response = self.model.chat(self.tokenizer, query, history, **kwargs)
            return response

def get_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
