
from typing import List

from .llama import LLaMA
from src.utils.chat_template import build_chat_template
from threading import Thread
from transformers import TextIteratorStreamer
from .base import LlmModel
from src.utils.env import compose_model_id
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig
import torch
from typing import List, Optional
from src.type import ChatMessage
from src.utils.request import parse_chat_kwargs


from src.contrib.offload.build_model import OffloadConfig, QuantConfig, build_model

class Mistral(LLaMA):
    def load(self):
        super().load()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self
    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        return super().chat(messages, stream, **kwargs)
    
class MixtralOffload(LlmModel):
    def load(self):
        model_id = compose_model_id(self.id, prefix=self.org)
        print(f"Loading model {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1', trust_remote_code=True)
        # self.model = AutoModel.from_pretrained(model_id, device_map="cuda", trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_id)
        state_path = snapshot_download(model_id)


        ##### Change this to 5 if you have only 12 GB of GPU VRAM #####
        offload_per_layer = 4
        # offload_per_layer = 5
        ###############################################################

        num_experts = config.num_local_experts

        offload_config = OffloadConfig(
            main_size=config.num_hidden_layers * (num_experts - offload_per_layer),
            offload_size=config.num_hidden_layers * offload_per_layer,
            buffer_size=4,
            offload_per_layer=offload_per_layer,
        )

        attn_config = BaseQuantizeConfig(
            nbits=4,
            group_size=64,
            quant_zero=True,
            quant_scale=True,
        )
        attn_config["scale_quant_params"]["group_size"] = 256

        ffn_config = BaseQuantizeConfig(
            nbits=2,
            group_size=16,
            quant_zero=True,
            quant_scale=True,
        )
        quant_config = QuantConfig(ffn_config=ffn_config, attn_config=attn_config)

        self.model = build_model(
            device=torch.device("cuda:0"),
            quant_config=quant_config,
            offload_config=offload_config,
            state_path=state_path,
        )
        
        self.model.cuda().eval()
        if self.token_format_config is not None:
            self.tokenizer.chat_template = build_chat_template(self.token_format_config)
        print(f"Model {model_id} loaded!")

        return self

    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        msgs = [_chat_message_to_mistral_message(m) for m in messages]
        streamer = _stream_chat(self.model, self.tokenizer, msgs **kwargs)
        if stream:
            return streamer, "delta"
        else:
            chunks = []
            for chunk in streamer:
                chunks.append(chunk)

            return "".join(chunks).strip(), None


def _stream_chat(model, tokenizer, messages: List[ChatMessage], **kwargs):
    gen_kwargs = _compose_args(tokenizer, messages, **kwargs)

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    return gen_kwargs["streamer"]

def _compose_args(tokenizer, messages: List[ChatMessage], **kwargs):
    gen_kwargs = {"max_length": 1024, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id}
    chat_kwargs = parse_chat_kwargs(**kwargs)
    gen_kwargs.update(chat_kwargs)

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    input_ids = torch.tensor(input_ids).long()
    # input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to("cuda")
    gen_kwargs["input_ids"] = input_ids

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer

    return gen_kwargs

def _chat_message_to_mistral_message(message: ChatMessage):
    return {
        "role": message.role if message.role == "assistant" else "user",  # "system" role is not supported by Mixtral
        "content": message.content
    }
