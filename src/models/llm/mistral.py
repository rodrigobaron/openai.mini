
from typing import List

from .llama import LLaMA
from src.utils.token import TokenFormatConfig
from .base import LlmModel
from src.utils.env import compose_model_id
from hqq.core.quantize import BaseQuantizeConfig
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig
import torch
from typing import List, Optional
from src.type import ChatMessage


from src.contrib.offload.build_model import OffloadConfig, QuantConfig, build_model

_SYSTEM_PROMPT = ""

_token_format_config = TokenFormatConfig(
    SYSTEM_PROMPT=_SYSTEM_PROMPT,
    B_INST="[INST]", E_INST="[/INST]",
)

class Mistral(LLaMA):
    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        return super().chat(messages, stream, token_format_config=_token_format_config) #, **kwargs)
    
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
        print(f"Model {model_id} loaded!")

        return self

    def chat(self, messages: List[str], stream: bool = False, token_format_config: Optional[TokenFormatConfig] = None, **kwargs):
        streamer = _stream_chat(self.model, self.tokenizer, messages, token_format_config) #, **kwargs)
        if stream:
            return streamer, "delta"
        else:
            chunks = []
            for chunk in streamer:
                chunks.append(chunk)

            return "".join(chunks).strip(), None


def _stream_chat(model, tokenizer, messages: List[ChatMessage], token_format_config: TokenFormatConfig = None, **kwargs):
    gen_kwargs = _compose_args(tokenizer, messages, token_format_config)

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    return gen_kwargs["streamer"]

def _compose_args(tokenizer, messages: List[ChatMessage], token_format_config: TokenFormatConfig = None):
    gen_kwargs = {"do_sample": True, "max_length": 8000, "temperature": 0.3,
                  "repetition_penalty": 1.2, "top_p": 0.95, "eos_token_id": tokenizer.eos_token_id}

    config = token_format_config if token_format_config is not None else TokenFormatConfig()
    chat = format_tokens(messages, tokenizer, config)
    input_ids = torch.tensor(chat).long()
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to("cuda")
    gen_kwargs["input_ids"] = input_ids

    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs["streamer"] = streamer

    return gen_kwargs
