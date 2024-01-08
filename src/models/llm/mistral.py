
from typing import List

from .llama import LLaMA
from src.utils.token import TokenFormatConfig
from .base import LlmModel
from src.utils.env import compose_model_id
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig

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
            device='cuda',
            quant_config=quant_config,
            offload_config=offload_config,
            state_path=state_path,
        )
        
        self.model.cuda().eval()
        print(f"Model {model_id} loaded!")

        return self

    def chat(self, messages: List[str], stream: bool = False, **kwargs):
        return super().chat(messages, stream, token_format_config=_token_format_config) #, **kwargs)
