# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen2 model compatible with HuggingFace weights."""
import os
from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, extract_layer_index,
                    is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers,
                    maybe_prefix)
from vllm.forward_context import get_forward_context

inference_fused = False
if os.getenv("INFERENCE_OP_MODE") == "fused":
    inference_fused = True
    print(f"run in INFERENCE FUSED MODE")

    # 量化设置
    quantization_bit_mode = os.getenv("SYSHAX_QUANTIZE")
    quantization_bit_code = 1  # 默认是f16
    if quantization_bit_mode != "":
        if quantization_bit_mode == "q8_0":
            quantization_bit_code = 8
            print(f"Use q8_0 quantization!")
        elif quantization_bit_mode == "q4_0":
            quantization_bit_code = 2
            print(f"Use q4_0 quantization!")
        else:
            print(f"Unsupported quantization type !")


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        rope_scaling: Optional[tuple] = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=self.rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            } if dual_chunk_attention_config else {})

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen2Model(nn.Module):

    def __init__(self,
                 *,
                 vllm_config: VllmConfig,
                 prefix: str = "",
                 decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                ))

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2DecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2DecoderLayer
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(config=config,
                                              cache_config=cache_config,
                                              quant_config=quant_config,
                                              prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if (self.quant_config is not None and
                (scale_name := self.quant_config.get_cache_scale(name))):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = (loaded_weight if loaded_weight.dim() == 0 else
                                 loaded_weight[0])
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen2ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(
                                                  prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self._cpp_weight_loaded = False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if inference_fused:
            self.fused_forward = True

            # 获取全局 attention metadata
            forward_ctx = get_forward_context()
            attn_metadata = forward_ctx.attn_metadata
            # 收集各层的 KV cache 列表
            ve = forward_ctx.virtual_engine
            kv_caches = [
                layer.self_attn.attn.kv_cache[ve]
                for layer in self.model.layers[self.model.start_layer:self.model.end_layer]
            ]

            model = self.model
            block_size = 16
            N_tokens = len(input_ids)
            hidden_states = model.get_input_embeddings(input_ids)
            model_output = torch.zeros(
                (len(attn_metadata.seq_lens)
                 if attn_metadata.prefill_metadata else N_tokens, self.config.vocab_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )

            torch.ops._C.get_next_token_for_torch(
                model_output,
                hidden_states,
                attn_metadata.prefill_metadata is not None,
                attn_metadata.block_tables,
                attn_metadata.seq_lens_tensor,
                attn_metadata.slot_mapping.flatten(),
                positions,
                kv_caches,
                block_size,
                N_tokens,
                True,
                False
            )
            return model_output
        else:
            self.fused_forward = False

        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        if inference_fused and self.fused_forward:
            return hidden_states
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loaded = loader.load_weights(weights)
        if inference_fused:
            quant_method_gptq = (
                getattr(self, 'quant_config', None) and 
                getattr(self.quant_config, "quant_method", None) == "gptq"
            )
            config_quant_gptq = (
                getattr(self, 'config', None) and 
                getattr(self.config, 'quantization_config', None) and 
                self.config.quantization_config.get("quant_method") == "gptq"
            )
            if quant_method_gptq or config_quant_gptq:
                print("dequantize gptq")
                convert_gptq_to_fp16(self.model)
                self.quant_config=None
                if hasattr(self.config, "quantization_config"):
                    del self.config.quantization_config
            self._load_weight_to_cpp()
        return loaded

    def _load_weight_to_cpp(self):
        if self._cpp_weight_loaded:
            return
        qkv_proj_weight, qkv_proj_bias, o_proj_weight, gate_up_proj_weight, down_proj_weight = [], [], [], [], []
        input_layernorm_weight, post_attention_layernorm_weight = [], []
        norm_weight, lm_head_weight = [], []
        for i in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[i]

            input_layernorm_weight.append(layer.input_layernorm.weight)
            layer.input_layernorm.weight = None

            qkv_proj_weight.append(layer.self_attn.qkv_proj.weight)
            layer.self_attn.qkv_proj.weight = None

            if layer.self_attn.qkv_proj.bias is not None:
                qkv_proj_bias.append(layer.self_attn.qkv_proj.bias)
                layer.self_attn.qkv_proj.bias = None

            o_proj_weight.append(layer.self_attn.o_proj.weight)
            layer.self_attn.o_proj.weight = None

            post_attention_layernorm_weight.append(
                layer.post_attention_layernorm.weight)
            layer.post_attention_layernorm.weight = None

            gate_up_proj_weight.append(layer.mlp.gate_up_proj.weight)
            layer.mlp.gate_up_proj.weight = None

            down_proj_weight.append(layer.mlp.down_proj.weight)
            layer.mlp.down_proj.weight = None

        norm_weight.append(self.model.norm.weight)
        self.model.norm.weight = None
        lm_head_weight.append(self.lm_head.weight)

        if self.model.config.tie_word_embeddings:
            embed_tokens_weight = self.lm_head.weight
        else:
            embed_tokens_weight = self.model.embed_tokens.weight
            self.lm_head.weight = None

        first_attn = self.model.layers[0].self_attn
        input_layernorm_weight = torch.stack(input_layernorm_weight)
        qkv_proj_weight = torch.stack(qkv_proj_weight)
        qkv_proj_bias = torch.stack(qkv_proj_bias)
        o_proj_weight = torch.stack(o_proj_weight)
        post_attention_layernorm_weight = torch.stack(
            post_attention_layernorm_weight)
        gate_up_proj_weight = torch.stack(gate_up_proj_weight)
        down_proj_weight = torch.stack(down_proj_weight)
        norm_weight = torch.stack(norm_weight)
        lm_head_weight = torch.stack(lm_head_weight)

        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        torch.ops._C.load_model_config(
            self.model.config.model_type,
            head_dim, self.model.config.hidden_size, self.model.config.intermediate_size,
            self.model.config.num_attention_heads, self.model.config.num_hidden_layers, self.model.config.vocab_size,
            self.model.config.num_key_value_heads, self.model.config.sliding_window if self.model.config.use_sliding_window else self.model.config.max_position_embeddings,
            self.model.config.rms_norm_eps, self.model.config.rope_theta, first_attn.attn.impl.scale,
            first_attn.rotary_emb.is_neox_style, first_attn.rotary_emb.cos_sin_cache,
            quantization_bit_code
        )

        torch.ops._C.load_weight_unified(
            self.model.config.model_type,
            embed_tokens_weight,
            input_layernorm_weight,
            post_attention_layernorm_weight,
            qkv_proj_weight,
            o_proj_weight,
            qkv_proj_bias,
            gate_up_proj_weight,
            down_proj_weight,
            norm_weight,
            lm_head_weight,
            None,
            None
        )
        self._cpp_weight_loaded = True

def convert_gptq_to_fp16(model):
    """
    遍历模型所有层，把 GPTQ 量化权重转换成 FP16。
    假设模型里已经 load 了 safetensors 中的 qweight/scales/qzeros/g_idx/bias。
    """
    print("Converting GPTQ weights to FP16...")


    def unpack_qweight(qweight: torch.Tensor, bits: int = 8) -> torch.Tensor:
        """
        解包 GPTQ 打包的 int32 权重为 int8。
        每个 int32 含 4 个 int8，展开后第一维扩展为 4 倍。
        例如: [1280, 55296] -> [5120, 55296]
        """
        assert qweight.dtype == torch.int32, f"Expected int32, got {qweight.dtype}"

        orig_shape = qweight.shape
        
        # 方法1：使用torch.view（最快最准确）
        # try:
        # 将int32重新解释为uint8
        q_uint8 = qweight.view(torch.uint8)
        q = q_uint8.reshape(orig_shape[0], orig_shape[1], 4)
        # 重新排列并展开
        q = q.permute(0, 2, 1).contiguous()
        q = q.view(orig_shape[0] * 4, orig_shape[1])
        return q

    def unpack_qzeros(qzeros: torch.Tensor) -> torch.Tensor:
        """
        解包 GPTQ 打包的 qzeros，从 int32 -> int8。
        例如 [40, 1792] -> [40, 7168]
        """
        assert qzeros.dtype == torch.int32
        orig_shape = qzeros.shape
        bytes = torch.stack([
            (qzeros >> 0) & 0xFF,
            (qzeros >> 8) & 0xFF,
            (qzeros >> 16) & 0xFF,
            (qzeros >> 24) & 0xFF,
        ], dim=-1).to(torch.int8)
        return bytes.view(orig_shape[0], orig_shape[1] * 4)
    
    def dequantize(qweight, scales, qzeros, g_idx, bits=8, group_size=128, debug=True):
        """
        将 GPTQ 格式的量化权重反量化为浮点数矩阵。
        """
        torch.set_num_threads(1) 
        # 1️⃣ 解包 qweight 和 qzeros
        # 解包
        qzeros = unpack_qzeros(qzeros)   # [40, 55296]
        qweight = unpack_qweight(qweight)  # [5120, 55296]

        # block 参数
        block_size = qweight.shape[0] // qzeros.shape[0]  # 5120 // 40 = 128
        num_blocks = qzeros.shape[0]  # 40

        # 创建结果矩阵
        w_fp16 = torch.empty_like(qweight, dtype=torch.float32)

        # 分块反量化
        for block_idx in range(num_blocks):
            start = block_idx * block_size
            end = start + block_size

            # 当前块的量化参数
            scale_block = scales[block_idx, :].unsqueeze(0)   # [1, 55296]
            zero_block = qzeros[block_idx, :].unsqueeze(0)    # [1, 55296]

            # 对应块范围反量化
            w_fp16[start:end, :] = (qweight[start:end, :].to(torch.int32) - zero_block.to(torch.int32) - 1) * scale_block

        # 4️⃣ 转置以匹配线性层 [out_features, in_features]
        return w_fp16.T.half()  # [7168, 5120]
    
    # 遍历模型所有模块
    for name, module in model.named_modules():
        
        # 处理 self_attn 的 q/k/v/o_proj
        for proj in ["q_proj", "k_proj", "v_proj","qkv_proj", "o_proj"]:
            if hasattr(module, proj):
                m = getattr(module, proj)
                if all(hasattr(m, x) for x in ["qweight", "scales", "qzeros"]):
                    g_idx = getattr(m, "g_idx", None)
                    weight_fp16 = dequantize(m.qweight, m.scales, m.qzeros, g_idx,debug=True)
                    m.weight = torch.nn.Parameter(weight_fp16)
                    for attr in ["qweight", "scales", "qzeros", "g_idx"]:
                        if hasattr(m, attr):
                            delattr(m, attr)
                    
                

        
        # 处理 MLP 的 up/down/gate_proj
        for proj in ["up_proj", "down_proj", "gate_proj", "gate_up_proj"]:
            if hasattr(module, proj):
                m = getattr(module, proj)
                if all(hasattr(m, x) for x in ["qweight", "scales", "qzeros"]):
                    g_idx = getattr(m, "g_idx", None)
                    weight_fp16 = dequantize(m.qweight, m.scales, m.qzeros, g_idx,debug=True)
                    m.weight = torch.nn.Parameter(weight_fp16)
                    for attr in ["qweight", "scales", "qzeros", "g_idx"]:
                        if hasattr(m, attr):
                            delattr(m, attr)
        # LayerNorm 层直接转 FP16
        for ln_attr in ["input_layernorm", "post_attention_layernorm"]:
            if hasattr(module, ln_attr):
                ln = getattr(module, ln_attr)
                if hasattr(ln, "weight"):
                    ln.weight = ln.weight.half()
                if hasattr(ln, "bias") and ln.bias is not None:
                    ln.bias = ln.bias.half()
    
    print("Conversion to FP16 done!")

