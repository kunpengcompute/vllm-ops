# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
"""Inference-only Qwen3 model compatible with HuggingFace weights."""
import os
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix
from vllm.forward_context import get_forward_context

logger = init_logger(__name__)

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


class Qwen3Attention(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_heads: int,
                 num_kv_heads: int,
                 max_position: int = 4096 * 32,
                 head_dim: Optional[int] = None,
                 rms_norm_eps: float = 1e-06,
                 qkv_bias: bool = False,
                 rope_theta: float = 10000,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 rope_scaling: Optional[tuple] = None,
                 prefix: str = "",
                 attn_type: str = AttentionType.DECODER) -> None:
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
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
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
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config,
                              prefix=f"{prefix}.attn",
                              attn_type=attn_type)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_scaling=rope_scaling,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = Qwen3MLP(
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


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    })
class Qwen3Model(Qwen2Model):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config,
                         prefix=prefix,
                         decoder_layer_type=Qwen3DecoderLayer)


class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
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
        self.model = Qwen3Model(vllm_config=vllm_config,
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
                False,
                True
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
            self._load_weight_to_cpp()
        return loaded

    def _load_weight_to_cpp(self):
        if self._cpp_weight_loaded:
            return

        qkv_proj_weight, qkv_proj_bias, o_proj_weight, gate_up_proj_weight, down_proj_weight = [], [], [], [], []
        q_norm_weight, k_norm_weight = [], []
        input_layernorm_weight, post_attention_layernorm_weight = [], []
        norm_weight, lm_head_weight = [], []
        for i in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[i]

            input_layernorm_weight.append(layer.input_layernorm.weight)
            layer.input_layernorm.weight = None

            qkv_proj_weight.append(layer.self_attn.qkv_proj.weight)
            layer.self_attn.qkv_proj.weight = None

            if layer.self_attn.q_norm.weight is not None:
                q_norm_weight.append(layer.self_attn.q_norm.weight)
                layer.self_attn.q_norm.weight = None

            if layer.self_attn.k_norm.weight is not None:
                k_norm_weight.append(layer.self_attn.k_norm.weight)
                layer.self_attn.k_norm.weight = None

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
        q_norm_weight = torch.stack(q_norm_weight)
        k_norm_weight = torch.stack(k_norm_weight)

        if len(qkv_proj_bias) > 0:
            qkv_proj_bias = torch.stack(qkv_proj_bias)
        else:
            device = qkv_proj_weight.device
            dtype = qkv_proj_weight.dtype
            qkv_dim = qkv_proj_weight.shape[-1]
            n_layers = qkv_proj_weight.shape[0]
            qkv_proj_bias = torch.zeros(
                n_layers, qkv_dim, device=device, dtype=dtype)

        o_proj_weight = torch.stack(o_proj_weight)
        post_attention_layernorm_weight = torch.stack(
            post_attention_layernorm_weight)
        gate_up_proj_weight = torch.stack(gate_up_proj_weight)
        down_proj_weight = torch.stack(down_proj_weight)
        norm_weight = torch.stack(norm_weight)
        lm_head_weight = torch.stack(lm_head_weight)

        torch.ops._C.load_model_config(
            self.model.config.model_type,
            self.model.config.head_dim, self.model.config.hidden_size, self.model.config.intermediate_size,
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
            q_norm_weight,
            k_norm_weight
        )
        self._cpp_weight_loaded = True
