# Copyright 2024 Tsinghua University and ByteDance.
#
# Licensed under the MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Reference: vLLM (https://github.com/vllm-project/vllm)

"""Inference-only Qwen2-ChatTS model compatible with HuggingFace weights."""
from functools import cached_property
from typing import (Any, Iterable, List, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchFeature, ProcessorMixin

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.inputs import InputContext
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import NestedTensors
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        MultiModalDataItems, ProcessorInputs,
                                        PromptReplacement)
from vllm.sequence import IntermediateTensors
from vllm.multimodal.base import MultiModalPlugin
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix, merge_multimodal_embeddings, WeightsMapper)
from vllm import ModelRegistry

from chatts.vllm.ts_encoder import TimeSeriesEmbedding, get_patch_cnt


# === Plugin ===
class TimeSeriesPlugin(MultiModalPlugin):
    """Plugin for timeseries data."""

    def get_data_key(self) -> str:
        return "timeseries"

    def _default_input_mapper(
        self,
        ctx: InputContext,
        data,
        **mm_processor_kwargs,
    ):
        raise NotImplementedError("There is no default timeseries input mapper")

    def _default_max_multimodal_tokens(self, ctx: InputContext) -> int:
        raise NotImplementedError(
            "There is no default maximum multimodal tokens")

timeseries_plugin = TimeSeriesPlugin()
MULTIMODAL_REGISTRY.register_plugin(timeseries_plugin)
print("[ChatTS VLLM] TimeSeriesPlugin is registered.")


def get_max_qwen2_ts_ts_tokens(ctx: InputContext) -> int:
    hf_config = ctx.get_hf_config()
    output_lengths = hf_config.ts['max_length'] // hf_config.ts['patch_size']
    return output_lengths


class Qwen2TSMultiModalProcessor(BaseMultiModalProcessor):
    def _get_hf_processor(self):
        return self.ctx.get_hf_processor()

    def _call_hf_processor(
        self,
        hf_processor: ProcessorMixin,
        prompt: str,
        processor_data: Mapping[str, object],
        mm_processor_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        processor_data = dict(processor_data)
        ts = processor_data.pop("timeseries", [])

        if ts:
            processor_data["timeseries"] = ts

        mm_processor_kwargs['vllm_flag'] = True
        result = super()._call_hf_processor(
            hf_processor,
            prompt=prompt,
            processor_data=processor_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )

        return result

    def _get_prompt_replacements(
        self,
        mm_items: MultiModalDataItems,
        hf_inputs: BatchFeature,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[PromptReplacement]:
        hf_config = self.ctx.get_hf_config()
        placeholder = hf_config.ts_token_start_index

        if hf_inputs['timeseries'] is None or len(hf_inputs['timeseries']) == 0:
            patch_cnt = []
        else:
            patch_cnt = get_patch_cnt(hf_inputs['timeseries'], hf_config.ts)

        def get_replacement_qwen2_ts(item_idx: int):
            return [placeholder] * patch_cnt[item_idx]

        return [
            PromptReplacement(
                modality="timeseries",
                target=[placeholder],
                replacement=get_replacement_qwen2_ts,
            )
        ]

    def _get_dummy_mm_inputs(
        self,
        mm_counts: Mapping[str, int],
    ) -> ProcessorInputs:
        hf_config = self.ctx.get_hf_config()
        max_ts_length = hf_config.ts['max_length']
        ts_count = mm_counts["timeseries"]
        ts = np.zeros(max_ts_length)
        data = {"timeseries": [ts] * ts_count}

        return ProcessorInputs(
            prompt_text="<ts><ts/>" * ts_count,
            mm_data=data,
            mm_processor_kwargs={},
        )


@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    "timeseries", get_max_qwen2_ts_ts_tokens)
@MULTIMODAL_REGISTRY.register_processor(Qwen2TSMultiModalProcessor)
class Qwen2TSForCausalLM(nn.Module, SupportsMultiModal,
                                         SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.multimodal_config = multimodal_config

        self.ts_encoder = TimeSeriesEmbedding(config.ts)
        self.quant_config = quant_config

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @cached_property
    def sampler(self):
        if hasattr(self.language_model, "sampler"):
            return self.language_model.sampler

        return get_sampler()

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                        name: str) -> torch.Tensor:
        if not isinstance(mm_input, (torch.Tensor, list)):
            raise ValueError(f"Incorrect type of {name}. "
                             f"Got type: {type(mm_input)}")
        if isinstance(mm_input, torch.Tensor):
            return torch.concat(list(mm_input))
        else:
            return torch.concat(mm_input)

    def _parse_and_validate_ts_input(
            self, **kwargs: object) -> torch.Tensor:
        input_features = kwargs.pop('timeseries', None)
        if input_features is None:
            return None
        input_features = self._validate_and_reshape_mm_tensor(
            input_features, 'timeseries')
        if not isinstance(input_features, (torch.Tensor, list)):
            raise ValueError("Incorrect type of ts input features. "
                             f"Got type: {type(input_features)}")
        return input_features

    def _process_ts_input(self,
                             timeseries: torch.Tensor) -> torch.Tensor:
        ts_features, patch_cnt = self.ts_encoder(timeseries)
        return ts_features, patch_cnt

    def get_multimodal_embeddings(self, **kwargs) -> Optional[NestedTensors]:
        ts_input = self._parse_and_validate_ts_input(**kwargs)
        if ts_input is None:
            return None, None
        ts_features, patch_cnt = self._process_ts_input(ts_input)
        return ts_features, patch_cnt

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[NestedTensors] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.ts_token_start_index)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            ts_features, patch_cnt = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids, ts_features)
            input_ids = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)

        autoloaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

        # The HF config doesn't specify whether these are tied,
        # so we detect it this way
        if "embed_tokens.weight" not in autoloaded_weights:
            self.embed_tokens = self.language_model.model.embed_tokens
            autoloaded_weights.add("embed_tokens.weight")

        return autoloaded_weights

# Register VLLM
ModelRegistry.register_model("Qwen2TSForCausalLM", Qwen2TSForCausalLM)
print(f"[ChatTS VLLM] Qwen2TSForCausalLM registered in vLLM!")
