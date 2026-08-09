from __future__ import annotations

import json
import typing as tp

import safetensors
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import PreTrainedModel
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import MistralModel
from transformers.models.mistral3.configuration_mistral3 import Mistral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

from ...torch_transfer_utils import resolve_pin_memory
from .linear import W4Linear
from .qwen_common import (
    collect_qwen_quantized_prefixes,
    load_qwen_runtime_state_dict,
    materialize_meta_module_tensors,
    register_rotary_materializer,
    resolve_checkpoint_path,
)

MISTRAL3_MULTIMODAL_GENERATE = "mistral3_multimodal_generate"
MISTRAL3_PROMPT_EMBEDS_ONLY = "mistral3_prompt_embeds_only"

__all__ = ["NunchakuMistral3EncoderModel"]


def _parse_json_metadata(metadata: dict[str, str], key: str, default: tp.Any) -> tp.Any:
    value = metadata.get(key)
    if value is None or not str(value).strip():
        return default
    return json.loads(value)


def _resolve_group_size(metadata: dict[str, str]) -> int:
    quantization_config = _parse_json_metadata(metadata, "quantization_config", {})
    group_size = int(quantization_config.get("weight", {}).get("group_size", -1))
    if group_size <= 0:
        raise ValueError(f"Invalid Mistral3 runtime group size: {group_size}")
    return group_size


def _resolve_attn_implementation(config_dict: dict[str, tp.Any], nested_key: str | None) -> str:
    if nested_key is None:
        value = config_dict.get("_attn_implementation")
    else:
        nested = config_dict.get(nested_key)
        value = nested.get("_attn_implementation") if isinstance(nested, dict) else None
        if value is None:
            value = config_dict.get("_attn_implementation")
    return str(value or "eager")


def _load_runtime_contract(metadata: dict[str, str]) -> str:
    runtime_contract = str(metadata.get("text_encoder_usage", metadata.get("runtime_contract", ""))).strip().lower()
    return runtime_contract or MISTRAL3_MULTIMODAL_GENERATE


def _load_text_config_dict(metadata: dict[str, str]) -> dict[str, tp.Any]:
    config_dict = _parse_json_metadata(metadata, "config", {})
    base_text_config = _parse_json_metadata(metadata, "base_text_config", {})
    if isinstance(base_text_config, dict) and base_text_config:
        return dict(base_text_config)
    if isinstance(config_dict, dict) and isinstance(config_dict.get("text_config"), dict):
        return dict(config_dict["text_config"])
    raise ValueError("Missing or invalid `base_text_config` metadata in Mistral3 checkpoint.")


def _patch_quantized_linears(
    module: nn.Module,
    *,
    quantized_prefixes: set[str],
    group_size: int,
    linear_dtype: torch.dtype | None = None,
) -> None:
    if not quantized_prefixes:
        raise ValueError("Checkpoint does not contain any quantized linear prefixes.")
    named_modules = dict(module.named_modules())
    missing: list[str] = []
    for module_name in sorted(quantized_prefixes):
        target = named_modules.get(module_name)
        if target is None:
            missing.append(module_name)
            continue
        if not isinstance(target, nn.Linear):
            raise TypeError(f"Expected `{module_name}` to be nn.Linear, got {type(target)}")
        if linear_dtype is not None and target.weight.dtype != linear_dtype:
            target.weight.data = target.weight.data.to(dtype=linear_dtype)
        qmodule = W4Linear.from_linear(target, group_size=group_size, init_only=True)
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = module.get_submodule(parent_name)
        setattr(parent, child_name, qmodule)
    if missing:
        raise RuntimeError(
            "Checkpoint expects quantized Mistral3 linears that were not found in the runtime model: "
            f"{missing[:20]}"
        )


def _restore_model_tensors(
    handle: safetensors.safe_open,
    *,
    model: nn.Module,
    torch_dtype: torch.dtype,
    output_device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> tuple[list[str], list[str]]:
    state_dict = load_qwen_runtime_state_dict(
        handle,
        torch_dtype=torch_dtype,
        output_device=output_device,
        pin_memory=pin_memory,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    materialize_meta_module_tensors(model, device=output_device or "cpu")
    return missing, unexpected


def _materialize_rope_with_init_fn(owner: nn.Module, name: str, device: torch.device) -> torch.Tensor | None:
    if name not in ("inv_freq", "original_inv_freq"):
        return None
    config = getattr(owner, "config")
    rope_type = str(getattr(owner, "rope_type", config.rope_parameters["rope_type"]))
    rope_init_fn = owner.compute_default_rope_parameters
    if rope_type != "default":
        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
    inv_freq, attention_scaling = rope_init_fn(config, device)
    owner.attention_scaling = attention_scaling
    if name == "original_inv_freq":
        return inv_freq.clone()
    return inv_freq


register_rotary_materializer("MistralRotaryEmbedding")(_materialize_rope_with_init_fn)
register_rotary_materializer("PixtralRotaryEmbedding")(_materialize_rope_with_init_fn)


class NunchakuMistral3MultimodalEncoderModel(Mistral3ForConditionalGeneration):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        pin_memory: bool | str = "auto",
    ) -> "NunchakuMistral3MultimodalEncoderModel":
        ckpt = resolve_checkpoint_path(pretrained_model_name_or_path)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        device = device if isinstance(device, torch.device) else torch.device(device)
        pin_memory_enabled = resolve_pin_memory(pin_memory, device)

        with safetensors.safe_open(str(ckpt), framework="pt", device="cpu") as handle:
            raw_metadata = dict(handle.metadata() or {})
            config_dict = _parse_json_metadata(raw_metadata, "config", {})
            if not isinstance(config_dict, dict) or not config_dict:
                raise ValueError("Missing or invalid `config` metadata in Mistral3 checkpoint.")
            runtime_contract = _load_runtime_contract(raw_metadata)
            model_type = str(raw_metadata.get("model_type", "")).strip().lower()
            if runtime_contract != MISTRAL3_MULTIMODAL_GENERATE:
                raise ValueError(
                    "Unsupported Mistral3 multimodal runtime contract. "
                    f"Expected `{MISTRAL3_MULTIMODAL_GENERATE}`, got `{runtime_contract or '<missing>'}`."
                )
            if model_type != "mistral3_text":
                raise ValueError(f"Unsupported Mistral3 model_type metadata: `{model_type or '<missing>'}`.")
            group_size = _resolve_group_size(raw_metadata)
            config = Mistral3Config(**config_dict)
            config._attn_implementation = _resolve_attn_implementation(config_dict, None)
            if hasattr(config, "text_config"):
                config.text_config._attn_implementation = _resolve_attn_implementation(config_dict, "text_config")
            if hasattr(config, "vision_config"):
                config.vision_config._attn_implementation = _resolve_attn_implementation(config_dict, "vision_config")
            with init_empty_weights():
                model = cls(config)
            quantized_prefixes = {
                prefix
                for prefix in collect_qwen_quantized_prefixes(handle.keys())
                if prefix.startswith("model.language_model.")
            }
            _patch_quantized_linears(
                model,
                quantized_prefixes=quantized_prefixes,
                group_size=group_size,
                linear_dtype=torch_dtype,
            )
            model.config._attn_implementation = _resolve_attn_implementation(config_dict, None)
            model.model.language_model.config._attn_implementation = _resolve_attn_implementation(
                config_dict, "text_config"
            )
            model.model.vision_tower.config._attn_implementation = _resolve_attn_implementation(
                config_dict, "vision_config"
            )
            missing, unexpected = _restore_model_tensors(
                handle,
                model=model,
                torch_dtype=torch_dtype,
                output_device=device,
                pin_memory=pin_memory_enabled,
            )
        if missing or unexpected:
            raise RuntimeError(
                "Failed to restore Mistral3 multimodal runtime strictly enough. "
                f"missing={missing[:12]}, unexpected={unexpected[:12]}"
            )
        model.eval()
        return model


class _Mistral3PromptBackbone(nn.Module):
    def __init__(self, config: MistralConfig) -> None:
        super().__init__()
        self.language_model = MistralModel(config)

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value) -> None:
        self.language_model.set_input_embeddings(value)


class NunchakuMistral3PromptEncoderModel(PreTrainedModel):
    config_class = MistralConfig
    base_model_prefix = "model"

    def __init__(self, config: MistralConfig) -> None:
        super().__init__(config)
        self.model = _Mistral3PromptBackbone(config)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value) -> None:
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        image_sizes: torch.Tensor | None = None,
        **kwargs: tp.Any,
    ):
        if pixel_values is not None or image_sizes is not None:
            raise ValueError(
                "This Nunchaku Mistral3 prompt-only runtime does not support `pixel_values`/`image_sizes`. "
                "Please load a full multimodal export for caption upsampling."
            )
        return self.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        device: str | torch.device = "cpu",
        torch_dtype: torch.dtype | None = None,
        pin_memory: bool | str = "auto",
    ) -> "NunchakuMistral3PromptEncoderModel":
        ckpt = resolve_checkpoint_path(pretrained_model_name_or_path)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        device = device if isinstance(device, torch.device) else torch.device(device)
        pin_memory_enabled = resolve_pin_memory(pin_memory, device)

        with safetensors.safe_open(str(ckpt), framework="pt", device="cpu") as handle:
            raw_metadata = dict(handle.metadata() or {})
            runtime_contract = _load_runtime_contract(raw_metadata)
            model_type = str(raw_metadata.get("model_type", "")).strip().lower()
            if runtime_contract != MISTRAL3_PROMPT_EMBEDS_ONLY:
                raise ValueError(
                    "Unsupported Mistral3 prompt-only runtime contract. "
                    f"Expected `{MISTRAL3_PROMPT_EMBEDS_ONLY}`, got `{runtime_contract or '<missing>'}`."
                )
            if model_type != "mistral3_text":
                raise ValueError(f"Unsupported Mistral3 model_type metadata: `{model_type or '<missing>'}`.")
            text_config_dict = _load_text_config_dict(raw_metadata)
            group_size = _resolve_group_size(raw_metadata)
            text_config = MistralConfig(**text_config_dict)
            text_config._attn_implementation = _resolve_attn_implementation(
                {"text_config": text_config_dict},
                "text_config",
            )
            with init_empty_weights():
                model = cls(text_config)
            quantized_prefixes = {
                prefix
                for prefix in collect_qwen_quantized_prefixes(handle.keys())
                if prefix.startswith("model.language_model.")
            }
            _patch_quantized_linears(
                model,
                quantized_prefixes=quantized_prefixes,
                group_size=group_size,
                linear_dtype=torch_dtype,
            )
            model.config._attn_implementation = _resolve_attn_implementation({"text_config": text_config_dict}, "text_config")
            model.model.language_model.config._attn_implementation = model.config._attn_implementation
            missing, unexpected = _restore_model_tensors(
                handle,
                model=model,
                torch_dtype=torch_dtype,
                output_device=device,
                pin_memory=pin_memory_enabled,
            )
        if missing or unexpected:
            raise RuntimeError(
                "Failed to restore Mistral3 prompt-only runtime strictly enough. "
                f"missing={missing[:12]}, unexpected={unexpected[:12]}"
            )
        model.eval()
        return model


class NunchakuMistral3EncoderModel:
    """Stable entry point that dispatches to the correct Mistral3 runtime contract."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        ckpt = resolve_checkpoint_path(pretrained_model_name_or_path)
        with safetensors.safe_open(str(ckpt), framework="pt", device="cpu") as handle:
            raw_metadata = dict(handle.metadata() or {})
        runtime_contract = _load_runtime_contract(raw_metadata)
        runtime_class = str(raw_metadata.get("runtime_class", "")).strip()
        if runtime_contract == MISTRAL3_PROMPT_EMBEDS_ONLY or runtime_class == "NunchakuMistral3PromptEncoderModel":
            return NunchakuMistral3PromptEncoderModel.from_pretrained(ckpt, **kwargs)
        return NunchakuMistral3MultimodalEncoderModel.from_pretrained(ckpt, **kwargs)
