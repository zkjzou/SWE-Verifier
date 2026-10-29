"""Dense Qwen3.5 training setup used by the four-H200 recipes."""

from __future__ import annotations

from functools import partial


class OffloadedCheckpoint:
    def __init__(self, original):
        self.original = original

    def __call__(self, function, *args, **kwargs):
        import torch

        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            return self.original(function, *args, **kwargs)


def configure_model(model):
    import torch

    if type(model).__name__ != "Qwen3_5ForConditionalGeneration":
        raise ValueError("Expected the native Qwen3.5 multimodal wrapper")
    model.config.use_cache = model.config.text_config.use_cache = False
    frozen = 0
    for name, param in model.named_parameters():
        vision = name.startswith("model.visual.")
        param.requires_grad_(not vision)
        param.data = param.data.to(torch.bfloat16 if vision else torch.float32)
        frozen += int(vision)
    if not frozen:
        raise ValueError("Vision/projector freeze matched no parameters")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    layers = [m for m in model.modules() if type(m).__name__ == "Qwen3_5DecoderLayer"]
    if len(layers) != model.config.text_config.num_hidden_layers:
        raise ValueError("Unexpected decoder layer inventory")
    for layer in layers:
        original = layer._gradient_checkpointing_func
        if not isinstance(original, partial) or original.keywords != {"use_reentrant": False}:
            raise ValueError("Expected nonreentrant gradient checkpointing")
        layer._gradient_checkpointing_func = OffloadedCheckpoint(original)


def release_host_cache():
    import torch

    torch.cuda.synchronize()
    release = getattr(torch._C, "_host_emptyCache", None)
    if release is None:
        raise RuntimeError("Pinned torch runtime lacks the checkpoint-offload host-cache release API")
    release()


def load_model(source, *, trained_checkpoint=False):
    import torch
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3_5
    from transformers import AutoModelForImageTextToText

    apply_liger_kernel_to_qwen3_5(
        rope=False, cross_entropy=False, fused_linear_cross_entropy=True, rms_norm=True, swiglu=False
    )
    model, info = AutoModelForImageTextToText.from_pretrained(
        source,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32 if trained_checkpoint else torch.bfloat16,
        attn_implementation="flash_attention_2",
        output_loading_info=True,
    )
    if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("Incomplete model checkpoint load")
    if any(not name.startswith("mtp.") for name in info.get("unexpected_keys", [])):
        raise ValueError("Unexpected checkpoint weights")
    configure_model(model)
    return model


def make_optimizer(model, config, *, fused=True):
    import torch
    from transformers import Trainer

    decay_names = set(Trainer.get_decay_parameter_names(None, model))
    groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and (n in decay_names) == decay],
            "weight_decay": config["weight_decay"] if decay else 0.0,
        }
        for decay in (True, False)
    ]
    return torch.optim.AdamW(
        groups, lr=config["learning_rate"], betas=tuple(config["adam_betas"]), eps=config["adam_epsilon"], fused=fused
    )
