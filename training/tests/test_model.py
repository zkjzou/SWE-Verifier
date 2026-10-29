"""Verify native wrapper precision, freezing, and optimizer grouping on a tiny model."""

import torch
from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig, Qwen3_5VisionConfig

from training.model import OffloadedCheckpoint, configure_model, make_optimizer
from training.train import load_recipe


def test_native_model_setup_and_checkpoint_precision(tmp_path):
    config = Qwen3_5Config(
        text_config=Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            pad_token_id=0,
            eos_token_id=1,
        ),
        vision_config=Qwen3_5VisionConfig(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=32,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=1,
            num_position_embeddings=16,
        ),
    )
    model = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    configure_model(model)
    assert not model.config.use_cache and not model.config.text_config.use_cache
    for name, parameter in model.named_parameters():
        vision = name.startswith("model.visual.")
        assert parameter.requires_grad is not vision
        assert parameter.dtype == (torch.bfloat16 if vision else torch.float32)
    for layer in model.model.language_model.layers:
        assert isinstance(layer._gradient_checkpointing_func, OffloadedCheckpoint)
    optimizer = make_optimizer(model, load_recipe("joint_rubric"), fused=False)
    decay_ids = {id(p) for group in optimizer.param_groups if group["weight_decay"] for p in group["params"]}
    for name, parameter in model.named_parameters():
        if name.endswith("A_log"):
            assert id(parameter) in decay_ids
        if "norm" in name or not parameter.requires_grad:
            assert id(parameter) not in decay_ids
    # Preserve FP32-trained values through save/load, including values BF16 cannot represent.
    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(0.000123456)
    model.save_pretrained(tmp_path)
    restored = Qwen3_5ForConditionalGeneration.from_pretrained(tmp_path, dtype=torch.float32)
    configure_model(restored)
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, dict(model.named_parameters())[name], rtol=0, atol=0)
