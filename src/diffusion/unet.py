"""Noise-prediction network (U-Net baseline) built on diffusers' UNet2DModel."""
import torch
import torch.nn as nn


class EpsUNet(nn.Module):
    """forward(x, t) -> predicted noise, same shape as x. t = original DDPM timesteps (long)."""

    def __init__(self, image_size=128, channels=(128, 256, 256, 512), attn_levels=(2,), layers=2, grad_checkpoint=False):
        super().__init__()
        from diffusers import UNet2DModel
        n = len(channels)
        down = tuple("AttnDownBlock2D" if i in attn_levels else "DownBlock2D" for i in range(n))
        up = tuple("AttnUpBlock2D" if (n - 1 - i) in attn_levels else "UpBlock2D" for i in range(n))
        self.cfg = dict(image_size=image_size, channels=tuple(channels), attn_levels=tuple(attn_levels), layers=layers)  # (grad_checkpoint is not saved: it only affects memory)
        self.net = UNet2DModel(sample_size=image_size, in_channels=3, out_channels=3, layers_per_block=layers,
                               block_out_channels=tuple(channels), down_block_types=down, up_block_types=up)
        if grad_checkpoint:
            self.net.enable_gradient_checkpointing()  # recompute activations in backward: less memory, ~30% slower

    def forward(self, x, t):
        return self.net(x, t).sample


def build_eps_model(cfg):
    """cfg saved in a checkpoint -> model. Checkpoints without an 'arch' entry are U-Nets (older runs)."""
    cfg = dict(cfg)
    if cfg.pop("arch", "unet") == "uvit":
        from .uvit import EpsUViT
        cfg.pop("arch", None)
        return EpsUViT(**cfg)
    return EpsUNet(**cfg)


def load_eps_model(ckpt_path, device="cpu", use_ema=True):
    ck = torch.load(ckpt_path, map_location=device)
    m = build_eps_model(ck["cfg"]).to(device)
    m.load_state_dict(ck["ema"] if use_ema and "ema" in ck else ck["model"])
    return m.eval()
