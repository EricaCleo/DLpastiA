"""Method 1: Masked Autoencoder (self-supervised, ViT backbone). Deterministic, fast.

Training (self-supervised, scripts/train_mae.py): start from public pretrained ViT-MAE weights, fine-tune on clean
AFHQ images with ~75% random patch masking; loss = MSE on the hidden patches. The model never sees our square holes.

Inference (filling a hole): every 16x16 patch that overlaps the hole is treated as hidden, the encoder sees only the
remaining patches, the decoder predicts pixels for the hidden ones; the evaluation then composites the prediction
into the hole (visible pixels are always copied from the original).

Built on Hugging Face `transformers` ViTMAEForPreTraining. Needs internet only to fetch the pretrained weights once,
on the login node (compute nodes are offline): see scripts/download_pretrained.py.
"""
import torch
import torch.nn.functional as F

from .base import Inpainter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _sincos(grid, dim):
    """Fixed 2-D sin-cos table (with cls slot), same formula/layout as transformers' own initialiser,
    which the pretrained MAE weights rely on (h/w halves swapped)."""
    from transformers.models.vit_mae.modeling_vit_mae import build_2d_sinusoidal_position_embedding
    pe = build_2d_sinusoidal_position_embedding(height=grid, width=grid, embed_dim=dim, cls_token=True)
    half = pe.shape[-1] // 2
    return torch.cat([pe[..., half:], pe[..., :half]], dim=-1).unsqueeze(0)


def set_position_embeddings(model):
    """Write correct fixed position tables for model.config.image_size (encoder and decoder)."""
    grid = model.config.image_size // model.config.patch_size
    with torch.no_grad():
        pe, dpe = model.vit.embeddings.position_embeddings, model.decoder.decoder_pos_embed
        pe.copy_(_sincos(grid, pe.shape[-1])); dpe.copy_(_sincos(grid, dpe.shape[-1]))


def build_mae(image_size=128, pretrained="facebook/vit-mae-base", from_scratch=False, config=None):
    """Returns a ViTMAEForPreTraining for `image_size` (multiple of 16). `config` = dict from a saved checkpoint."""
    from transformers import ViTMAEConfig, ViTMAEForPreTraining
    if config is not None:
        return ViTMAEForPreTraining(ViTMAEConfig(**config))
    if from_scratch:
        model = ViTMAEForPreTraining(ViTMAEConfig(image_size=image_size, norm_pix_loss=False))
    else:
        model = ViTMAEForPreTraining.from_pretrained(
            pretrained, image_size=image_size, norm_pix_loss=False, ignore_mismatched_sizes=True)
    set_position_embeddings(model)  # tables depend on the grid size (224 -> 128 changes it); all other weights stay pretrained
    return model


def to_mae_input(x):
    """[-1,1] images -> ImageNet-normalised input expected by the pretrained model."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (((x + 1) / 2) - mean) / std


def from_mae_output(y):
    mean = torch.tensor(IMAGENET_MEAN, device=y.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=y.device).view(1, 3, 1, 1)
    return ((y * std + mean) * 2 - 1).clamp(-1, 1)


def save_mae(model, path, extra=None):
    torch.save({"model": model.state_dict(), "config": model.config.to_dict(), **(extra or {})}, path)


def load_mae(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = build_mae(config=ck["config"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


class MAEInpainter(Inpainter):
    name = "mae"
    stochastic = False

    def __init__(self, model, device="cuda"):
        self.model = model.to(device).eval()
        self.device = device
        self.patch = model.config.patch_size

    @torch.no_grad()
    def fill(self, masked, mask, K=1):
        masked, mask = masked.to(self.device), mask.to(self.device)
        B, _, H, W = masked.shape
        p = self.patch
        hidden = (F.max_pool2d(mask, p, p) > 0).flatten(1).float()   # (B, L): 1 = patch touches the hole
        L = hidden.shape[1]
        out = []
        for b in range(B):                                            # per image: the number of hidden patches differs
            n_hidden = int(hidden[b].sum().item())
            if n_hidden == 0:
                out.append(masked[b:b + 1]); continue
            n_hidden = min(n_hidden, L - 1)                           # encoder needs at least one visible patch
            # the library keeps int(L * (1 - mask_ratio)) patches: choose the ratio so exactly L - n_hidden are kept
            self.model.config.mask_ratio = (n_hidden - 0.5) / L
            noise = hidden[b:b + 1] + 0.01 * torch.arange(L, device=self.device).float().view(1, L) / L  # visible first, hidden last; fixed order
            o = self.model(pixel_values=to_mae_input(masked[b:b + 1]), noise=noise)
            out.append(from_mae_output(self.model.unpatchify(o.logits)))
        return [torch.cat(out, 0)]
