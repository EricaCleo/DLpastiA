"""Generator inputs for the supervised inpainters (masked image, mask in [-1,1] / {0,1}; mask = 1 in the hole)."""
import torch


def comod_input(masked, mask):
    """Method 3: masked image (3) + mask (1) = 4 channels."""
    return torch.cat([masked, mask], 1)


def mirror_input(masked, mask):
    """Method 4: [masked, mask, flip(masked), flip(mask)] = 8 channels (the mirrored copy sits at the mirrored location)."""
    return torch.cat([masked, mask, torch.flip(masked, [3]), torch.flip(mask, [3])], 1)


INPUTS = {"comod": (comod_input, 4), "mirror": (mirror_input, 8)}
