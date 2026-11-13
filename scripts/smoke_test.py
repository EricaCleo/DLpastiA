"""Quick CPU smoke test of ALL six methods with tiny models, before using the cluster GPU. Takes a few minutes.
    python scripts/smoke_test.py            # LPIPS off (no weight download needed)
    python scripts/smoke_test.py --lpips    # also test LPIPS (needs the AlexNet/LPIPS weights, cached after first use)
Needs the dataset (data/raw/afhq) for the evaluation parts. Writes nothing into outputs/.
"""
import argparse
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader, TensorDataset

ap = argparse.ArgumentParser()
ap.add_argument("--lpips", action="store_true")
a = ap.parse_args()
S, TMP = 64, Path(tempfile.mkdtemp())
results = []


def check(name):
    def deco(fn):
        try:
            fn()
            results.append((name, "PASS", ""))
        except Exception as e:
            traceback.print_exc()
            results.append((name, "FAIL", f"{type(e).__name__}: {str(e)[:90]}"))
        return fn
    return deco


def evaluate_tiny(method, mask="one_sided"):
    from src.data import load_config
    from src.eval import evaluate
    cfg = load_config(); cfg["image_size"] = S
    r = evaluate(method, mask, n_images=4, batch_size=2, K=2, device="cpu", use_lpips=a.lpips, cfg=cfg)
    assert all(v == v for v in r.values()), f"NaN in {r}"
    return r


def rand_loader(n=16, bs=8):
    return DataLoader(TensorDataset(torch.rand(n, 3, S, S) * 2 - 1), batch_size=bs, drop_last=True)


@check("data: dataset, fixed test masks")
def _():
    from src.data import get_datasets
    tr, te = get_datasets()
    x, m, xm = te[0]
    assert len(tr) > 1000 and len(te) > 100 and torch.equal(m, te[0][1])


@check("baselines (grey fill, mirror paste)")
def _():
    from src.methods import GreyFill, MirrorPaste
    for M in (GreyFill, MirrorPaste):
        evaluate_tiny(M())


@check("method 5 RePaint: train tiny DDPM + sample + evaluate")
def _():
    from src.diffusion import EpsUNet, load_eps_model
    from src.diffusion.train import train_ddpm
    net = EpsUNet(image_size=S, channels=(32, 64), attn_levels=(1,), layers=1)
    train_ddpm(net, rand_loader(), TMP / "ddpm.pt", max_steps=4, save_every=4, device="cpu", log_every=100)
    from src.methods.repaint import RePaint
    evaluate_tiny(RePaint(load_eps_model(TMP / "ddpm.pt"), steps=5, jump_length=2, resamples=2, device="cpu"))


@check("method 6 mirror-guided RePaint (fixed + adaptive)")
def _():
    from src.diffusion import load_eps_model
    from src.methods.mirror_diffusion import MirrorGuidedRePaint
    for adaptive in (False, True):
        evaluate_tiny(MirrorGuidedRePaint(load_eps_model(TMP / "ddpm.pt"), lam=0.1, adaptive=adaptive,
                                          steps=5, jump_length=2, resamples=2, device="cpu"))


@check("method 5b U-ViT backbone: train tiny + RePaint + mirror-guided evaluate")
def _():
    from src.diffusion import EpsUViT, load_eps_model
    from src.diffusion.train import train_ddpm
    from src.methods.mirror_diffusion import MirrorGuidedRePaint
    from src.methods.repaint import RePaint
    train_ddpm(EpsUViT(image_size=S, patch=8, dim=32, depth=3, heads=4), rand_loader(), TMP / "uvit.pt", max_steps=4, save_every=4, device="cpu", log_every=100)
    net = load_eps_model(TMP / "uvit.pt")
    assert type(net).__name__ == "EpsUViT"
    evaluate_tiny(RePaint(net, steps=5, jump_length=2, resamples=2, device="cpu"))
    evaluate_tiny(MirrorGuidedRePaint(net, lam=0.1, adaptive=True, steps=5, jump_length=2, resamples=2, device="cpu"))


@check("method 1 MAE: tiny model train + resume + evaluate")
def _():
    from src.methods.mae import MAEInpainter, build_mae, load_mae
    from src.methods.mae_train import train_mae
    cfg = dict(image_size=S, hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64,
               decoder_hidden_size=32, decoder_num_hidden_layers=1, decoder_num_attention_heads=4,
               decoder_intermediate_size=64, norm_pix_loss=False)
    train_mae(build_mae(config=cfg), rand_loader(), TMP / "mae.pt", epochs=1, lr=1e-3, device="cpu", log_every=100)
    train_mae(build_mae(config=cfg), rand_loader(), TMP / "mae.pt", epochs=2, lr=1e-3, device="cpu", log_every=100)  # resumes
    evaluate_tiny(MAEInpainter(load_mae(TMP / "mae.pt"), device="cpu"))


def gan_cfg(mode):
    return dict(mode=mode, size=S, z_dim=32, w_dim=32, base=1024, cap=32, map_layers=2)


def gan_loader():
    from src.data import get_datasets
    tr, _ = get_datasets()
    tr.size = S
    return DataLoader(tr, batch_size=8, shuffle=True, drop_last=True)


@check("methods 3/4 GAN train (comod, mirror 04a, mirror 04b with symmetry) + resume")
def _():
    from src.gan import train_gan
    for mode, sym, name in [("comod", 0.0, "comod"), ("mirror", 0.0, "m04a"), ("mirror", 0.1, "m04b")]:
        ck = TMP / f"{name}.pt"
        train_gan(gan_cfg(mode), gan_loader(), ck, device="cpu", lpips_w=1.0 if a.lpips else 0.0, sym_w=sym,
                  max_steps=3, save_every=3, sample_every=100, log_every=100, r1_every=2)
        train_gan(gan_cfg(mode), gan_loader(), ck, device="cpu", lpips_w=1.0 if a.lpips else 0.0, sym_w=sym,
                  max_steps=5, save_every=5, sample_every=100, log_every=100, r1_every=2)           # resumes at step 3
        assert torch.load(ck)["step"] == 5


@check("methods 3/4 evaluate (comod, mirror 04a/04b)")
def _():
    from src.methods.mirror_conditioned import MirrorConditionedInpainter
    from src.methods.stylegan_noninversion import CoModGANInpainter
    evaluate_tiny(CoModGANInpainter.from_ckpt(TMP / "comod.pt", "cpu"))
    for n in ("m04a", "m04b"):
        evaluate_tiny(MirrorConditionedInpainter.from_ckpt(TMP / f"{n}.pt", "cpu"), "one_sided")


@check("method 2 StyleGAN inversion: train unconditional G + invert + evaluate")
def _():
    from src.gan import train_gan
    from src.methods.stylegan_inversion import StyleGANInversion
    ck = TMP / "uncond.pt"
    train_gan(gan_cfg("uncond"), gan_loader(), ck, device="cpu", max_steps=3, save_every=3, sample_every=100, log_every=100, r1_every=2)
    evaluate_tiny(StyleGANInversion.from_ckpt(ck, "cpu", steps=3, use_lpips=a.lpips))


@check("per-method scripts: command-line interfaces parse")
def _():
    for f in ["m1_mae", "m2_stylegan_inversion", "m3_stylegan_noninversion", "m4_mirror_conditioned", "m5_repaint", "m6_mirror_guided_repaint"]:
        for sub in ([], ["train"], ["eval"]) if f != "m6_mirror_guided_repaint" else ([], ["eval"]):
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / f"{f}.py"), *sub, "--help"], capture_output=True, text=True)
            assert r.returncode == 0, f"{f} {' '.join(sub)}: {r.stderr[-200:]}"


print("\n" + "=" * 78)
for name, status, msg in results:
    print(f"{status:5} {name}" + (f"   <- {msg}" if msg else ""))
failed = [r for r in results if r[1] == "FAIL"]
print("=" * 78 + f"\n{len(results) - len(failed)}/{len(results)} passed")
sys.exit(1 if failed else 0)
