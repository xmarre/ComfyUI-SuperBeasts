"""Tests for HDR Look (SuperBeasts.AI). Run: python3 tests/test_hdr_look.py (needs torch only)."""
import importlib.util, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("hdr_look", os.path.join(HERE, "..", "hdr_look.py"))
hl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hl)


def scene(h=256, w=320, seed=0):
    """Synthetic sRGB frame: dark-to-bright gradient, a bright window, coloured patches, texture."""
    g = torch.Generator().manual_seed(seed)
    y = torch.linspace(0, 1, h)[:, None].expand(h, w)
    x = torch.linspace(0, 1, w)[None, :].expand(h, w)
    lum = 0.02 + 0.6 * x + 0.05 * torch.rand(h, w, generator=g)
    lum[40:120, 200:300] = 0.98                      # near-clipped window
    img = torch.stack([lum, lum * 0.9, lum * 0.8], -1)
    img[150:220, 20:90] = torch.tensor([0.7, 0.15, 0.1])   # saturated red patch
    img[150:220, 110:180] = torch.tensor([0.05, 0.3, 0.1]) # dark green patch
    return img.clamp(0, 1) * (0.9 + 0.1 * y[..., None])


def run():
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    img = scene()
    neutral = hl.hdr_look_frame(img, strength=1.0, highlight_recovery=0.0, shadow_lift=0.0,
                                contrast=1.0, local_contrast=0.0, saturation=1.0)
    err = (neutral - img).abs().max().item()
    check("neutral settings are the identity", err < 2e-4, f"max err {err:.2e}")

    zero = hl.hdr_look_frame(img, strength=0.0)
    check("strength 0 returns the input", torch.equal(zero, img))

    out = hl.hdr_look_frame(img)
    check("output finite and in [0,1]", bool(torch.isfinite(out).all()) and out.min() >= 0 and out.max() <= 1)
    check("shape preserved", out.shape == img.shape)

    neutral_kw = dict(shadow_lift=0.0, contrast=1.0, local_contrast=0.0, saturation=1.0)
    rec0 = hl.hdr_look_frame(img, highlight_recovery=0.0, **neutral_kw)
    rec1 = hl.hdr_look_frame(img, highlight_recovery=0.8, **neutral_kw)
    w0 = rec0[40:120, 200:300].mean().item()
    w1 = rec1[40:120, 200:300].mean().item()
    check("highlight_recovery pulls a bright area down", w1 < w0 - 0.01, f"{w0:.3f} -> {w1:.3f}")
    flat = (out.max(-1).values >= 0.999).float().mean().item()
    check("no flat clipping at defaults", flat < 1e-3, f"{100 * flat:.3f}% of pixels at 1.0")
    win_std_in = img[40:120, 200:300, 1].std().item()
    win_std_out = out[40:120, 200:300, 1].std().item()
    check("bright window keeps its texture", win_std_out >= win_std_in, f"std {win_std_in:.4f} -> {win_std_out:.4f}")

    # Hue preservation: with saturation 1, the red patch keeps its linear-light chromaticity.
    lin_in = hl._srgb_to_linear(img[160:210, 30:80]).reshape(-1, 3)
    o1 = hl.hdr_look_frame(img, saturation=1.0)
    lin_out = hl._srgb_to_linear(o1[160:210, 30:80]).reshape(-1, 3)
    c_in = lin_in / lin_in.sum(-1, keepdim=True)
    c_out = lin_out / lin_out.sum(-1, keepdim=True)
    d = (c_in - c_out).abs().max().item()
    check("chromaticity preserved (saturation 1)", d < 1e-3, f"max chroma diff {d:.2e}")

    more = hl.hdr_look_frame(img, local_contrast=1.5)
    less = hl.hdr_look_frame(img, local_contrast=0.0)
    hp = lambda t: (t[1:, :, 1] - t[:-1, :, 1]).abs().mean().item()
    check("local_contrast increases fine detail", hp(more) > hp(less), f"{hp(less):.4f} -> {hp(more):.4f}")

    batch = torch.stack([img, scene(seed=1)], 0)
    node_out = hl.HDRLook().apply(batch, 1.0, 0.4, 0.3, 1.15, 0.6, 0.03, 0.05, 1.1)[0]
    same = (node_out[0] - hl.hdr_look_frame(img)).abs().max().item()
    check("node batch output matches per-frame", node_out.shape == batch.shape and same < 1e-6, f"diff {same:.2e}")

    rgba = torch.cat([batch, torch.ones_like(batch[..., :1])], -1)
    check("RGBA input accepted (RGB output)", hl.HDRLook().apply(rgba, 1.0, 0.4, 0.3, 1.15, 0.6, 0.03, 0.05, 1.1)[0].shape == batch.shape)

    black = torch.zeros(64, 64, 3)
    check("black frame stays black", hl.hdr_look_frame(black).abs().max().item() < 1e-6)
    white = torch.ones(64, 64, 3)
    wo = hl.hdr_look_frame(white)
    check("white frame stays finite and white-ish", bool(torch.isfinite(wo).all()) and wo.min().item() > 0.95, f"min {wo.min().item():.3f}")

    print("\nALL PASSED" if not fails else "\nFAILED: " + ", ".join(fails))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
