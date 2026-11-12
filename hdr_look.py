"""HDR Look (SuperBeasts.AI): photographic local tone mapping for SDR images.

Produces the "HDR look" (recovered highlights, lifted shadows, strong local
contrast) on an ordinary display-referred image, entirely in float32:

1. sRGB -> linear light, Rec.709 luminance, log2 luminance (stops).
2. Edge-aware base/detail split of the log luminance with a self-guided filter
   (He et al., "Guided Image Filtering"), the base/detail structure of
   Durand & Dorsey, "Fast Bilateral Filtering for the Display of HDR Images".
3. The base layer goes through a filmic curve in stops around the image's
   median: a midtone slope (contrast) with soft roll-offs that pull large
   bright areas down and lift large dark areas. The detail layer is
   amplified, faded out near black so shadow noise is not boosted.
4. The new luminance is applied as a per-pixel ratio in linear light, which
   keeps chromaticity (hue and saturation) unchanged.
5. Overshoot above 1.0 is rolled off with a smooth shoulder that is the
   identity when nothing overshoots; remaining out-of-gamut pixels are pulled
   toward their own luminance instead of being clipped per channel.

With highlight_recovery = shadow_lift = local_contrast = 0 and contrast = saturation = 1
the node returns its input unchanged (up to float rounding).
"""
from __future__ import annotations

import torch

_REC709 = (0.2126, 0.7152, 0.0722)
_LOG_FLOOR = -14.0          # stops; luminance below 2^-14 is treated as black
_SHOULDER_KNEE = 0.75       # linear luminance where the highlight shoulder starts


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp(min=1e-12) ** (1.0 / 2.4) - 0.055)


def _luminance(rgb: torch.Tensor) -> torch.Tensor:
    return rgb[..., 0] * _REC709[0] + rgb[..., 1] * _REC709[1] + rgb[..., 2] * _REC709[2]


def _box_mean(x: torch.Tensor, r: int) -> torch.Tensor:
    """Mean over a (2r+1)^2 window, windows clipped at the image border. x: (H, W)."""
    h, w = x.shape
    dev = x.device
    out = x
    for dim, n in ((0, h), (1, w)):
        pad = [0, 0, 1, 0] if dim == 0 else [1, 0]
        c = torch.cumsum(torch.nn.functional.pad(out, pad), dim=dim)
        i = torch.arange(n, device=dev)
        hi = (i + r + 1).clamp(max=n)
        lo = (i - r).clamp(min=0)
        s = c.index_select(dim, hi) - c.index_select(dim, lo)
        cnt = (hi - lo).to(x.dtype)
        out = s / (cnt[:, None] if dim == 0 else cnt[None, :])
    return out


def _guided_self(x: torch.Tensor, r: int, eps: float) -> torch.Tensor:
    """Self-guided filter: edge-preserving smoothing of x. x: (H, W), centred for precision."""
    offset = x.mean()
    i = x - offset
    mean_i = _box_mean(i, r)
    var_i = (_box_mean(i * i, r) - mean_i * mean_i).clamp(min=0.0)
    a = var_i / (var_i + eps)
    b = mean_i - a * mean_i
    return _box_mean(a, r) * i + _box_mean(b, r) + offset


_HIGHLIGHT_RANGE_STOPS = 2.5  # base highlights above the anchor are limited to this at recovery 1
_SHADOW_RANGE_STOPS = 4.0     # base shadows below the anchor are limited to this at lift 1


def _soft_limit(rel: torch.Tensor, range_stops: float, amount: float) -> torch.Tensor:
    """A*tanh(rel/A) with A = range_stops/amount; the identity when amount == 0."""
    if amount <= 0.0:
        return rel
    a = range_stops / amount
    return a * torch.tanh(rel / a)


def _smoothstep(e0: float, e1: float, x: torch.Tensor) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def hdr_look_frame(
    rgb: torch.Tensor,
    strength: float = 1.0,
    highlight_recovery: float = 0.4,
    shadow_lift: float = 0.3,
    contrast: float = 1.15,
    local_contrast: float = 0.6,
    radius: float = 0.03,
    edge_preservation: float = 0.05,
    saturation: float = 1.1,
) -> torch.Tensor:
    """Apply the HDR look to one sRGB image (H, W, 3) in [0, 1]. Returns (H, W, 3) in [0, 1]."""
    src = rgb.clamp(0.0, 1.0)
    h, w = src.shape[:2]
    lin = _srgb_to_linear(src)
    y = _luminance(lin)
    y_floor = 2.0 ** _LOG_FLOOR
    log_y = torch.log2(y.clamp(min=y_floor))

    r = max(1, int(round(radius * min(h, w))))
    base = _guided_self(log_y, r, max(float(edge_preservation), 1e-6))
    detail = log_y - base

    # Anchor at the median of the base layer (a strided sample is enough and fast on 4K+).
    step = max(1, int((h * w / 1_000_000) ** 0.5))
    anchor = base[::step, ::step].median()

    # Filmic tone curve on the base layer, in stops around the anchor: midtone slope
    # `contrast`, with tanh roll-offs whose ceilings shrink as recovery/lift grow
    # (no roll-off at 0, i.e. a straight line through the anchor).
    rel = (base - anchor) * float(contrast)
    base_c = anchor + torch.where(
        rel > 0,
        _soft_limit(rel, _HIGHLIGHT_RANGE_STOPS, float(highlight_recovery)),
        _soft_limit(rel, _SHADOW_RANGE_STOPS, float(shadow_lift)),
    )

    gain = 1.0 + float(local_contrast) * _smoothstep(_LOG_FLOOR + 2.0, _LOG_FLOOR + 6.0, base)
    log_out = base_c + detail * gain

    ratio = torch.exp2(log_out) / y.clamp(min=y_floor)
    out = lin * ratio[..., None]

    # Highlight shoulder: extended-Reinhard above the knee with the white point at the
    # frame's peak luminance; exact identity when the peak does not exceed 1.0.
    y_out = _luminance(out)
    peak = float(y_out.max().clamp(min=1.0))
    if peak > 1.0:
        knee = _SHOULDER_KNEE
        x = ((y_out - knee) / (1.0 - knee)).clamp(min=0.0)
        wp = (peak - knee) / (1.0 - knee)
        x_c = x * (1.0 + x / (wp * wp)) / (1.0 + x)
        y_c = torch.where(y_out > knee, knee + (1.0 - knee) * x_c, y_out)
        out = out * (y_c / y_out.clamp(min=1e-12))[..., None]
        y_out = y_c

    if saturation != 1.0:
        out = y_out[..., None] + (out - y_out[..., None]) * float(saturation)

    # Gamut map toward the pixel's own luminance instead of clipping channels.
    yy = y_out[..., None].clamp(0.0, 1.0)
    cmax = out.max(dim=-1, keepdim=True).values
    cmin = out.min(dim=-1, keepdim=True).values
    t_hi = torch.where(cmax > 1.0, (cmax - 1.0) / (cmax - yy).clamp(min=1e-12), torch.zeros_like(cmax))
    t_lo = torch.where(cmin < 0.0, (0.0 - cmin) / (yy - cmin).clamp(min=1e-12), torch.zeros_like(cmin))
    t = torch.maximum(t_hi, t_lo).clamp(0.0, 1.0)
    out = out + (yy - out) * t

    result = _linear_to_srgb(out)
    s = float(strength)
    if s != 1.0:
        result = src * (1.0 - s) + result * s
    return result.clamp(0.0, 1.0)


def _processing_device(image: torch.Tensor) -> torch.device:
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        return image.device


class HDRLook:
    DESCRIPTION = (
        "Photographic HDR look for ordinary SDR images: edge-aware local tone mapping that "
        "recovers highlights, lifts shadows and boosts local contrast without halos, banding "
        "or hue shifts. Float32 throughout; neutral settings leave the image unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "Blend between the input (0) and the full effect (1)."}),
                "highlight_recovery": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                                                 "tooltip": "Compresses large bright areas toward mid-grey to reveal highlight detail."}),
                "shadow_lift": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Compresses large dark areas toward mid-grey to reveal shadow detail."}),
                "contrast": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 2.0, "step": 0.01,
                                       "tooltip": "Midtone contrast of the large-scale tones; 1.0 keeps the input's."}),
                "local_contrast": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.01,
                                             "tooltip": "Amplifies detail relative to its surroundings (the 'HDR' punch)."}),
                "radius": ("FLOAT", {"default": 0.03, "min": 0.002, "max": 0.25, "step": 0.001,
                                     "tooltip": "Size of the local region as a fraction of the image's short side."}),
                "edge_preservation": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 2.0, "step": 0.001,
                                                "tooltip": "Lower keeps edges sharper (fewer halos); higher behaves more like a plain blur."}),
                "saturation": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.01,
                                         "tooltip": "Saturation in linear light; 1.0 keeps the input's saturation."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "SuperBeastsAI/Image"

    def apply(self, image, strength, highlight_recovery, shadow_lift, contrast, local_contrast,
              radius, edge_preservation, saturation):
        device = _processing_device(image)
        frames = []
        with torch.no_grad():
            for i in range(image.shape[0]):
                frame = image[i, ..., :3].to(device=device, dtype=torch.float32)
                out = hdr_look_frame(
                    frame, strength=strength, highlight_recovery=highlight_recovery,
                    shadow_lift=shadow_lift, contrast=contrast, local_contrast=local_contrast, radius=radius,
                    edge_preservation=edge_preservation, saturation=saturation,
                )
                frames.append(out.to(device=image.device, dtype=image.dtype))
                del frame, out
        return (torch.stack(frames, dim=0),)


NODE_CLASS_MAPPINGS = {"HDR Look (SuperBeasts.AI)": HDRLook}
NODE_DISPLAY_NAME_MAPPINGS = {"HDRLook": "HDR Look (SuperBeasts.AI)"}
