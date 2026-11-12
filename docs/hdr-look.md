## HDR Look (SuperBeasts.AI)

A photographic "HDR look" for ordinary SDR images: highlight detail is recovered, shadows are opened up, midtones keep (or gain) punch and local detail is amplified, without halos, banding, clipping or hue shifts. It is a separate node from **HDR Effects**, which is unchanged.

### How it works

1. The image is converted from sRGB to linear light; luminance is taken in log2 (stops).
2. A self-guided edge-preserving filter splits log luminance into a large-scale **base** layer and a fine **detail** layer, so strong edges do not produce halos.
3. The base layer goes through a filmic curve in stops around the image's median: a midtone slope (`contrast`) with soft roll-offs that pull large bright areas down (`highlight_recovery`) and lift large dark areas (`shadow_lift`).
4. The detail layer is multiplied by `1 + local_contrast`; the boost fades out near black so sensor-style shadow noise is not amplified.
5. The new luminance is applied as a ratio to linear RGB, which keeps every pixel's hue and saturation. `saturation` then scales chroma in linear light.
6. Anything brighter than white is rolled off with a smooth shoulder (identity when nothing exceeds white), and out-of-gamut colours are pulled toward their own luminance instead of being clipped per channel.

All processing is float32 on the ComfyUI torch device. With `highlight_recovery = shadow_lift = local_contrast = 0` and `contrast = saturation = 1` the output equals the input.

### Parameters

| parameter | default | range | effect |
|---|---|---|---|
| `strength` | 1.0 | 0–1 | blend between input and full effect |
| `highlight_recovery` | 0.4 | 0–1 | compress large bright areas toward mid-grey |
| `shadow_lift` | 0.3 | 0–1 | compress large dark areas toward mid-grey |
| `contrast` | 1.15 | 0.5–2 | midtone contrast of the large-scale tones |
| `local_contrast` | 0.6 | 0–3 | amplification of local detail |
| `radius` | 0.03 | 0.002–0.25 | size of the "local" region, as a fraction of the short image side |
| `edge_preservation` | 0.05 | 0.001–2 | lower keeps edges sharper (fewer halos), higher acts more like a blur |
| `saturation` | 1.1 | 0–2 | chroma scale in linear light |

Starting points: subtle `0.3 / 0.2 / 1.1 / 0.4`, punchy (defaults), strong `0.7 / 0.5 / 1.3 / 1.0` (highlight_recovery / shadow_lift / contrast / local_contrast).

### Tests

`python3 tests/test_hdr_look.py` (requires only torch).
