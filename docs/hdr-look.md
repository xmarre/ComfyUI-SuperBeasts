## HDR Look (SuperBeasts.AI)

A photographic "HDR look" for ordinary SDR images: highlight detail is recovered, shadows are opened up, midtones keep (or gain) punch and local detail is amplified, without halos, banding, clipping or hue shifts. It is a separate node from **HDR Effects**, which is unchanged.

### How it works

1. The image is converted from sRGB to linear light; luminance is taken in log2 (stops).
2. A self-guided edge-preserving filter splits log luminance into a large-scale **base** layer and a fine **detail** layer, so strong edges do not produce halos.
3. The base layer goes through a filmic curve in stops: a midtone slope (`contrast`) pivoting on display mid-grey (0.18), a soft roll-off of tones within about 1.5 stops of white (`highlight_recovery`), and a lift of large areas below the image's median (`shadow_lift`). Using absolute levels for contrast and highlights means a low-key image's subject (for example a face against a dark background) is not treated as a highlight and pulled down. `exposure` then shifts the result; any brightening saturates against the headroom left below white for each pixel's brightest channel, so bright and warm areas keep room for their texture.
4. The detail layer is multiplied by `1 + local_contrast` and added back; the boost fades out near black so sensor-style shadow noise is not amplified.
5. The new luminance is applied as a ratio to linear RGB, which keeps every pixel's hue and saturation.
6. A smooth shoulder on each pixel's brightest channel rolls off anything above white (identity when nothing exceeds white), so no channel clips.
7. `black_depth` applies a toe below a quarter of the frame's median luminance, so blacks get deeper relative to the scene without crushing its midtones.
8. `saturation` scales chroma in linear light and `vibrance` adds extra chroma to muted colours; both boosts fade out toward white so bright colours are not pushed out of gamut. Remaining out-of-gamut colours are pulled toward their own luminance instead of being clipped per channel.

All processing is float32 on the ComfyUI torch device. With `highlight_recovery = shadow_lift = local_contrast = exposure = black_depth = vibrance = 0` and `contrast = saturation = 1` the output equals the input.

### Parameters

| parameter | default | range | effect |
|---|---|---|---|
| `strength` | 1.0 | 0–1 | blend between input and full effect |
| `highlight_recovery` | 0.5 | 0–1 | roll off large areas close to white to reveal their detail |
| `shadow_lift` | 0.25 | 0–1 | compress large dark areas toward mid-grey |
| `contrast` | 1.3 | 0.5–2 | midtone contrast of the large-scale tones |
| `local_contrast` | 0.9 | 0–3 | amplification of local detail |
| `radius` | 0.03 | 0.002–0.25 | size of the "local" region, as a fraction of the short image side |
| `edge_preservation` | 0.05 | 0.001–2 | lower keeps edges sharper (fewer halos), higher acts more like a blur |
| `saturation` | 1.15 | 0–2 | chroma scale in linear light |
| `exposure` | 0.15 | −2–2 | overall brightness in stops |
| `black_depth` | 0.75 | 0–3 | deeper, richer blacks relative to the scene |
| `vibrance` | 0.35 | −1–2 | extra saturation for muted colours, protecting already-saturated ones |

Starting points (highlight_recovery / contrast / local_contrast / saturation / exposure / black_depth / vibrance):
- subtle: `0.3 / 1.1 / 0.4 / 1.05 / 0 / 0.3 / 0.15`
- punchy (defaults): `0.5 / 1.3 / 0.9 / 1.15 / 0.15 / 0.75 / 0.35`
- strong: `0.6 / 1.45 / 1.3 / 1.25 / 0.3 / 1.5 / 0.6`

### Tests

`python3 tests/test_hdr_look.py` (requires only torch).
