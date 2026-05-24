# broimage

Image processing for the bro stack — decode/encode, geometric and color ops,
normalization, and composable typed-buffer kernels. Pure C++20, built on
[bromath](https://github.com/wlejon/bromath) and
[brotensor](https://github.com/wlejon/brotensor) (for GPU preprocessing when
a backend is enabled). CPU-by-default; GPU paths forward to brotensor's
CUDA / Metal backends.

broimage is the single home for image work that used to be duplicated across
the bro stack: bro's HTML `Image` decode and `bro.image` JS kernels, brolm's
CLIP/SAM/VLM host-side resize + normalize, brodiffusion's pixel preprocessing,
and the `image_preproc` ops that previously lived in brotensor.

## Scope

- **Decode / encode** — stb_image-backed RGBA load and PNG/JPEG write.
  EXIF orientation honored on demand via `decode_file_oriented` (phone JPEGs
  load upright instead of sideways).
- **Geometric** — resize (nearest / bilinear / bicubic / lanczos3 / area),
  crop, letterbox / pad, flip, rotate.
- **Alpha** — premultiply / unpremultiply and alpha-correct resize +
  letterbox for RGBA composites (no edge fringing on transparent inputs).
- **Color** — RGBA ↔ RGB ↔ gray, HWC ↔ CHW, gamma, sRGB ↔ linear,
  RGB ↔ HSV / HSL, 3x3 / 3x4 color-matrix apply.
- **Normalize** — per-channel `(x - mean) / std`, with CLIP / ImageNet / SAM
  presets.
- **Preproc** — `u8 NHWC → f32 NCHW` scale+bias shuffle, plus the plain
  NHWC ↔ NCHW float reshuffles. Host-side mirror of brotensor's
  `image_preproc` ops.
- **Kernel verbs** — `reduce`, `map`, `combine`, `lookup`, `stencil`,
  `resample`, `gradient` over flat typed buffers. The general image-math
  surface that bro.image exposes to JS.
- **Tensor adapter** — forwarders to brotensor's `image_normalize` and
  `image_u8_to_f32_nhwc_to_nchw` so callers reach for broimage even when
  the destination is a `brotensor::Tensor`.

GPU image ops live in brotensor (one place for CUDA / Metal kernels); broimage
calls into them when handed a GPU `brotensor::Tensor`.

See [bro/docs/multi-repo-workflow.md](https://github.com/wlejon/bro/blob/main/docs/multi-repo-workflow.md)
for how this slots into the multi-repo dev loop.

## Build

```
cmake -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug
```
