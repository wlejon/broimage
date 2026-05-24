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
- **Geometric** — resize (nearest / bilinear / bicubic / lanczos), crop,
  letterbox / pad, flip, rotate.
- **Color** — RGBA ↔ RGB ↔ gray, HWC ↔ CHW, gamma, sRGB ↔ linear.
- **Normalize** — per-channel `(x - mean) / std`, with CLIP / ImageNet presets.
- **Kernel verbs** — `reduce`, `map`, `combine`, `lookup`, `stencil`,
  `resample` over flat typed buffers. The general image-math surface that
  bro.image exposes to JS.

GPU image ops live in brotensor (one place for CUDA / Metal kernels); broimage
calls into them when handed a GPU `brotensor::Tensor`.

## Status

Skeleton. Nothing implemented yet beyond `version_string()`. See
[bro/docs/multi-repo-workflow.md](https://github.com/wlejon/bro/blob/main/docs/multi-repo-workflow.md)
for how this slots into the multi-repo dev loop.

## Build

```
cmake -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug
```
