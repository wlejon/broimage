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

- **Decode / encode** — stb_image-backed RGBA load (PNG, JPEG, BMP, TGA, GIF,
  PSD, PIC, PNM) and PNG/JPEG write, from files or in-memory buffers.
  `probe_dimensions_memory` reads width/height/channels from a header without
  decoding pixels. EXIF orientation honored on demand via `decode_file_oriented`
  (phone JPEGs load upright instead of sideways), with `read_exif_orientation`
  and `apply_exif_orientation` exposed for callers holding their own buffers.
  16-bit (`decode_file_u16`) and HDR / float (`decode_file_f32`) paths for depth
  maps and Radiance sources.
- **Geometric** — resize (nearest / bilinear / bicubic / lanczos3 / area) over
  HWC / CHW float32 and HWC u8, crop, center-crop, letterbox, constant-pad,
  flip (horizontal / vertical), rotate (90-degree turns). The u8 ops take row
  strides so they operate on a sub-rect of a larger buffer without a copy.
- **Alpha** — premultiply / unpremultiply and alpha-correct resize +
  letterbox for RGBA composites (no edge fringing on transparent inputs).
- **Color** — RGBA ↔ RGB ↔ gray, HWC ↔ CHW, gamma, sRGB ↔ linear,
  RGB ↔ HSV / HSL, 3x3 / 3x4 color-matrix apply.
- **Normalize** — per-channel `(x - mean) / std`, with CLIP / ImageNet / SAM
  presets.
- **Preproc** — `u8 NHWC → f32 NCHW` scale+bias shuffle and its inverse
  `f32 NCHW → u8 NHWC` (for decode-side visualization), plus the plain
  NHWC ↔ NCHW float reshuffles. Host-side mirror of brotensor's
  `image_preproc` ops.
- **Kernel verbs** — `reduce` (minmax / sum / mean / histogram), `map`
  (affine / abs / log / sqrt / exp / pow), `combine` (add / sub / mul / min /
  max / lerp / weighted-sum), `lookup`, `stencil` (single-channel and
  multi-channel HWC convolution), `resample`, `gradient` over flat typed
  buffers. The general image-math surface that bro.image exposes to JS.
- **Tiling** — feather-window generation, weighted tile accumulation, and final
  accumulator normalize: the building blocks for splitting a large image into
  overlapping tiles, running a local operator per tile, and gluing the outputs
  back into one seamless full-res map.
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

broimage ships no GPU language of its own. The `BROTENSOR_WITH_CUDA` /
`BROTENSOR_WITH_METAL` options only forward the backend choice so a standalone
GPU build resolves brotensor's CUDA / Metal backend (the GPU image kernels live
there). Built inside the bro tree, brotensor is already a target and gets reused
backend and all. Tests are on by default and build only for a standalone
configure (`BROIMAGE_TESTS`); installation is opt-in via `BROIMAGE_INSTALL`.
