# broimage

[![CI](https://github.com/wlejon/broimage/actions/workflows/ci.yml/badge.svg)](https://github.com/wlejon/broimage/actions/workflows/ci.yml)
[![CodeQL](https://github.com/wlejon/broimage/actions/workflows/codeql.yml/badge.svg)](https://github.com/wlejon/broimage/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Image processing for the bro stack — decode/encode, geometric and color ops,
normalization, and composable typed-buffer kernels. Pure C++20, built on
[bromath](https://github.com/wlejon/bromath) and, optionally,
[brotensor](https://github.com/wlejon/brotensor) (the tensor adapter). The
kernels run on the CPU, accelerated by the [brass](https://github.com/wlejon/brass)
JIT where brass is present. GPU preprocessing goes through brotensor and runs on
whichever backend the tensor lives on: CUDA, Metal or Vulkan.

broimage is the single home for image work across the
[bro ecosystem](https://github.com/wlejon/bro/blob/main/docs/ecosystem.md).
Everything routes through it: [bro](https://github.com/wlejon/bro)'s HTML
`Image` decode and `bro.image` JS kernels (the binding in `src/api/`,
`broimage_api`), brokit, brolm's CLIP/SAM/VLM host-side resize + normalize,
brovisionml and brodiffusion's pixel preprocessing, bromux's inline terminal
images, and brothumb's thumbnails. Nothing else in the tree should carry its own
decode path or resize kernel.

Built and tested on Windows, Linux and macOS (x86-64 and arm64).

## Scope

- **Decode / encode** — stb_image-backed RGBA load (PNG, JPEG, BMP, TGA, GIF,
  PSD, PIC, PNM) and PNG/JPEG write, from files or in-memory buffers.
  `probe_dimensions_memory` reads width/height/channels from a header without
  decoding pixels. EXIF orientation honored on demand via `decode_file_oriented`
  (phone JPEGs load upright instead of sideways), with `read_exif_orientation`
  and `apply_exif_orientation` exposed for callers holding their own buffers.
  16-bit (`decode_file_u16`) and HDR / float (`decode_file_f32`) paths for depth
  maps and Radiance sources.
- **Bounded and animated decode** — `decode_memory_bounded` reads dimensions
  and frame count from the headers and refuses input over caller-set
  `DecodeLimits` before decoding anything (for untrusted bytes); decodes every
  frame of an animated GIF, and baseline uncompressed TIFF besides stb's
  formats.
- **KTX2** — `transcode_ktx2` turns a `.ktx2` (ETC1S/BasisLZ or UASTC,
  optionally Zstd-supercompressed) into RGBA8 or BC1/3/4/5/7 mips, through a
  vendored transcode-only slice of basis_universal.
- **PNG text** — `read_png_info` reads IHDR and every tEXt / zTXt / iTXt chunk
  without decoding pixels; `encode_png_memory_with_text` writes a PNG carrying
  key/value tags (the Freedesktop thumbnail cache's `Thumb::URI` /
  `Thumb::MTime`, `Software`, ...).
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

- **JIT** — with brass present (`BROIMAGE_WITH_JIT`, on by default), the fused
  resize + u8 → f32 + normalize + HWC → CHW preprocessing pipeline compiles to
  native code per shape (`preproc_jit.h`, `broimage/jit/`); without it the same
  call runs the plain C++ path.
- **JavaScript binding** — `broimage_api` mounts the decode, codec, geometric,
  color, preproc and kernel surface on `bro.image` in bronze, the JavaScript
  runtime bro runs apps on.

GPU image ops live in brotensor (one place for the CUDA, Metal and Vulkan
kernels); broimage calls into them when handed a GPU `brotensor::Tensor`, and
the op runs on that tensor's device.

broimage's GPU story is brotensor *compute* on tensors — it has no WebGL and
does not render. The JS `bro.image.gpu.*` surface (`colormap`, `fbm2D`) is a
**WebGL2 renderer that lives in bro**, not in broimage; it shares the
`bro.image` namespace with these CPU kernels for ergonomics (e.g. the CPU
`lookup` and GPU `colormap` both consume a LUT built by `bro.image.gradient`),
but is a separate layer.

See bro's [multi-repo workflow](https://github.com/wlejon/bro/blob/main/docs/multi-repo-workflow.md)
for how this slots into the multi-repo dev loop.

## Build

```
cmake -B build
cmake --build build --config Debug
ctest --test-dir build -C Debug
```

broimage ships no GPU language of its own. The `BROTENSOR_WITH_CUDA` /
`BROTENSOR_WITH_METAL` / `BROTENSOR_WITH_VULKAN` options only forward the
backend choice so a standalone GPU build resolves that brotensor backend (the
GPU image kernels live there). Built inside the bro tree, brotensor is already a
target and gets reused backend and all. Tests are on by default and build only
for a standalone configure (`BROIMAGE_TESTS`); installation is opt-in via
`BROIMAGE_INSTALL`. `BROIMAGE_WITH_TENSOR=OFF` drops brotensor entirely,
`BROIMAGE_ENABLE_API=OFF` drops the JavaScript binding (and with it bronze), and
`BROIMAGE_WITH_JIT=OFF` keeps brass out of the build even when it is present
(plain C++ kernels only), for consumers that only decode, encode and resize.

**Dependencies.** bromath, brotensor, and (for the JavaScript binding)
[bronze](https://github.com/wlejon/bronze) with [brass](https://github.com/wlejon/brass)
resolve the way every repo in the ecosystem resolves one (`cmake/bro_deps.cmake`):
an existing target wins, then a working tree beside this one (`../<name>`), then the
commit `CMakeLists.txt` pins, fetched at configure. There are no submodules; override
any dependency with `-DFETCHCONTENT_SOURCE_DIR_<NAME>=<path>`.

```bash
# A plain clone builds as is; working trees beside it (../bromath, ../bronze, ...)
# replace the fetched heads of their main branches for local development.
git clone https://github.com/wlejon/broimage
cmake -S broimage -B build
```

bronze and brass compile inside the build tree, because the binding has to be
compiled against the same bronze as the program that loads it.

## CI

Builds and tests on Linux (GCC + Clang), Windows (MSVC) and macOS/arm64, in the
default configuration that brolm and brosoundml consume. A separate job builds
`BROIMAGE_WITH_TENSOR=OFF` — the minimal, Tensor-free configuration a bare `bro`
build asks for. Nobody develops in it and every default build has the adapter on,
so a symbol that leaks outside the `BROIMAGE_WITH_TENSOR` guard compiles fine
everywhere except there; the job exists to catch that. Every job builds a plain
clone with no sibling checkouts, so the pinned dependencies stay buildable.

Coverage of `src/` + `include/broimage/` lands in each run's job summary
(`-DBROIMAGE_COVERAGE=ON` locally; GCC/Clang only). [CodeQL](.github/workflows/codeql.yml)
analyses the decoders weekly and on every push — PNG/JPEG bytes are the most
untrusted input in the stack, and the ops downstream index buffers using
width/height arithmetic taken straight from a file header. Vendored
`third_party/stb` is excluded from both: it isn't ours to fix, and its findings
belong upstream.

## Versioning

Pre-1.0. Consumers build this repo from source (a sibling working tree or a
pinned commit), so a tag is a pin point rather than a compatibility promise.

## License

[MIT](LICENSE)
