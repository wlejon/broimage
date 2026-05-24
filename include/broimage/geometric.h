#pragma once

#include <cstdint>

namespace broimage {

// ----- Resize ----------------------------------------------------------------
//
// All resizes operate on interleaved HWC float32 buffers. The center-pixel rule
// (`(i + 0.5) * src/dst - 0.5`) matches the brokit `bro.image.resample` op and
// PyTorch's `align_corners=False`, which is what every model preprocessor in
// the bro stack (CLIP, SAM, Qwen3.5-VL) expects.

enum class Filter {
    Nearest,
    Bilinear,
    Bicubic,    // Catmull-Rom cubic
    Lanczos3,   // Lanczos windowed sinc, radius 3 — the high-quality choice
                // for upscales and arbitrary-ratio resizes.
    Area,       // Box / area average. The correct choice for *downscales* —
                // bilinear/bicubic alias badly at large reduction ratios.
                // Falls back to bilinear when the destination is larger than
                // the source (no upscale meaning for an area filter).
};

// HWC interleaved float32. dst must be at least dst_w*dst_h*channels floats.
void resize_hwc_f32(const float* src, int src_w, int src_h, int channels,
                    float*       dst, int dst_w, int dst_h,
                    Filter filter = Filter::Bilinear);

// CHW planar float32 (per-channel). dst must be at least channels*dst_w*dst_h
// floats. The layout used by brolm/qwen3.5-VL and brodiffusion preprocessors.
void resize_chw_f32(const float* src, int src_w, int src_h, int channels,
                    float*       dst, int dst_w, int dst_h,
                    Filter filter = Filter::Bilinear);

// HWC interleaved uint8 (RGBA8 the decode_file output uses). dst must be at
// least dst_w*dst_h*channels bytes. Computes in float internally then rounds.
void resize_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                   uint8_t*       dst, int dst_w, int dst_h,
                   Filter filter = Filter::Bilinear);

// ----- Letterbox / pad -------------------------------------------------------
//
// "Letterbox" scales the source to fit within (dst_w, dst_h) preserving aspect
// ratio, centers it, and fills the remaining area with `pad`. The actual
// content rect inside dst is reported back via `out_x/out_y/out_w/out_h` (set
// to nullptr to discard).
void letterbox_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                      uint8_t*       dst, int dst_w, int dst_h,
                      uint8_t pad_r, uint8_t pad_g, uint8_t pad_b, uint8_t pad_a,
                      Filter filter = Filter::Bilinear,
                      int* out_x = nullptr, int* out_y = nullptr,
                      int* out_w = nullptr, int* out_h = nullptr);

// Constant-pad an HWC u8 image into a destination rectangle with the source
// placed at (off_x, off_y). The source area outside dst is clipped.
void pad_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                uint8_t*       dst, int dst_w, int dst_h,
                int off_x, int off_y,
                uint8_t pad_r, uint8_t pad_g, uint8_t pad_b, uint8_t pad_a);

// ----- Crop ------------------------------------------------------------------
//
// Copies a [x, y, w, h] rect out of `src` into `dst` (which must be at least
// w*h*channels bytes). Out-of-range pixels are clamped to the source edge.
void crop_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                 uint8_t*       dst,
                 int x, int y, int w, int h);

// Center-crop the largest centered square (or arbitrary rect) out of an HWC u8
// image. Equivalent to crop_hwc_u8 with the rect derived from src dims.
void center_crop_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                        uint8_t*       dst,
                        int crop_w, int crop_h);

// ----- Flip / rotate ---------------------------------------------------------

void flip_horizontal_hwc_u8(const uint8_t* src, uint8_t* dst,
                            int w, int h, int channels);
void flip_vertical_hwc_u8(const uint8_t* src, uint8_t* dst,
                          int w, int h, int channels);

// Rotate by a multiple of 90 degrees. `turns` is the number of 90-CCW turns
// (0..3; other values are reduced mod 4). dst dims swap when turns is odd.
// dst must be sized for the rotated image.
void rotate_90_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                      uint8_t* dst, int turns);

} // namespace broimage
