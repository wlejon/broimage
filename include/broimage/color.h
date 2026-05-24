#pragma once

#include <cstdint>

namespace broimage {

// ----- Channel conversion (uint8) -------------------------------------------
//
// `pixels` is always interpreted as a flat HWC interleaved buffer; the kernels
// don't care about dims beyond a flat element count.

void rgba_to_rgb_u8(const uint8_t* src, uint8_t* dst, int pixel_count);
void rgb_to_rgba_u8(const uint8_t* src, uint8_t* dst, int pixel_count,
                    uint8_t alpha = 255);

// Rec.601 luma (matches stb_image's grayscale conversion).
void rgba_to_gray_u8(const uint8_t* src, uint8_t* dst, int pixel_count);
void rgb_to_gray_u8(const uint8_t* src, uint8_t* dst, int pixel_count);

// ----- Layout conversion (float) --------------------------------------------
//
// HWC <-> CHW. The interleaved->planar shuffle that every model preprocess
// pipeline does on the way into a Tensor.

void hwc_to_chw_f32(const float* src, float* dst,
                    int width, int height, int channels);
void chw_to_hwc_f32(const float* src, float* dst,
                    int width, int height, int channels);

// ----- Gamma / sRGB ---------------------------------------------------------
//
// In-place when src == dst is allowed. Buffers are interpreted as flat element
// counts. The sRGB <-> linear pair uses the IEC 61966-2-1 piecewise curve.

void apply_gamma_f32(const float* src, float* dst, int element_count,
                     float gamma);

void srgb_to_linear_f32(const float* src, float* dst, int element_count);
void linear_to_srgb_f32(const float* src, float* dst, int element_count);

void srgb_to_linear_u8_to_f32(const uint8_t* src, float* dst, int element_count);
void linear_f32_to_srgb_u8(const float* src, uint8_t* dst, int element_count);

// ----- HSV / HSL (float, all components in [0, 1]) --------------------------
//
// `pixel_count` triples are read from src and written to dst. RGB and HSV/HSL
// are both stored as three floats in [0, 1]; hue is normalized to [0, 1)
// (multiply by 360 for degrees). src and dst may alias.
//
// Standard Wikipedia formulas — non-linear in RGB, intended for tint /
// saturation tweaks rather than colorimetric work.

void rgb_to_hsv_f32(const float* src, float* dst, int pixel_count);
void hsv_to_rgb_f32(const float* src, float* dst, int pixel_count);
void rgb_to_hsl_f32(const float* src, float* dst, int pixel_count);
void hsl_to_rgb_f32(const float* src, float* dst, int pixel_count);

// ----- Color matrix ---------------------------------------------------------
//
// Apply a 3x3 or 3x4 color matrix to each pixel of an interleaved RGB-or-RGBA
// float32 buffer.
//
// 3x3 form: `[R' G' B']^T = M * [R G B]^T`. Alpha (when channels == 4) passes
// through unchanged. Useful for color-space conversions (e.g. sRGB->YCbCr),
// saturation / hue rotations, channel mixers.
//
// 3x4 form: `[R' G' B']^T = M * [R G B 1]^T` — the last column is a bias.
// Useful for "matrix + offset" pipelines (white balance, channel rescale +
// DC shift). Same alpha pass-through.
//
// `channels` must be 3 or 4. src and dst may alias. Matrices are row-major.
void apply_color_matrix_3x3_f32(const float* src, float* dst, int pixel_count,
                                int channels, const float matrix[9]);
void apply_color_matrix_3x4_f32(const float* src, float* dst, int pixel_count,
                                int channels, const float matrix[12]);

} // namespace broimage
