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

} // namespace broimage
