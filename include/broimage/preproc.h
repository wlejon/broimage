#pragma once

#include <cstdint>

namespace broimage {

// `Y = src * scale + bias`, while shuffling layout from packed NHWC uint8 to
// planar NCHW float32. Host-side mirror of brotensor's
// `image_u8_to_f32_nhwc_to_nchw`. Covers the typical scaling conventions:
//   [0, 255] -> [0, 1]   : scale = 1/255,  bias =  0
//   [0, 255] -> [-1, 1]  : scale = 2/255,  bias = -1
//
// Y must be sized for at least N*C*H*W floats.
void u8_nhwc_to_f32_nchw(const uint8_t* src,
                         int N, int H, int W, int C,
                         float scale, float bias,
                         float* Y);

// Convenience: shuffle without dtype change (float NHWC -> float NCHW).
void nhwc_to_nchw_f32(const float* src,
                      int N, int H, int W, int C,
                      float* Y);

void nchw_to_nhwc_f32(const float* src,
                      int N, int C, int H, int W,
                      float* Y);

// Inverse of `u8_nhwc_to_f32_nchw`: `Y[n,h,w,c] = clamp(round(
// src[n,c,h,w] * scale + bias), 0, 255)`. Round-trips a preprocessor
// tensor back to a writable image (debug visualization, brodiffusion
// decode side). The typical inverse scalings are:
//   [0, 1]   -> [0, 255]  : scale = 255,    bias = 0
//   [-1, 1]  -> [0, 255]  : scale = 127.5,  bias = 127.5
//
// Y must be sized for at least N*H*W*C bytes.
void f32_nchw_to_u8_nhwc(const float* src,
                         int N, int C, int H, int W,
                         float scale, float bias,
                         uint8_t* Y);

} // namespace broimage
