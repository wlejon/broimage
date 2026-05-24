#pragma once

#include <cstdint>

// Forward-declare brotensor::Tensor to keep this header free of brotensor's
// public surface. The .cpp pulls in <brotensor/tensor.h>.
namespace brotensor { struct Tensor; }

namespace broimage {

// Adapter calls that route through brotensor's `image_preproc` op set when a
// Tensor is involved. brotensor owns the GPU kernel variants (CUDA / Metal);
// these forwarders are how broimage stays the single library callers reach
// for, without absorbing GPU code itself.
//
// Each wrapper enforces the same shape rules as the underlying brotensor op
// and throws std::runtime_error on mismatch (matching brotensor's behavior).

// Per-channel mean/std on NCHW. `mean` and `std_` are Tensors of length C
// (FP32). `Y` is resized to (N, C*H*W) FP32 by the underlying op.
void image_normalize(const brotensor::Tensor& X,
                     const brotensor::Tensor& mean,
                     const brotensor::Tensor& std_,
                     int N, int C, int H, int W,
                     brotensor::Tensor& Y);

// `Y = src * scale + bias`, NHWC uint8 -> NCHW FP32. `Y` is resized to
// (N, C*H*W) FP32 by the underlying op.
void image_u8_to_f32_nhwc_to_nchw(const uint8_t* src,
                                  int N, int H, int W, int C,
                                  float scale, float bias,
                                  brotensor::Tensor& Y);

} // namespace broimage
