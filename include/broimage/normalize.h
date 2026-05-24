#pragma once

namespace broimage {

// Per-channel `(x - mean[c]) / std[c]` on NCHW float32. Host-side mirror of
// brotensor's `image_normalize` op (which owns the GPU variants). When called
// with a brotensor::Tensor, prefer the tensor_adapter that dispatches to the
// brotensor op directly — this is the raw host-pointer variant for everything
// that hasn't been wrapped in a Tensor yet (decoded RGBA, brokit kernel
// outputs, etc).
//
// `X` and `Y` may alias.
void image_normalize_nchw_f32(const float* X,
                              const float* mean,    // length C
                              const float* std_,    // length C
                              int N, int C, int H, int W,
                              float* Y);

} // namespace broimage
