#pragma once

#include <cstdint>

namespace broimage {

// JIT-accelerated fused image preprocessing:
// Bilinear resize + RGB uint8 to float32 + per-channel normalization ((val / 255.0f - mean) / std_dev)
// + HWC-to-CHW planar transposition directly to dst_planar.
// Falls back to scalar pipeline if JIT is unsupported or unavailable.
void fused_u8_nhwc_to_f32_nchw_normalized(
    const uint8_t* src,
    int src_w, int src_h,
    float* dst_planar,
    int dst_w, int dst_h,
    const float* mean = nullptr,
    const float* std_dev = nullptr
);

} // namespace broimage
