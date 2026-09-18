#pragma once

#include <cstdint>
#include <vector>

namespace broimage::jit {

struct ImagePipelineDesc {
    int32_t src_w = 0, src_h = 0;
    int32_t dst_w = 0, dst_h = 0;
    bool premultiply_alpha = true;
    bool to_float_planar = false;
    std::vector<float> mean;    // per channel
    std::vector<float> std_dev; // per channel

    uint64_t compute_hash() const;

    bool operator==(const ImagePipelineDesc& o) const {
        return src_w == o.src_w &&
               src_h == o.src_h &&
               dst_w == o.dst_w &&
               dst_h == o.dst_h &&
               premultiply_alpha == o.premultiply_alpha &&
               to_float_planar == o.to_float_planar &&
               mean == o.mean &&
               std_dev == o.std_dev;
    }
};

} // namespace broimage::jit

namespace broimage {
using jit::ImagePipelineDesc;
} // namespace broimage
