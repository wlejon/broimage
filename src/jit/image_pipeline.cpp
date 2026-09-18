#include "broimage/jit/image_pipeline.h"

#include <cstddef>
#include <cstdint>

namespace broimage::jit {

uint64_t ImagePipelineDesc::compute_hash() const {
    // 64-bit FNV-1a hash algorithm
    uint64_t hash = 14695981039346656037ULL;
    const uint64_t prime = 1099511628211ULL;

    auto hash_bytes = [&](const void* data, size_t len) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; ++i) {
            hash ^= static_cast<uint64_t>(bytes[i]);
            hash *= prime;
        }
    };

    hash_bytes(&src_w, sizeof(src_w));
    hash_bytes(&src_h, sizeof(src_h));
    hash_bytes(&dst_w, sizeof(dst_w));
    hash_bytes(&dst_h, sizeof(dst_h));

    const uint8_t pa = premultiply_alpha ? 1 : 0;
    hash_bytes(&pa, sizeof(pa));

    const uint8_t fp = to_float_planar ? 1 : 0;
    hash_bytes(&fp, sizeof(fp));

    const uint64_t mean_len = static_cast<uint64_t>(mean.size());
    hash_bytes(&mean_len, sizeof(mean_len));
    if (!mean.empty()) {
        hash_bytes(mean.data(), mean.size() * sizeof(float));
    }

    const uint64_t std_len = static_cast<uint64_t>(std_dev.size());
    hash_bytes(&std_len, sizeof(std_len));
    if (!std_dev.empty()) {
        hash_bytes(std_dev.data(), std_dev.size() * sizeof(float));
    }

    return hash;
}

} // namespace broimage::jit
