// The fused preprocessing entry point (broimage/preproc_jit.h): the JIT's
// kernel where brass is built in, else the scalar pipeline. Compiled in every
// build, so callers link whether or not BROIMAGE_WITH_JIT is on.
#include "broimage/geometric.h"
#include "broimage/normalize.h"
#include "broimage/preproc.h"
#include "broimage/preproc_jit.h"

#if BROIMAGE_HAS_BRASS_JIT
#include "broimage/jit/jit_image_compiler.h"
#endif

#include <cstddef>
#include <vector>

namespace broimage {

void fused_u8_nhwc_to_f32_nchw_normalized(
    const uint8_t* src,
    int src_w, int src_h,
    float* dst_planar,
    int dst_w, int dst_h,
    const float* mean,
    const float* std_dev
) {
    if (!src || !dst_planar || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return;
    }

#if BROIMAGE_HAS_BRASS_JIT
    if (jit::JitImageCompiler::instance().execute_fused_preproc(
            src, dst_planar, src_w, src_h, dst_w, dst_h, mean, std_dev)) {
        return;
    }
#endif

    // Fallback scalar pipeline
    std::vector<uint8_t> resized(static_cast<size_t>(dst_w) * dst_h * 3);
    resize_hwc_u8(src, src_w, src_h, 3, resized.data(), dst_w, dst_h, Filter::Bilinear);
    u8_nhwc_to_f32_nchw(resized.data(), 1, dst_h, dst_w, 3, 1.0f / 255.0f, 0.0f, dst_planar);
    const float default_mean[3] = {0.0f, 0.0f, 0.0f};
    const float default_std[3]  = {1.0f, 1.0f, 1.0f};
    image_normalize_nchw_f32(
        dst_planar,
        mean ? mean : default_mean,
        std_dev ? std_dev : default_std,
        1, 3, dst_h, dst_w,
        dst_planar
    );
}

} // namespace broimage
