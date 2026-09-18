#pragma once

#include "broimage/jit/image_pipeline.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

#if BROIMAGE_HAS_BRASS_JIT
#include <brass/codegen/image_builder.hpp>
#include <brass/codegen/kernel_jit.hpp>
#else
namespace brass::codegen {
typedef void (*ImagePreprocFusedFn)(
    const uint8_t* src,
    float* dst_planar,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h,
    int32_t y_start,
    int32_t y_end,
    const float* mean,
    const float* inv_std
);

typedef void (*ImageResizeRgba8Fn)(
    const uint8_t* src,
    uint8_t* dst,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h,
    int32_t y_start,
    int32_t y_end,
    bool premultiply_alpha
);

class KernelFunction {};
} // namespace brass::codegen
#endif

namespace broimage::jit {

struct CompiledKernelEntry {
#if BROIMAGE_HAS_BRASS_JIT
    brass::codegen::KernelFunction kfn;
#endif
    brass::codegen::ImagePreprocFusedFn preproc_fn = nullptr;
    brass::codegen::ImageResizeRgba8Fn resize_fn = nullptr;
};

class JitImageCompiler {
public:
    static JitImageCompiler& instance();

    JitImageCompiler(const JitImageCompiler&) = delete;
    JitImageCompiler& operator=(const JitImageCompiler&) = delete;

    // Returns whether JIT compilation is available and supported on this host
    static bool is_supported();

    // High-level multithreaded JIT execution
    bool execute_resize_rgba8(const uint8_t* src, uint8_t* dst,
                              int src_w, int src_h, int dst_w, int dst_h,
                              bool premultiply_alpha);

    bool execute_fused_preproc(const uint8_t* src, float* dst_planar,
                               int src_w, int src_h, int dst_w, int dst_h,
                               const float* mean, const float* std_dev);

    // Kernel compilation and retrieval
    brass::codegen::ImageResizeRgba8Fn get_or_compile_resize(const ImagePipelineDesc& desc);
    brass::codegen::ImagePreprocFusedFn get_or_compile_preproc(const ImagePipelineDesc& desc);

    // Cache management
    void clear_cache();
    size_t cache_size();

    // Holds the compiled function pointers
    brass::codegen::ImagePreprocFusedFn preproc_fn = nullptr;
    brass::codegen::ImageResizeRgba8Fn resize_fn = nullptr;

private:
    JitImageCompiler() = default;
    ~JitImageCompiler() = default;

    std::mutex mutex_;
    std::unordered_map<uint64_t, CompiledKernelEntry> cache_;
};

} // namespace broimage::jit

namespace broimage {
using jit::JitImageCompiler;
} // namespace broimage
