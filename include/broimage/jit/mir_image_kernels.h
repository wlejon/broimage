#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if BROIMAGE_HAS_BRASS_JIT
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/gpu_portability.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>
#endif

namespace broimage::jit {

enum class MirKernelKind : uint8_t {
    AffineClampF32 = 0,
    RgbToGrayF32 = 1,
    ChwNormalizeF32 = 2,
    SeparableBlurHF32 = 3,
    SeparableBlurVF32 = 4,
    BilinearResizeF32 = 5,
};

enum class MirExecTarget : uint8_t {
    CpuSimd = 0,
    Gpu = 1,
};

using MirAffineClampF32Fn = void (*)(
    const float* src,
    float* dst,
    int32_t total,
    float alpha,
    float beta,
    float clamp_lo,
    float clamp_hi
);

using MirRgbToGrayF32Fn = void (*)(
    const float* src_rgb,
    float* dst_gray,
    int32_t width,
    int32_t height
);

using MirChwNormalizeF32Fn = void (*)(
    const float* src_chw,
    float* dst_chw,
    int32_t width,
    int32_t height,
    float mean0,
    float mean1,
    float mean2,
    float inv_std0,
    float inv_std1,
    float inv_std2
);

using MirSeparableBlur1DF32Fn = void (*)(
    const float* src,
    float* dst,
    const float* weights,
    int32_t width,
    int32_t height,
    int32_t radius
);

using MirBilinearResizeF32Fn = void (*)(
    const float* src,
    float* dst,
    int32_t src_w,
    int32_t src_h,
    int32_t dst_w,
    int32_t dst_h
);

#if BROIMAGE_HAS_BRASS_JIT

// Lowers one of the core 2D image kernels to portable Brass MIR for either
// CPU SIMD (loop-based vectorizable MIR) or GPU (portable gpu_* intrinsics
// with bounds guard and global memory accesses). Both modes share the same
// single-source per-pixel/per-element MIR math emitter.
brass::Function* build_mir_image_kernel(
    brass::Module& mod,
    MirKernelKind kind,
    MirExecTarget target,
    const char* fn_name = nullptr
);

struct CompiledGpuArtifacts {
    bool portability_ok = false;
    std::string portability_diagnostics;
    brass::target::SpirvKernel spirv;
    std::string ptx;
    brass::target::MslKernel msl;
};

// Verifies GPU portability on `fn` and compiles it to SPIR-V, PTX, and MSL.
bool compile_mir_kernel_gpu_targets(
    const brass::Function& fn,
    CompiledGpuArtifacts& out,
    uint32_t local_size_x = 64
);

class MirImageKernelCompiler {
public:
    static MirImageKernelCompiler& instance();

    MirImageKernelCompiler(const MirImageKernelCompiler&) = delete;
    MirImageKernelCompiler& operator=(const MirImageKernelCompiler&) = delete;

    static bool is_cpu_supported();
    static bool is_gpu_vulkan_available();

    // CPU SIMD JIT compilation & execution
    bool execute_affine_clamp_f32_cpu(
        const float* src, float* dst, int width, int height, int channels,
        float alpha, float beta, float clamp_lo, float clamp_hi
    );
    bool execute_rgb_to_gray_f32_cpu(
        const float* src_rgb, float* dst_gray, int width, int height
    );
    bool execute_chw_normalize_f32_cpu(
        const float* src_chw, float* dst_chw, int width, int height,
        const float mean[3], const float std_dev[3]
    );
    bool execute_separable_blur_1d_f32_cpu(
        const float* src, float* dst, const float* weights,
        int width, int height, int radius, bool horizontal
    );
    bool execute_separable_blur_f32_cpu(
        const float* src, float* dst, const float* weights,
        int width, int height, int radius
    );
    bool execute_bilinear_resize_f32_cpu(
        const float* src, int src_w, int src_h,
        float* dst, int dst_w, int dst_h
    );

    // GPU Vulkan SPIR-V compilation & execution
    bool execute_affine_clamp_f32_gpu(
        const float* src, float* dst, int width, int height, int channels,
        float alpha, float beta, float clamp_lo, float clamp_hi
    );
    bool execute_rgb_to_gray_f32_gpu(
        const float* src_rgb, float* dst_gray, int width, int height
    );
    bool execute_chw_normalize_f32_gpu(
        const float* src_chw, float* dst_chw, int width, int height,
        const float mean[3], const float std_dev[3]
    );
    bool execute_separable_blur_1d_f32_gpu(
        const float* src, float* dst, const float* weights,
        int width, int height, int radius, bool horizontal
    );
    bool execute_separable_blur_f32_gpu(
        const float* src, float* dst, const float* weights,
        int width, int height, int radius
    );
    bool execute_bilinear_resize_f32_gpu(
        const float* src, int src_w, int src_h,
        float* dst, int dst_w, int dst_h
    );

    // Cache retrieval
    const brass::target::SpirvKernel* get_or_compile_spirv(MirKernelKind kind);
    void clear_cache();
    size_t cpu_cache_size();
    size_t gpu_cache_size();

private:
    MirImageKernelCompiler() = default;
    ~MirImageKernelCompiler() = default;

    template <typename FnPtr>
    FnPtr get_or_compile_cpu(MirKernelKind kind);

    std::mutex mutex_;
    std::unordered_map<uint8_t, brass::codegen::KernelFunction> cpu_cache_;
    std::unordered_map<uint8_t, brass::target::SpirvKernel> spirv_cache_;
};

#endif // BROIMAGE_HAS_BRASS_JIT

} // namespace broimage::jit

namespace broimage {

// Host-usable dispatch APIs (route to GPU Vulkan when prefer_gpu && available,
// otherwise CPU SIMD Brass JIT).
bool mir_affine_clamp_f32(
    const float* src, float* dst, int width, int height, int channels,
    float alpha, float beta, float clamp_lo, float clamp_hi,
    bool prefer_gpu = false
);

bool mir_rgb_to_gray_f32(
    const float* src_rgb, float* dst_gray, int width, int height,
    bool prefer_gpu = false
);

bool mir_chw_normalize_f32(
    const float* src_chw, float* dst_chw, int width, int height,
    const float mean[3], const float std_dev[3],
    bool prefer_gpu = false
);

bool mir_separable_blur_f32(
    const float* src, float* dst, int width, int height,
    const float* weights, int radius,
    bool prefer_gpu = false
);

bool mir_bilinear_resize_f32(
    const float* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    bool prefer_gpu = false
);

} // namespace broimage
