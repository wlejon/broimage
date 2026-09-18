#include "broimage/jit/jit_image_compiler.h"
#include "broimage/geometric.h"
#include "broimage/normalize.h"
#include "broimage/preproc.h"
#include "broimage/preproc_jit.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#include <thread>
#endif

namespace broimage::jit {

namespace {

template <typename SliceFn>
void run_multithreaded_slices(int dst_h, SliceFn&& slice_fn) {
#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    if (dst_h < num_threads * 2 || num_threads <= 1) {
        slice_fn(0, dst_h);
    } else {
        int chunk = (dst_h + num_threads - 1) / num_threads;
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < num_threads; ++t) {
            int y_start = t * chunk;
            int y_end = std::min(y_start + chunk, dst_h);
            if (y_start < y_end) {
                slice_fn(y_start, y_end);
            }
        }
    }
#else
    unsigned num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 1;
    if (dst_h < static_cast<int>(num_threads) * 2 || num_threads <= 1) {
        slice_fn(0, dst_h);
    } else {
        int chunk = (dst_h + static_cast<int>(num_threads) - 1) / static_cast<int>(num_threads);
        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for (unsigned t = 0; t < num_threads; ++t) {
            int y_start = static_cast<int>(t) * chunk;
            int y_end = std::min(y_start + chunk, dst_h);
            if (y_start < y_end) {
                workers.emplace_back([=, &slice_fn]() {
                    slice_fn(y_start, y_end);
                });
            }
        }
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
    }
#endif
}

} // namespace

JitImageCompiler& JitImageCompiler::instance() {
    static JitImageCompiler s_instance;
    return s_instance;
}

bool JitImageCompiler::is_supported() {
#if BROIMAGE_HAS_BRASS_JIT
    return brass::Target::host().is_x64() || brass::Target::host().is_aarch64();
#else
    return false;
#endif
}

void JitImageCompiler::clear_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    preproc_fn = nullptr;
    resize_fn = nullptr;
}

size_t JitImageCompiler::cache_size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

brass::codegen::ImageResizeRgba8Fn JitImageCompiler::get_or_compile_resize(const ImagePipelineDesc& desc) {
#if BROIMAGE_HAS_BRASS_JIT
    if (!is_supported()) return nullptr;

    uint64_t hash = desc.compute_hash();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(hash);
        if (it != cache_.end() && it->second.resize_fn) {
            this->resize_fn = it->second.resize_fn;
            return this->resize_fn;
        }
    }

    brass::Module mod("jit_resize_rgba8_" + std::to_string(hash));
    brass::codegen::ImageBuilder builder(mod);
    auto* fn = builder.build_resize_rgba8_function("image_resize_rgba8");
    if (!fn) return nullptr;

    brass::codegen::KernelJit jit;
    auto kfn = jit.compile(*fn);
    if (!kfn) return nullptr;

    auto fn_ptr = kfn.as<brass::codegen::ImageResizeRgba8Fn>();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CompiledKernelEntry entry;
        entry.kfn = std::move(kfn);
        entry.resize_fn = fn_ptr;
        cache_[hash] = std::move(entry);
        this->resize_fn = fn_ptr;
    }
    return fn_ptr;
#else
    (void)desc;
    return nullptr;
#endif
}

brass::codegen::ImagePreprocFusedFn JitImageCompiler::get_or_compile_preproc(const ImagePipelineDesc& desc) {
#if BROIMAGE_HAS_BRASS_JIT
    if (!is_supported()) return nullptr;

    uint64_t hash = desc.compute_hash();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(hash);
        if (it != cache_.end() && it->second.preproc_fn) {
            this->preproc_fn = it->second.preproc_fn;
            return this->preproc_fn;
        }
    }

    brass::Module mod("jit_fused_preproc_" + std::to_string(hash));
    brass::codegen::ImageBuilder builder(mod);
    auto* fn = builder.build_fused_preproc_function("image_preproc_fused", 3);
    if (!fn) return nullptr;

    brass::codegen::KernelJit jit;
    auto kfn = jit.compile(*fn);
    if (!kfn) return nullptr;

    auto fn_ptr = kfn.as<brass::codegen::ImagePreprocFusedFn>();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        CompiledKernelEntry entry;
        entry.kfn = std::move(kfn);
        entry.preproc_fn = fn_ptr;
        cache_[hash] = std::move(entry);
        this->preproc_fn = fn_ptr;
    }
    return fn_ptr;
#else
    (void)desc;
    return nullptr;
#endif
}

bool JitImageCompiler::execute_resize_rgba8(
    const uint8_t* src, uint8_t* dst,
    int src_w, int src_h, int dst_w, int dst_h,
    bool premultiply_alpha
) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return false;
    }
    if (!is_supported()) {
        return false;
    }

    ImagePipelineDesc desc;
    desc.src_w = src_w;
    desc.src_h = src_h;
    desc.dst_w = dst_w;
    desc.dst_h = dst_h;
    desc.premultiply_alpha = premultiply_alpha;
    desc.to_float_planar = false;

    auto fn = get_or_compile_resize(desc);
    if (!fn) {
        return false;
    }

    run_multithreaded_slices(dst_h, [&](int y_start, int y_end) {
        fn(src, dst, src_w, src_h, dst_w, dst_h, y_start, y_end, premultiply_alpha);
    });

    return true;
}

bool JitImageCompiler::execute_fused_preproc(
    const uint8_t* src, float* dst_planar,
    int src_w, int src_h, int dst_w, int dst_h,
    const float* mean, const float* std_dev
) {
    if (!src || !dst_planar || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        return false;
    }
    if (!is_supported()) {
        return false;
    }

    const float default_mean[3] = {0.0f, 0.0f, 0.0f};
    const float default_std[3]  = {1.0f, 1.0f, 1.0f};
    const float* m = mean ? mean : default_mean;
    const float* s = std_dev ? std_dev : default_std;

    const float inv_std[3] = {
        s[0] != 0.0f ? (1.0f / s[0]) : 0.0f,
        s[1] != 0.0f ? (1.0f / s[1]) : 0.0f,
        s[2] != 0.0f ? (1.0f / s[2]) : 0.0f
    };

    ImagePipelineDesc desc;
    desc.src_w = src_w;
    desc.src_h = src_h;
    desc.dst_w = dst_w;
    desc.dst_h = dst_h;
    desc.premultiply_alpha = false;
    desc.to_float_planar = true;
    desc.mean.assign(m, m + 3);
    desc.std_dev.assign(s, s + 3);

    auto fn = get_or_compile_preproc(desc);
    if (!fn) {
        return false;
    }

    run_multithreaded_slices(dst_h, [&](int y_start, int y_end) {
        fn(src, dst_planar, src_w, src_h, dst_w, dst_h, y_start, y_end, m, inv_std);
    });

    return true;
}

} // namespace broimage::jit

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
