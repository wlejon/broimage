#include "broimage/jit/jit_image_compiler.h"
#include "broimage/geometric.h"
#include "broimage/normalize.h"
#include "broimage/preproc.h"
#include "broimage/preproc_jit.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#if BROIMAGE_HAS_BRASS_JIT
#include <brass/cache/artifact_cache.hpp>
#include <brass/cache/native_module_cache.hpp>
#endif

#ifdef _OPENMP
#include <omp.h>
#else
#include <thread>
#endif

namespace broimage::jit {

namespace {

template <typename SliceFn>
void run_multithreaded_slices(int dst_h, SliceFn&& slice_fn) {
    if (dst_h <= 0) return;
#if BROIMAGE_HAS_BRASS_JIT
    if (dst_h < 4) {
        slice_fn(0, dst_h);
    } else {
        brass::codegen::parallel_for(0, static_cast<int64_t>(dst_h), [&](int64_t y_start, int64_t y_end) {
            if (y_start < y_end) {
                slice_fn(static_cast<int>(y_start), static_cast<int>(y_end));
            }
        });
    }
#elif defined(_OPENMP)
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

#if BROIMAGE_HAS_BRASS_JIT
std::filesystem::path image_cache_directory() {
    if (const char* env_dir = std::getenv("BROIMAGE_CACHE_DIR"); env_dir && env_dir[0] != '\0') {
        return std::filesystem::path(env_dir);
    }
#if defined(_WIN32)
    if (const char* local_app = std::getenv("LOCALAPPDATA"); local_app && local_app[0] != '\0') {
        return std::filesystem::path(local_app) / "broimage" / "kernels";
    }
    if (const char* user_prof = std::getenv("USERPROFILE"); user_prof && user_prof[0] != '\0') {
        return std::filesystem::path(user_prof) / ".cache" / "broimage" / "kernels";
    }
#else
    if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && xdg[0] != '\0') {
        return std::filesystem::path(xdg) / "broimage" / "kernels";
    }
    if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
        return std::filesystem::path(home) / ".cache" / "broimage" / "kernels";
    }
#endif
    std::error_code ec;
    auto tmp = std::filesystem::temp_directory_path(ec);
    if (!ec && !tmp.empty()) {
        return tmp / "broimage" / "kernels";
    }
    return std::filesystem::path(".cache") / "broimage" / "kernels";
}

brass::cache::NativeModuleCache& image_native_cache() {
    static constexpr uint64_t kMaxBytes = 64ULL * 1024 * 1024;
    static auto* store = new brass::cache::ArtifactCache(image_cache_directory(), kMaxBytes);
    static auto* cache = new brass::cache::NativeModuleCache(*store);
    return *cache;
}

template <typename BuildFn>
brass::codegen::KernelFunction compile_cached_image_kernel(
    const char* kind,
    const char* module_name,
    const char* sym_name,
    BuildFn&& build_fn) {
    using namespace brass::codegen;
    KernelOptions opts;
    brass::cache::KeyBuilder kb("broimage.kernel/1");
    kb.add("kind", kind);
    kb.add_u64("opt_level", static_cast<uint64_t>(opts.opt_level));
    kb.add_bool("enable_optimizations", opts.enable_optimizations);
    kb.add_bool("enable_sroa", opts.enable_sroa);
    kb.add_bool("enable_bce", opts.enable_bce);
    kb.add_bool("enable_smt_guard_elim", opts.enable_smt_guard_elim);
    kb.add_bool("enable_superopt", opts.enable_superopt);
    kb.add_bool("enable_avx2", opts.enable_avx2);
    kb.add_bool("enable_fma", opts.enable_fma);
    kb.add_bool("enable_fp_reassociation", opts.enable_fp_reassociation);
    kb.add_bool("enable_vectorize", opts.enable_vectorize);
    kb.add_bool("enable_slp", opts.enable_slp);
    kb.add_bool("enable_unroll", opts.enable_unroll);
    kb.add_u64("unroll_factor", static_cast<uint64_t>(opts.unroll_factor));
    kb.add_bool("enable_parallel", opts.enable_parallel);
    kb.add_bool("enable_ftz_daz", opts.enable_ftz_daz);

    const brass::cache::NativeOptions native_opts;
    std::string cache_err;
    auto native_mod = image_native_cache().get_or_build_object(
        kb.finish(),
        [&](brass::object::ObjectFile& obj, std::string& err) -> bool {
            brass::Module mod(module_name);
            if (!build_fn(mod)) {
                err = "Image kernel MIR build failed";
                return false;
            }
            KernelJit jit(opts);
            obj = jit.compile_object(mod);
            return true;
        },
        native_opts,
        &cache_err);
    if (native_mod) {
        if (void* sym = native_mod->symbol(sym_name)) {
            return KernelFunction::from_owner(sym_name, sym, 0, std::move(native_mod));
        }
    }

    brass::Module mod(module_name);
    if (!build_fn(mod)) return {};
    KernelJit jit(opts);
    return jit.compile(mod, sym_name);
}
#endif

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

    auto kfn = compile_cached_image_kernel(
        "image_resize_rgba8",
        "jit_resize_rgba8",
        "image_resize_rgba8",
        [](brass::Module& mod) {
            brass::codegen::ImageBuilder builder(mod);
            return builder.build_resize_rgba8_function("image_resize_rgba8") != nullptr;
        });
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

    auto kfn = compile_cached_image_kernel(
        "image_preproc_fused_c3",
        "jit_fused_preproc_c3",
        "image_preproc_fused",
        [](brass::Module& mod) {
            brass::codegen::ImageBuilder builder(mod);
            return builder.build_fused_preproc_function("image_preproc_fused", 3) != nullptr;
        });
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

// (The public fused_u8_nhwc_to_f32_nchw_normalized is src/preproc_fused.cpp's,
// built with or without the JIT.)
