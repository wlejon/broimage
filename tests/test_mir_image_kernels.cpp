#include "broimage/geometric.h"
#include "broimage/jit/mir_image_kernels.h"
#include "broimage/kernels.h"
#include "broimage/normalize.h"

#if BROIMAGE_HAS_BRASS_JIT
#include <brass/gpu/vulkan_driver.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_failed = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++g_failed; \
    } \
} while (0)

#define CHECK_LE(a, b) do { \
    if (!((a) <= (b))) { \
        std::fprintf(stderr, "FAIL %s:%d: %s <= %s (%.8g <= %.8g)\n", \
                     __FILE__, __LINE__, #a, #b, static_cast<double>(a), static_cast<double>(b)); \
        ++g_failed; \
    } \
} while (0)

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30f;
    float mx = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

std::vector<float> make_pattern(size_t n, float scale = 1.0f, float bias = 0.0f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>((i * 37 + 11) % 257) / 256.0f;
        v[i] = t * scale + bias;
    }
    return v;
}

void scalar_ref_rgb_to_gray(const float* src, float* dst, int w, int h) {
    const int n = w * h;
    for (int i = 0; i < n; ++i) {
        dst[i] = 0.299f * src[i * 3 + 0] +
                 0.587f * src[i * 3 + 1] +
                 0.114f * src[i * 3 + 2];
    }
}

void scalar_ref_separable_blur(const float* src, float* dst, int w, int h,
                               const float* weights, int radius) {
    std::vector<float> tmp(static_cast<size_t>(w) * h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int sx = std::clamp(x + k, 0, w - 1);
                acc += src[y * w + sx] * weights[k + radius];
            }
            tmp[y * w + x] = acc;
        }
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                const int sy = std::clamp(y + k, 0, h - 1);
                acc += tmp[sy * w + x] * weights[k + radius];
            }
            dst[y * w + x] = acc;
        }
    }
}

void test_cpu_simd_parity() {
    std::printf("--- Test: CPU SIMD Brass MIR vs Scalar Reference ---\n");
    using broimage::jit::MirImageKernelCompiler;
    if (!MirImageKernelCompiler::is_cpu_supported()) {
        std::printf("  [SKIP] CPU JIT unsupported on current host architecture\n");
        return;
    }

    struct Dim { int w; int h; };
    const Dim dims[] = {{7, 5}, {17, 13}, {64, 48}, {129, 33}};

    for (const Dim d : dims) {
        // 1. Affine + Clamp (3 channels)
        {
            const int ch = 3;
            const size_t n = static_cast<size_t>(d.w) * d.h * ch;
            const auto src = make_pattern(n, 2.0f, -0.5f);
            std::vector<float> ref(n, 0.0f);
            std::vector<float> jit(n, 0.0f);
            const float alpha = 1.35f, beta = -0.15f, lo = 0.0f, hi = 1.0f;
            broimage::map_affine_clamp_f32(src.data(), ref.data(), static_cast<int>(n), alpha, beta, lo, hi);
            CHECK(broimage::mir_affine_clamp_f32(src.data(), jit.data(), d.w, d.h, ch, alpha, beta, lo, hi, false));
            const float diff = max_abs_diff(jit, ref);
            CHECK_LE(diff, 1e-5f);
        }

        // 2. RGB -> Grayscale
        {
            const size_t pix = static_cast<size_t>(d.w) * d.h;
            const auto src = make_pattern(pix * 3, 1.0f, 0.0f);
            std::vector<float> ref(pix, 0.0f);
            std::vector<float> jit(pix, 0.0f);
            scalar_ref_rgb_to_gray(src.data(), ref.data(), d.w, d.h);
            CHECK(broimage::mir_rgb_to_gray_f32(src.data(), jit.data(), d.w, d.h, false));
            const float diff = max_abs_diff(jit, ref);
            CHECK_LE(diff, 1e-5f);
        }

        // 3. CHW Normalize
        {
            const size_t n = static_cast<size_t>(3) * d.w * d.h;
            const auto src = make_pattern(n, 1.0f, 0.0f);
            std::vector<float> ref(n, 0.0f);
            std::vector<float> jit(n, 0.0f);
            const float mean[3] = {0.485f, 0.456f, 0.406f};
            const float std_dev[3] = {0.229f, 0.224f, 0.225f};
            broimage::image_normalize_nchw_f32(src.data(), mean, std_dev, 1, 3, d.h, d.w, ref.data());
            CHECK(broimage::mir_chw_normalize_f32(src.data(), jit.data(), d.w, d.h, mean, std_dev, false));
            const float diff = max_abs_diff(jit, ref);
            CHECK_LE(diff, 1e-5f);
        }

        // 4. Separable 2D Gaussian Blur (radius = 2 -> 5-tap)
        {
            const size_t n = static_cast<size_t>(d.w) * d.h;
            const auto src = make_pattern(n, 1.0f, 0.0f);
            const float weights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
            std::vector<float> ref(n, 0.0f);
            std::vector<float> jit(n, 0.0f);
            scalar_ref_separable_blur(src.data(), ref.data(), d.w, d.h, weights, 2);
            CHECK(broimage::mir_separable_blur_f32(src.data(), jit.data(), d.w, d.h, weights, 2, false));
            const float diff = max_abs_diff(jit, ref);
            CHECK_LE(diff, 1e-5f);
        }

        // 5. Bilinear Resize
        {
            const int dw = d.w + 9;
            const int dh = d.h + 7;
            const auto src = make_pattern(static_cast<size_t>(d.w) * d.h, 1.0f, 0.0f);
            std::vector<float> ref(static_cast<size_t>(dw) * dh, 0.0f);
            std::vector<float> jit(static_cast<size_t>(dw) * dh, 0.0f);
            broimage::resize_hwc_f32(src.data(), d.w, d.h, 1, ref.data(), dw, dh, broimage::Filter::Bilinear);
            CHECK(broimage::mir_bilinear_resize_f32(src.data(), d.w, d.h, jit.data(), dw, dh, false));
            const float diff = max_abs_diff(jit, ref);
            CHECK_LE(diff, 1e-5f);
        }
    }
    std::printf("  CPU SIMD parity passed across all 5 kernel families and 4 image sizes (<= 1e-5)\n");
}

void test_cpu_simd_vectorization_verified() {
    std::printf("--- Test: CPU MIR Vectorization Pass ---\n");
    using namespace broimage::jit;
    auto check_vectorized = [](MirKernelKind kind, const char* label) {
        brass::Module mod(label);
        brass::Function* fn = build_mir_image_kernel(mod, kind, MirExecTarget::CpuSimd, label);
        CHECK(fn != nullptr);
        if (!fn) return;
        brass::codegen::KernelOptions opts;
        opts.opt_level = 2;
        opts.enable_vectorize = true;
        opts.enable_slp = true;
        opts.enable_avx2 = true;
        opts.enable_fma = true;
        opts.enable_fp_reassociation = true;
        brass::codegen::KernelJit::prepare_module(mod, opts);
        bool saw_vector = false;
        for (const brass::BasicBlock* bb : fn->blocks()) {
            for (const brass::Instruction* inst : *bb) {
                if (inst->type().is_vector()) {
                    saw_vector = true;
                    break;
                }
            }
            if (saw_vector) break;
        }
        std::printf("  %s vector IR emitted: %s\n", label, saw_vector ? "yes" : "no");
        CHECK(saw_vector);
    };
    check_vectorized(MirKernelKind::AffineClampF32, "vec_affine_clamp");
    check_vectorized(MirKernelKind::ChwNormalizeF32, "vec_chw_normalize");
}

void test_gpu_portability_and_codegen() {
    std::printf("--- Test: Single-Source GPU MIR Portability + SPIR-V / PTX / MSL ---\n");
    using namespace broimage::jit;

    const struct {
        MirKernelKind kind;
        const char* name;
    } cases[] = {
        {MirKernelKind::AffineClampF32, "affine_clamp_f32"},
        {MirKernelKind::RgbToGrayF32, "rgb_to_gray_f32"},
        {MirKernelKind::ChwNormalizeF32, "chw_normalize_f32"},
        {MirKernelKind::SeparableBlurHF32, "separable_blur_h_f32"},
        {MirKernelKind::SeparableBlurVF32, "separable_blur_v_f32"},
        {MirKernelKind::BilinearResizeF32, "bilinear_resize_f32"},
    };

    for (const auto& c : cases) {
        brass::Module mod(c.name);
        brass::Function* fn = build_mir_image_kernel(mod, c.kind, MirExecTarget::Gpu, c.name);
        CHECK(fn != nullptr);
        if (!fn) continue;

        brass::DiagnosticReporter mir_diag;
        CHECK(brass::verify_module(mod, &mir_diag));

        CompiledGpuArtifacts art;
        const bool ok = compile_mir_kernel_gpu_targets(*fn, art, 64);
        if (!art.portability_ok) {
            std::fprintf(stderr, "  Portability failure for %s:\n%s\n",
                         c.name, art.portability_diagnostics.c_str());
        }
        CHECK(ok);
        CHECK(art.portability_ok);
        CHECK(!art.spirv.words.empty());
        CHECK(art.spirv.words[0] == 0x07230203u); // SPIR-V magic number
        CHECK(!art.ptx.empty());
        CHECK(!art.msl.source.empty());
        CHECK(art.msl.source.find(c.name) != std::string::npos);

        std::printf("  %-22s: portable=ok, spirv=%zu words, ptx=%zu B, msl=%zu B\n",
                    c.name, art.spirv.words.size(), art.ptx.size(), art.msl.source.size());
    }
}

void test_gpu_vulkan_parity() {
    std::printf("--- Test: GPU Vulkan SPIR-V Execution Parity ---\n");
    using broimage::jit::MirImageKernelCompiler;
    if (!MirImageKernelCompiler::is_gpu_vulkan_available()) {
        std::printf("  [SKIP] Vulkan GPU unavailable (%s)\n",
                    brass::gpu::vulkan_last_error().c_str());
        return;
    }

    const brass::gpu::VulkanDeviceCaps* caps = brass::gpu::vulkan_device_caps();
    if (caps) {
        std::printf("  [vulkan] device=%s (%s), driver=%s\n",
                    caps->info.name.c_str(), caps->info.type.c_str(), caps->driver_name.c_str());
    }

    struct Dim { int w; int h; };
    const Dim dims[] = {{17, 13}, {64, 48}, {129, 33}};

    for (const Dim d : dims) {
        // 1. AffineClamp
        {
            const int ch = 3;
            const size_t n = static_cast<size_t>(d.w) * d.h * ch;
            const auto src = make_pattern(n, 2.0f, -0.5f);
            std::vector<float> cpu(n, 0.0f), gpu(n, 0.0f);
            CHECK(broimage::mir_affine_clamp_f32(src.data(), cpu.data(), d.w, d.h, ch, 1.35f, -0.15f, 0.0f, 1.0f, false));
            CHECK(broimage::mir_affine_clamp_f32(src.data(), gpu.data(), d.w, d.h, ch, 1.35f, -0.15f, 0.0f, 1.0f, true));
            const float diff = max_abs_diff(gpu, cpu);
            std::printf("  [vulkan] AffineClamp %dx%dx%d max_diff=%.8g\n", d.w, d.h, ch, static_cast<double>(diff));
            CHECK_LE(diff, 1e-4f);
        }

        // 2. RgbToGray
        {
            const size_t pix = static_cast<size_t>(d.w) * d.h;
            const auto src = make_pattern(pix * 3, 1.0f, 0.0f);
            std::vector<float> cpu(pix, 0.0f), gpu(pix, 0.0f);
            CHECK(broimage::mir_rgb_to_gray_f32(src.data(), cpu.data(), d.w, d.h, false));
            CHECK(broimage::mir_rgb_to_gray_f32(src.data(), gpu.data(), d.w, d.h, true));
            const float diff = max_abs_diff(gpu, cpu);
            std::printf("  [vulkan] RgbToGray   %dx%d   max_diff=%.8g\n", d.w, d.h, static_cast<double>(diff));
            CHECK_LE(diff, 1e-4f);
        }

        // 3. ChwNormalize
        {
            const size_t n = static_cast<size_t>(3) * d.w * d.h;
            const auto src = make_pattern(n, 1.0f, 0.0f);
            const float mean[3] = {0.485f, 0.456f, 0.406f};
            const float std_dev[3] = {0.229f, 0.224f, 0.225f};
            std::vector<float> cpu(n, 0.0f), gpu(n, 0.0f);
            CHECK(broimage::mir_chw_normalize_f32(src.data(), cpu.data(), d.w, d.h, mean, std_dev, false));
            CHECK(broimage::mir_chw_normalize_f32(src.data(), gpu.data(), d.w, d.h, mean, std_dev, true));
            const float diff = max_abs_diff(gpu, cpu);
            std::printf("  [vulkan] ChwNorm     %dx%d   max_diff=%.8g\n", d.w, d.h, static_cast<double>(diff));
            CHECK_LE(diff, 1e-4f);
        }

        // 4. SeparableBlur
        {
            const size_t n = static_cast<size_t>(d.w) * d.h;
            const auto src = make_pattern(n, 1.0f, 0.0f);
            const float weights[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
            std::vector<float> cpu(n, 0.0f), gpu(n, 0.0f), ref(n, 0.0f);
            scalar_ref_separable_blur(src.data(), ref.data(), d.w, d.h, weights, 2);
            CHECK(broimage::mir_separable_blur_f32(src.data(), cpu.data(), d.w, d.h, weights, 2, false));
            CHECK(broimage::mir_separable_blur_f32(src.data(), gpu.data(), d.w, d.h, weights, 2, true));
            const float diff = max_abs_diff(gpu, cpu);
            const float diff_ref = max_abs_diff(gpu, ref);
            std::printf("  [vulkan] SepBlur2D   %dx%d   max_diff=%.8g (gpu_vs_ref=%.8g)\n",
                        d.w, d.h, static_cast<double>(diff), static_cast<double>(diff_ref));
            CHECK_LE(diff, 1e-4f);
        }

        // 5. BilinearResize
        {
            const int dw = d.w + 9, dh = d.h + 7;
            const auto src = make_pattern(static_cast<size_t>(d.w) * d.h, 1.0f, 0.0f);
            std::vector<float> cpu(static_cast<size_t>(dw) * dh, 0.0f);
            std::vector<float> gpu(static_cast<size_t>(dw) * dh, 0.0f);
            CHECK(broimage::mir_bilinear_resize_f32(src.data(), d.w, d.h, cpu.data(), dw, dh, false));
            CHECK(broimage::mir_bilinear_resize_f32(src.data(), d.w, d.h, gpu.data(), dw, dh, true));
            const float diff = max_abs_diff(gpu, cpu);
            std::printf("  [vulkan] Bilinear    %dx%d->%dx%d max_diff=%.8g\n",
                        d.w, d.h, dw, dh, static_cast<double>(diff));
            CHECK_LE(diff, 1e-4f);
        }
    }
}

void test_error_handling_and_caching() {
    std::printf("--- Test: Error Handling & Kernel Cache ---\n");
    float buf[16] = {};
    const float weights[3] = {0.25f, 0.5f, 0.25f};
    const float mean[3] = {0.0f, 0.0f, 0.0f};
    const float std_dev[3] = {1.0f, 1.0f, 1.0f};

    CHECK(!broimage::mir_affine_clamp_f32(nullptr, buf, 4, 4, 1, 1.0f, 0.0f, 0.0f, 1.0f));
    CHECK(!broimage::mir_affine_clamp_f32(buf, nullptr, 4, 4, 1, 1.0f, 0.0f, 0.0f, 1.0f));
    CHECK(!broimage::mir_affine_clamp_f32(buf, buf, 0, 4, 1, 1.0f, 0.0f, 0.0f, 1.0f));
    CHECK(!broimage::mir_rgb_to_gray_f32(nullptr, buf, 2, 2));
    CHECK(!broimage::mir_chw_normalize_f32(buf, buf, -1, 2, mean, std_dev));
    CHECK(!broimage::mir_separable_blur_f32(buf, buf, 4, 4, nullptr, 1));
    CHECK(!broimage::mir_separable_blur_f32(buf, buf, 4, 4, weights, -1));
    CHECK(!broimage::mir_bilinear_resize_f32(buf, 4, 4, buf, 0, 4));

    auto& c = broimage::jit::MirImageKernelCompiler::instance();
    if (broimage::jit::MirImageKernelCompiler::is_cpu_supported()) {
        CHECK(c.cpu_cache_size() >= 6u);
    }
    if (broimage::jit::MirImageKernelCompiler::is_gpu_vulkan_available()) {
        CHECK(c.gpu_cache_size() >= 6u);
    }
}

} // namespace

int main() {
    std::printf("Running broimage Single-Source MIR Image Kernels test suite...\n");

    test_cpu_simd_parity();
    test_cpu_simd_vectorization_verified();
    test_gpu_portability_and_codegen();
    test_gpu_vulkan_parity();
    test_error_handling_and_caching();

    if (g_failed == 0) {
        std::printf("ALL MIR IMAGE KERNEL TESTS PASSED.\n");
        return 0;
    }
    std::fprintf(stderr, "%d TEST(S) FAILED.\n", g_failed);
    return 1;
}
