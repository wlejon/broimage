#include "broimage/alpha.h"
#include "broimage/geometric.h"
#include "broimage/normalize.h"
#include "broimage/preproc.h"
#include "broimage/preproc_jit.h"
#include "broimage/jit/image_pipeline.h"
#include "broimage/jit/jit_image_compiler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

#define CHECK_LE(a, b) do { \
    if (!((a) <= (b))) { \
        std::fprintf(stderr, "FAIL %s:%d: %s <= %s (%f <= %f)\n", __FILE__, __LINE__, #a, #b, static_cast<double>(a), static_cast<double>(b)); \
        g_failed++; \
    } \
} while (0)

#define CHECK_LT(a, b) do { \
    if (!((a) < (b))) { \
        std::fprintf(stderr, "FAIL %s:%d: %s < %s (%f < %f)\n", __FILE__, __LINE__, #a, #b, static_cast<double>(a), static_cast<double>(b)); \
        g_failed++; \
    } \
} while (0)

namespace {

void scalar_ref_resize_rgba8_alpha(const uint8_t* src, int src_w, int src_h,
                                   uint8_t* dst, int dst_w, int dst_h) {
    const std::size_t src_n = static_cast<std::size_t>(src_w) * src_h;
    const std::size_t dst_n = static_cast<std::size_t>(dst_w) * dst_h;
    std::vector<uint8_t> prem_src(src_n * 4);
    broimage::premultiply_alpha_rgba8(src, prem_src.data(), static_cast<int>(src_n));
    std::vector<uint8_t> prem_dst(dst_n * 4);
    broimage::resize_hwc_u8(prem_src.data(), src_w, src_h, 4,
                            prem_dst.data(), dst_w, dst_h, broimage::Filter::Bilinear);
    broimage::unpremultiply_alpha_rgba8(prem_dst.data(), dst, static_cast<int>(dst_n));
}

void scalar_ref_fused_preproc(const uint8_t* src, int src_w, int src_h,
                              float* dst_planar, int dst_w, int dst_h,
                              const float* mean, const float* std_dev) {
    std::vector<uint8_t> resized(static_cast<std::size_t>(dst_w) * dst_h * 3);
    broimage::resize_hwc_u8(src, src_w, src_h, 3, resized.data(), dst_w, dst_h, broimage::Filter::Bilinear);
    broimage::u8_nhwc_to_f32_nchw(resized.data(), 1, dst_h, dst_w, 3, 1.0f / 255.0f, 0.0f, dst_planar);
    broimage::image_normalize_nchw_f32(dst_planar, mean, std_dev, 1, 3, dst_h, dst_w, dst_planar);
}

} // namespace

static void test_pipeline_desc_hash() {
    std::printf("--- Test: Pipeline Descriptor Hash ---\n");
    broimage::jit::ImagePipelineDesc desc1;
    desc1.src_w = 64;
    desc1.src_h = 48;
    desc1.dst_w = 32;
    desc1.dst_h = 24;
    desc1.premultiply_alpha = true;
    desc1.to_float_planar = false;
    desc1.mean = {0.485f, 0.456f, 0.406f};
    desc1.std_dev = {0.229f, 0.224f, 0.225f};

    broimage::jit::ImagePipelineDesc desc2 = desc1;
    CHECK(desc1.compute_hash() == desc2.compute_hash());
    CHECK(desc1 == desc2);

    // Modify each property and verify hash changes
    desc2.src_w = 128;
    CHECK(desc1.compute_hash() != desc2.compute_hash());

    desc2 = desc1;
    desc2.premultiply_alpha = false;
    CHECK(desc1.compute_hash() != desc2.compute_hash());

    desc2 = desc1;
    desc2.to_float_planar = true;
    CHECK(desc1.compute_hash() != desc2.compute_hash());

    desc2 = desc1;
    desc2.mean[0] = 0.5f;
    CHECK(desc1.compute_hash() != desc2.compute_hash());

    desc2 = desc1;
    desc2.std_dev[1] = 0.5f;
    CHECK(desc1.compute_hash() != desc2.compute_hash());
}

static void test_jit_rgba8_resize() {
    std::printf("--- Test: JIT RGBA8 Resize vs Reference ---\n");
    if (!broimage::jit::JitImageCompiler::is_supported()) {
        std::printf("  [SKIP] JIT unsupported on current host architecture\n");
        return;
    }

    auto& compiler = broimage::jit::JitImageCompiler::instance();

    // 1. Downscaling test with premultiplied alpha
    {
        const int src_w = 96, src_h = 72;
        const int dst_w = 48, dst_h = 36;
        std::vector<uint8_t> src(static_cast<std::size_t>(src_w * src_h * 4));
        for (int y = 0; y < src_h; ++y) {
            for (int x = 0; x < src_w; ++x) {
                int idx = (y * src_w + x) * 4;
                src[idx + 0] = static_cast<uint8_t>((x * 255) / (src_w - 1));
                src[idx + 1] = static_cast<uint8_t>((y * 255) / (src_h - 1));
                src[idx + 2] = static_cast<uint8_t>(255 - src[idx + 0]);
                src[idx + 3] = static_cast<uint8_t>((x + y) % 256);
            }
        }

        std::vector<uint8_t> dst_ref(static_cast<std::size_t>(dst_w * dst_h * 4), 0);
        std::vector<uint8_t> dst_jit(static_cast<std::size_t>(dst_w * dst_h * 4), 0);

        scalar_ref_resize_rgba8_alpha(src.data(), src_w, src_h, dst_ref.data(), dst_w, dst_h);

        bool ok = compiler.execute_resize_rgba8(src.data(), dst_jit.data(), src_w, src_h, dst_w, dst_h, true);
        CHECK(ok);

        int max_diff = 0;
        for (std::size_t i = 0; i < dst_jit.size(); ++i) {
            int diff = std::abs(static_cast<int>(dst_jit[i]) - static_cast<int>(dst_ref[i]));
            if (diff > max_diff) max_diff = diff;
        }
        std::printf("  Downscale 96x72 -> 48x36 max pixel diff: %d LSB\n", max_diff);
        CHECK_LE(max_diff, 1);
    }

    // 2. Upscaling test with premultiplied alpha
    {
        const int src_w = 16, src_h = 16;
        const int dst_w = 32, dst_h = 32;
        std::vector<uint8_t> src(static_cast<std::size_t>(src_w * src_h * 4));
        for (int y = 0; y < src_h; ++y) {
            for (int x = 0; x < src_w; ++x) {
                int idx = (y * src_w + x) * 4;
                src[idx + 0] = static_cast<uint8_t>((x * 255) / (src_w - 1));
                src[idx + 1] = static_cast<uint8_t>((y * 255) / (src_h - 1));
                src[idx + 2] = static_cast<uint8_t>(128);
                src[idx + 3] = static_cast<uint8_t>(200);
            }
        }

        std::vector<uint8_t> dst_ref(static_cast<std::size_t>(dst_w * dst_h * 4), 0);
        std::vector<uint8_t> dst_jit(static_cast<std::size_t>(dst_w * dst_h * 4), 0);

        scalar_ref_resize_rgba8_alpha(src.data(), src_w, src_h, dst_ref.data(), dst_w, dst_h);

        bool ok = compiler.execute_resize_rgba8(src.data(), dst_jit.data(), src_w, src_h, dst_w, dst_h, true);
        CHECK(ok);

        int max_diff = 0;
        for (std::size_t i = 0; i < dst_jit.size(); ++i) {
            int diff = std::abs(static_cast<int>(dst_jit[i]) - static_cast<int>(dst_ref[i]));
            if (diff > max_diff) max_diff = diff;
        }
        std::printf("  Upscale 16x16 -> 32x32 max pixel diff: %d LSB\n", max_diff);
        CHECK_LE(max_diff, 1);
    }

    // 3. Test hook in broimage::resize_rgba8_alpha with Filter::Bilinear
    {
        const int src_w = 32, src_h = 32;
        const int dst_w = 16, dst_h = 16;
        std::vector<uint8_t> src(static_cast<std::size_t>(src_w * src_h * 4));
        for (int i = 0; i < src_w * src_h * 4; ++i) {
            src[i] = static_cast<uint8_t>((i * 17) % 256);
        }

        std::vector<uint8_t> dst_hook(static_cast<std::size_t>(dst_w * dst_h * 4), 0);
        std::vector<uint8_t> dst_ref(static_cast<std::size_t>(dst_w * dst_h * 4), 0);

        scalar_ref_resize_rgba8_alpha(src.data(), src_w, src_h, dst_ref.data(), dst_w, dst_h);
        broimage::resize_rgba8_alpha(src.data(), src_w, src_h, dst_hook.data(), dst_w, dst_h, broimage::Filter::Bilinear);

        int max_diff = 0;
        for (std::size_t i = 0; i < dst_hook.size(); ++i) {
            int diff = std::abs(static_cast<int>(dst_hook[i]) - static_cast<int>(dst_ref[i]));
            if (diff > max_diff) max_diff = diff;
        }
        std::printf("  resize_rgba8_alpha hook max pixel diff: %d LSB\n", max_diff);
        CHECK_LE(max_diff, 1);
    }
}

static void test_jit_fused_preproc() {
    std::printf("--- Test: JIT Fused Preproc vs Scalar Pipeline ---\n");
    if (!broimage::jit::JitImageCompiler::is_supported()) {
        std::printf("  [SKIP] JIT unsupported on current host architecture\n");
        return;
    }

    const int src_w = 64, src_h = 48;
    const int dst_w = 32, dst_h = 24;
    const int channels = 3;

    std::vector<uint8_t> src(static_cast<std::size_t>(src_w * src_h * channels));
    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
            int idx = (y * src_w + x) * channels;
            src[idx + 0] = static_cast<uint8_t>((x * 255) / (src_w - 1));
            src[idx + 1] = static_cast<uint8_t>((y * 255) / (src_h - 1));
            src[idx + 2] = static_cast<uint8_t>(((x + y) * 255) / (src_w + src_h - 2));
        }
    }

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_dev[3] = {0.229f, 0.224f, 0.225f};

    const std::size_t planar_size = static_cast<std::size_t>(channels * dst_w * dst_h);
    std::vector<float> dst_ref(planar_size, 0.0f);
    std::vector<float> dst_jit(planar_size, 0.0f);

    scalar_ref_fused_preproc(src.data(), src_w, src_h, dst_ref.data(), dst_w, dst_h, mean, std_dev);

    // Call through public function broimage::fused_u8_nhwc_to_f32_nchw_normalized
    broimage::fused_u8_nhwc_to_f32_nchw_normalized(src.data(), src_w, src_h, dst_jit.data(), dst_w, dst_h, mean, std_dev);

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < planar_size; ++i) {
        float diff = std::fabs(dst_jit[i] - dst_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    std::printf("  fused preproc max float diff: %f (threshold: 0.01)\n", static_cast<double>(max_diff));
    CHECK_LT(max_diff, 0.01f);
}

static void test_error_handling_and_cache() {
    std::printf("--- Test: Error Handling & Kernel Cache ---\n");
    auto& compiler = broimage::jit::JitImageCompiler::instance();

    // Invalid arguments
    uint8_t dummy_src[16] = {0};
    uint8_t dummy_dst[16] = {0};
    float dummy_f[16] = {0};

    CHECK(!compiler.execute_resize_rgba8(nullptr, dummy_dst, 4, 4, 2, 2, true));
    CHECK(!compiler.execute_resize_rgba8(dummy_src, nullptr, 4, 4, 2, 2, true));
    CHECK(!compiler.execute_resize_rgba8(dummy_src, dummy_dst, 0, 4, 2, 2, true));
    CHECK(!compiler.execute_resize_rgba8(dummy_src, dummy_dst, 4, 4, -2, 2, true));

    CHECK(!compiler.execute_fused_preproc(nullptr, dummy_f, 4, 4, 2, 2, nullptr, nullptr));
    CHECK(!compiler.execute_fused_preproc(dummy_src, nullptr, 4, 4, 2, 2, nullptr, nullptr));
    CHECK(!compiler.execute_fused_preproc(dummy_src, dummy_f, 4, 0, 2, 2, nullptr, nullptr));

    if (broimage::jit::JitImageCompiler::is_supported()) {
        size_t initial_cache = compiler.cache_size();
        CHECK(compiler.execute_resize_rgba8(dummy_src, dummy_dst, 4, 4, 2, 2, true));
        size_t cache_after_first = compiler.cache_size();
        CHECK(cache_after_first >= initial_cache);

        // Second call with same descriptor should hit cache without increasing cache size
        CHECK(compiler.execute_resize_rgba8(dummy_src, dummy_dst, 4, 4, 2, 2, true));
        CHECK(compiler.cache_size() == cache_after_first);
    }
}

static void run_benchmarks() {
    std::printf("--- Benchmark: JIT Throughput ---\n");
    if (!broimage::jit::JitImageCompiler::is_supported()) {
        std::printf("  [SKIP] JIT unsupported on current host architecture\n");
        return;
    }

    auto& compiler = broimage::jit::JitImageCompiler::instance();

    // 1. 4K (3840x2160) -> 1080p (1920x1080) RGBA8 resize
    {
        const int src_w = 3840, src_h = 2160;
        const int dst_w = 1920, dst_h = 1080;
        std::vector<uint8_t> src(static_cast<std::size_t>(src_w * src_h * 4), 180);
        std::vector<uint8_t> dst(static_cast<std::size_t>(dst_w * dst_h * 4), 0);

        // Warmup
        compiler.execute_resize_rgba8(src.data(), dst.data(), src_w, src_h, dst_w, dst_h, true);

        const int iterations = 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            compiler.execute_resize_rgba8(src.data(), dst.data(), src_w, src_h, dst_w, dst_h, true);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
        double src_mpixels = (static_cast<double>(src_w) * src_h) / 1e6;
        double throughput = src_mpixels / (elapsed_ms / 1000.0);

        std::printf("  [BENCHMARK] 4K (3840x2160) -> 1080p (1920x1080) RGBA8 Resize:\n"
                    "    Execution time: %.2f ms | Throughput: %.2f MPixels/sec\n",
                    elapsed_ms, throughput);
    }

    // 2. 4K (3840x2160) -> 512x512 Fused Preprocessing
    {
        const int src_w = 3840, src_h = 2160;
        const int dst_w = 512, dst_h = 512;
        std::vector<uint8_t> src(static_cast<std::size_t>(src_w * src_h * 3), 128);
        std::vector<float> dst_planar(static_cast<std::size_t>(dst_w * dst_h * 3), 0.0f);
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std_dev[3] = {0.229f, 0.224f, 0.225f};

        // Warmup
        broimage::fused_u8_nhwc_to_f32_nchw_normalized(src.data(), src_w, src_h, dst_planar.data(), dst_w, dst_h, mean, std_dev);

        const int iterations = 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            broimage::fused_u8_nhwc_to_f32_nchw_normalized(src.data(), src_w, src_h, dst_planar.data(), dst_w, dst_h, mean, std_dev);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
        double src_mpixels = (static_cast<double>(src_w) * src_h) / 1e6;
        double throughput = src_mpixels / (elapsed_ms / 1000.0);

        std::printf("  [BENCHMARK] 4K (3840x2160) -> 512x512 Fused Preproc:\n"
                    "    Execution time: %.2f ms | Throughput: %.2f MPixels/sec\n",
                    elapsed_ms, throughput);
    }
}

int main() {
    std::printf("Running broimage JIT test suite...\n");

    test_pipeline_desc_hash();
    test_jit_rgba8_resize();
    test_jit_fused_preproc();
    test_error_handling_and_cache();
    run_benchmarks();

    if (g_failed == 0) {
        std::printf("ALL TESTS PASSED.\n");
        return 0;
    } else {
        std::fprintf(stderr, "%d TEST(S) FAILED.\n", g_failed);
        return 1;
    }
}
