#include "broimage/kernels.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

static bool nearf(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    // ----- gradient ----------------------------------------------------------
    const broimage::GradientStop stops[] = {
        {0.0f,   0,   0,   0, 255},
        {1.0f, 255, 255, 255, 255},
    };
    std::vector<uint8_t> lut;
    broimage::gradient(stops, 2, 256, lut);
    CHECK(lut.size() == 256 * 4);
    CHECK(lut[0] == 0 && lut[3] == 255);
    CHECK(lut[255 * 4 + 0] == 255 && lut[255 * 4 + 3] == 255);
    // Midpoint should be around 127.
    CHECK(lut[128 * 4 + 0] >= 125 && lut[128 * 4 + 0] <= 130);

    // ----- lookup ------------------------------------------------------------
    const float src[4] = { 0.0f, 0.5f, 1.0f, 2.0f };
    uint8_t dst[4 * 4] = {0};
    broimage::lookup_f32(src, 4, lut.data(), 256, dst, 0.0f, 1.0f);
    // src=0 -> idx 0 -> black; src=1 -> idx 255 -> white; src=2 (clamped) -> white.
    CHECK(dst[0] == 0   && dst[3] == 255);
    CHECK(dst[8] == 255 && dst[11] == 255);
    CHECK(dst[12] == 255);

    // Wrap mode.
    uint8_t wrap_dst[4];
    const float over[1] = { 1.5f };  // hi=1, lo=0, t=1.5*255 = 382, wrap -> 382 % 256 = 126
    broimage::lookup_f32(over, 1, lut.data(), 256, wrap_dst, 0.0f, 1.0f,
                         broimage::LookupEdge::Wrap);
    CHECK(wrap_dst[0] > 100 && wrap_dst[0] < 150);

    // ----- reduce ------------------------------------------------------------
    const float field[6] = { -3.0f, 0.0f, 1.0f, 4.0f, 2.5f, -0.5f };
    auto mm = broimage::reduce_minmax_f32(field, 6);
    CHECK(nearf(mm.min, -3.0f));
    CHECK(nearf(mm.max,  4.0f));

    const double sum = broimage::reduce_sum_f32(field, 6);
    CHECK(nearf(static_cast<float>(sum), 4.0f, 1e-4f));
    const double mean = broimage::reduce_mean_f32(field, 6);
    CHECK(nearf(static_cast<float>(mean), 4.0f / 6.0f, 1e-4f));

    // Stride samples [0, 2, 4] = -3, 1, 2.5.
    auto mm_s = broimage::reduce_minmax_f32(field, 6, 2);
    CHECK(nearf(mm_s.min, -3.0f));
    CHECK(nearf(mm_s.max,  2.5f));

    // Histogram: bins=4 over [0, 4]. Float idx truncates toward zero (matches
    // the brokit reference impl). Elements:
    //   -3   -> t=-0.75, idx=-3       (drop, < 0)
    //    0   -> t=0,     idx=0        (bin 0)
    //    1   -> t=0.25,  idx=1        (bin 1)
    //    4   -> t=1,     idx=4        (drop, >= bins)
    //    2.5 -> t=0.625, idx=2        (bin 2)
    //   -0.5 -> t=-0.125, idx=trunc(-0.5)=0  (bin 0 — trunc-toward-zero quirk)
    uint32_t bins[4] = {0};
    broimage::reduce_histogram_f32(field, 6, 4, 0.0f, 4.0f, bins);
    CHECK(bins[0] == 2);
    CHECK(bins[1] == 1);
    CHECK(bins[2] == 1);
    CHECK(bins[3] == 0);

    // ----- map ---------------------------------------------------------------
    float vs[4] = { -2, -1, 0, 1 };
    float out[4];
    broimage::map_abs_f32(vs, out, 4);
    CHECK(nearf(out[0], 2) && nearf(out[1], 1) && nearf(out[2], 0) && nearf(out[3], 1));

    broimage::map_affine_f32(vs, out, 4, 2.0f, 1.0f);
    CHECK(nearf(out[0], -3) && nearf(out[3], 3));

    broimage::map_affine_clamp_f32(vs, out, 4, 1.0f, 0.0f, -1.5f, 0.5f);
    CHECK(nearf(out[0], -1.5f) && nearf(out[3], 0.5f));

    broimage::map_sqrt_f32(vs + 2, out, 2);  // sqrt(0), sqrt(1)
    CHECK(nearf(out[0], 0) && nearf(out[1], 1));

    broimage::map_pow_f32(vs + 2, out, 2, 2.0f);  // 0, 1
    CHECK(nearf(out[0], 0) && nearf(out[1], 1));

    // ----- combine -----------------------------------------------------------
    float a[3] = { 1, 2, 3 };
    float b[3] = { 4, 5, 6 };
    float c[3];
    broimage::combine_add_f32(a, b, c, 3);
    CHECK(nearf(c[0], 5) && nearf(c[2], 9));
    broimage::combine_sub_f32(b, a, c, 3);
    CHECK(nearf(c[0], 3));
    broimage::combine_mul_f32(a, b, c, 3);
    CHECK(nearf(c[2], 18));
    broimage::combine_min_f32(a, b, c, 3);
    CHECK(nearf(c[0], 1));
    broimage::combine_max_f32(a, b, c, 3);
    CHECK(nearf(c[0], 4));
    broimage::combine_lerp_f32(a, b, c, 3, 0.25f);
    CHECK(nearf(c[0], 1.75f));
    broimage::combine_wsum_f32(a, b, c, 3, 2.0f, 0.5f);
    CHECK(nearf(c[0], 4.0f));

    // ----- stencil (3x3 identity = pass-through) ----------------------------
    const float img3[9] = { 1, 2, 3,  4, 5, 6,  7, 8, 9 };
    float so[9];
    const float ident[9] = { 0, 0, 0,  0, 1, 0,  0, 0, 0 };
    broimage::stencil_f32(img3, so, 3, 3, ident, 3, 3);
    for (int i = 0; i < 9; ++i) CHECK(nearf(so[i], img3[i]));

    // 3x3 box blur on a uniform field should give the same uniform field.
    const float uniform[9] = { 7, 7, 7,  7, 7, 7,  7, 7, 7 };
    const float box[9] = { 1, 1, 1,  1, 1, 1,  1, 1, 1 };
    broimage::stencil_f32(uniform, so, 3, 3, box, 3, 3, 9.0f);
    for (int i = 0; i < 9; ++i) CHECK(nearf(so[i], 7.0f));

    // ----- resample ----------------------------------------------------------
    // 2x2 -> 4x4 bilinear (1-channel).
    const float small[4] = { 0, 1, 1, 0 };
    float big[16];
    broimage::resample_f32(small, 2, 2, big, 4, 4, 1,
                           broimage::Filter::Bilinear);
    CHECK(nearf(big[0],  0.0f, 0.05f));
    CHECK(nearf(big[3],  1.0f, 0.05f));
    CHECK(nearf(big[15], 0.0f, 0.05f));

    return g_failed == 0 ? 0 : 1;
}
