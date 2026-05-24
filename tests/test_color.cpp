#include "broimage/color.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

static bool nearf(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    // RGBA <-> RGB.
    const uint8_t rgba[8] = { 10, 20, 30, 255,  40, 50, 60, 128 };
    uint8_t rgb[6];
    broimage::rgba_to_rgb_u8(rgba, rgb, 2);
    CHECK(rgb[0] == 10 && rgb[1] == 20 && rgb[2] == 30);
    CHECK(rgb[3] == 40 && rgb[4] == 50 && rgb[5] == 60);

    uint8_t back[8];
    broimage::rgb_to_rgba_u8(rgb, back, 2, 200);
    CHECK(back[0] == 10 && back[3] == 200);
    CHECK(back[4] == 40 && back[7] == 200);

    // RGBA -> gray (Rec.601).
    const uint8_t white[4] = { 255, 255, 255, 255 };
    uint8_t gw = 0;
    broimage::rgba_to_gray_u8(white, &gw, 1);
    CHECK(gw >= 250);  // ~(77+150+29)/256 * 255 = 255

    const uint8_t pure_g[4] = { 0, 255, 0, 255 };
    uint8_t gg = 0;
    broimage::rgba_to_gray_u8(pure_g, &gg, 1);
    CHECK(gg >= 140 && gg <= 160);  // 150/256 * 255 ≈ 149

    // HWC <-> CHW.
    // 2x2 RGB image, interleaved: [R0 G0 B0 R1 G1 B1 R2 G2 B2 R3 G3 B3]
    const float hwc[12] = { 1, 2, 3,  4, 5, 6,  7, 8, 9,  10, 11, 12 };
    float chw[12];
    broimage::hwc_to_chw_f32(hwc, chw, 2, 2, 3);
    // R plane: 1, 4, 7, 10
    CHECK(nearf(chw[0], 1)  && nearf(chw[1], 4)  && nearf(chw[2], 7)  && nearf(chw[3], 10));
    // G plane: 2, 5, 8, 11
    CHECK(nearf(chw[4], 2)  && nearf(chw[5], 5)  && nearf(chw[6], 8)  && nearf(chw[7], 11));
    // B plane: 3, 6, 9, 12
    CHECK(nearf(chw[8], 3)  && nearf(chw[9], 6)  && nearf(chw[10], 9) && nearf(chw[11], 12));

    float back_hwc[12];
    broimage::chw_to_hwc_f32(chw, back_hwc, 2, 2, 3);
    for (int i = 0; i < 12; ++i) CHECK(nearf(back_hwc[i], hwc[i]));

    // sRGB <-> linear round trip.
    const float v_srgb[5] = { 0.0f, 0.04f, 0.5f, 0.8f, 1.0f };
    float v_lin[5], back_srgb[5];
    broimage::srgb_to_linear_f32(v_srgb, v_lin, 5);
    broimage::linear_to_srgb_f32(v_lin, back_srgb, 5);
    for (int i = 0; i < 5; ++i) CHECK(nearf(v_srgb[i], back_srgb[i], 1e-5f));

    // sRGB(0) == 0, sRGB(1) ≈ 1.
    CHECK(nearf(v_lin[0], 0.0f));
    CHECK(nearf(v_lin[4], 1.0f, 1e-4f));

    // ----- HSV / HSL ---------------------------------------------------------
    // Known conversions: pure red (1, 0, 0) -> H=0,    S=1, V=1
    //                    pure green       -> H=1/3,  S=1, V=1
    //                    pure blue        -> H=2/3,  S=1, V=1
    const float rgb_test[9] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };
    float hsv_out[9];
    broimage::rgb_to_hsv_f32(rgb_test, hsv_out, 3);
    CHECK(nearf(hsv_out[0], 0.0f));        CHECK(nearf(hsv_out[1], 1.0f)); CHECK(nearf(hsv_out[2], 1.0f));
    CHECK(nearf(hsv_out[3], 1.0f / 3.0f)); CHECK(nearf(hsv_out[4], 1.0f)); CHECK(nearf(hsv_out[5], 1.0f));
    CHECK(nearf(hsv_out[6], 2.0f / 3.0f)); CHECK(nearf(hsv_out[7], 1.0f)); CHECK(nearf(hsv_out[8], 1.0f));

    // RGB -> HSV -> RGB round trip over a small grid.
    float trip[12] = { 0.1f, 0.5f, 0.9f,  0.7f, 0.2f, 0.4f,
                       0.5f, 0.5f, 0.5f,  0.0f, 0.0f, 0.0f };
    float trip_hsv[12], trip_back[12];
    broimage::rgb_to_hsv_f32(trip, trip_hsv, 4);
    broimage::hsv_to_rgb_f32(trip_hsv, trip_back, 4);
    for (int i = 0; i < 12; ++i) CHECK(nearf(trip[i], trip_back[i], 1e-4f));

    // RGB -> HSL -> RGB round trip.
    float trip_hsl[12], trip_hsl_back[12];
    broimage::rgb_to_hsl_f32(trip, trip_hsl, 4);
    broimage::hsl_to_rgb_f32(trip_hsl, trip_hsl_back, 4);
    for (int i = 0; i < 12; ++i) CHECK(nearf(trip[i], trip_hsl_back[i], 1e-4f));

    // HSL lightness of grey 0.5 = 0.5, saturation = 0.
    float grey[3] = { 0.5f, 0.5f, 0.5f };
    float grey_hsl[3];
    broimage::rgb_to_hsl_f32(grey, grey_hsl, 1);
    CHECK(nearf(grey_hsl[1], 0.0f));
    CHECK(nearf(grey_hsl[2], 0.5f));

    // ----- Color matrix ------------------------------------------------------
    // 3x3 identity preserves pixels.
    const float I[9] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };
    float mid[12] = { 0.2f, 0.4f, 0.6f, 0.8f,
                      0.3f, 0.5f, 0.7f, 0.9f,
                      0.1f, 0.2f, 0.3f, 0.4f };
    float midout[12];
    broimage::apply_color_matrix_3x3_f32(mid, midout, 3, 4, I);
    for (int i = 0; i < 12; ++i) CHECK(nearf(midout[i], mid[i]));

    // Channel swap RGB -> BGR via 3x3, alpha unchanged.
    const float swap[9] = { 0, 0, 1,  0, 1, 0,  1, 0, 0 };
    broimage::apply_color_matrix_3x3_f32(mid, midout, 3, 4, swap);
    CHECK(nearf(midout[0], 0.6f) && nearf(midout[1], 0.4f) && nearf(midout[2], 0.2f) && nearf(midout[3], 0.8f));

    // 3x4 luma-with-bias: take just the green channel and shift by 0.1.
    const float luma_bias[12] = {
        0, 1, 0, 0.1f,
        0, 1, 0, 0.1f,
        0, 1, 0, 0.1f,
    };
    broimage::apply_color_matrix_3x4_f32(mid, midout, 3, 4, luma_bias);
    for (int i = 0; i < 3; ++i) {
        CHECK(nearf(midout[i * 4 + 0], mid[i * 4 + 1] + 0.1f));
        CHECK(nearf(midout[i * 4 + 1], mid[i * 4 + 1] + 0.1f));
        CHECK(nearf(midout[i * 4 + 2], mid[i * 4 + 1] + 0.1f));
        CHECK(nearf(midout[i * 4 + 3], mid[i * 4 + 3])); // alpha pass-through
    }

    // 3 channel path: no alpha to preserve.
    const float rgb_only[9] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };
    float rgb_out[9];
    broimage::apply_color_matrix_3x3_f32(rgb_only, rgb_out, 3, 3, swap);
    CHECK(nearf(rgb_out[0], 0) && nearf(rgb_out[1], 0) && nearf(rgb_out[2], 1));
    CHECK(nearf(rgb_out[3], 0) && nearf(rgb_out[4], 1) && nearf(rgb_out[5], 0));
    CHECK(nearf(rgb_out[6], 1) && nearf(rgb_out[7], 0) && nearf(rgb_out[8], 0));

    // In-place aliasing.
    float alias[4] = { 0.2f, 0.4f, 0.6f, 0.8f };
    broimage::apply_color_matrix_3x3_f32(alias, alias, 1, 4, swap);
    CHECK(nearf(alias[0], 0.6f) && nearf(alias[1], 0.4f) && nearf(alias[2], 0.2f) && nearf(alias[3], 0.8f));

    return g_failed == 0 ? 0 : 1;
}
