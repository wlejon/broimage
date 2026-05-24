#include "broimage/preproc.h"

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
    // 1 image, 2x2, RGB. NHWC layout, packed [R G B R G B R G B R G B].
    const int N = 1, H = 2, W = 2, C = 3;
    const std::vector<uint8_t> src = {
        10, 20, 30,   40, 50, 60,    // row 0
        70, 80, 90,  100, 110, 120,  // row 1
    };
    std::vector<float> Y(N * C * H * W);

    // [0,255] -> [0,1]: scale = 1/255, bias = 0.
    broimage::u8_nhwc_to_f32_nchw(src.data(), N, H, W, C,
                                  1.0f / 255.0f, 0.0f, Y.data());
    // R-plane (c=0): src R values were 10, 40, 70, 100 -> divided by 255.
    CHECK(nearf(Y[0], 10.0f / 255.0f));
    CHECK(nearf(Y[1], 40.0f / 255.0f));
    CHECK(nearf(Y[2], 70.0f / 255.0f));
    CHECK(nearf(Y[3], 100.0f / 255.0f));
    // G-plane (c=1): 20, 50, 80, 110.
    CHECK(nearf(Y[4], 20.0f / 255.0f));
    CHECK(nearf(Y[5], 50.0f / 255.0f));
    CHECK(nearf(Y[7], 110.0f / 255.0f));
    // B-plane (c=2): 30, 60, 90, 120.
    CHECK(nearf(Y[8],  30.0f / 255.0f));
    CHECK(nearf(Y[11], 120.0f / 255.0f));

    // [0,255] -> [-1,1].
    broimage::u8_nhwc_to_f32_nchw(src.data(), N, H, W, C,
                                  2.0f / 255.0f, -1.0f, Y.data());
    CHECK(nearf(Y[0], 10.0f * 2.0f / 255.0f - 1.0f));

    // nhwc <-> nchw round trip on floats.
    std::vector<float> hwc(N * H * W * C);
    for (size_t i = 0; i < src.size(); ++i) hwc[i] = static_cast<float>(src[i]);
    std::vector<float> chw(hwc.size());
    broimage::nhwc_to_nchw_f32(hwc.data(), N, H, W, C, chw.data());
    std::vector<float> back(hwc.size());
    broimage::nchw_to_nhwc_f32(chw.data(), N, C, H, W, back.data());
    for (size_t i = 0; i < hwc.size(); ++i) CHECK(nearf(back[i], hwc[i]));

    return g_failed == 0 ? 0 : 1;
}
