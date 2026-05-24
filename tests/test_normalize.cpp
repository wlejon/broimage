#include "broimage/normalize.h"
#include "broimage/presets.h"

#include <cmath>
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
    // 1 image, 3 channels, 2x2. Channel c: every pixel = c+1.
    const int N = 1, C = 3, H = 2, W = 2;
    const int spatial = H * W;
    std::vector<float> X(N * C * spatial);
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < spatial; ++i) X[c * spatial + i] = static_cast<float>(c + 1);
    }
    const float mean[3] = { 1.0f, 2.0f, 3.0f };
    const float std_[3] = { 2.0f, 4.0f, 8.0f };
    std::vector<float> Y(X.size());
    broimage::image_normalize_nchw_f32(X.data(), mean, std_, N, C, H, W, Y.data());
    // Channel 0: (1-1)/2 = 0; Channel 1: (2-2)/4 = 0; Channel 2: (3-3)/8 = 0.
    for (float v : Y) CHECK(nearf(v, 0.0f));

    // Offset values to confirm scaling: X[c] = c + 1 + std[c], Y[c] = 1.
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < spatial; ++i) {
            X[c * spatial + i] = static_cast<float>(c + 1) + std_[c];
        }
    }
    broimage::image_normalize_nchw_f32(X.data(), mean, std_, N, C, H, W, Y.data());
    for (float v : Y) CHECK(nearf(v, 1.0f));

    // Aliasing: X == Y in-place.
    std::vector<float> XY(N * C * spatial);
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < spatial; ++i) XY[c * spatial + i] = static_cast<float>(c + 1);
    }
    broimage::image_normalize_nchw_f32(XY.data(), mean, std_, N, C, H, W, XY.data());
    for (float v : XY) CHECK(nearf(v, 0.0f));

    // Presets are non-empty and sensible.
    CHECK(broimage::CLIP_MEAN[0] > 0.0f && broimage::CLIP_MEAN[0] < 1.0f);
    CHECK(broimage::IMAGENET_STD[0] > 0.0f);
    CHECK(broimage::SAM_MEAN[0] == broimage::IMAGENET_MEAN[0]);

    return g_failed == 0 ? 0 : 1;
}
