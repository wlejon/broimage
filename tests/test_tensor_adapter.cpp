#include "broimage/tensor_adapter.h"

#include <brotensor/tensor.h>

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
    // image_u8_to_f32_nhwc_to_nchw via the brotensor adapter.
    const int N = 1, H = 2, W = 2, C = 3;
    const std::vector<uint8_t> src = {
        10, 20, 30,   40, 50, 60,
        70, 80, 90,  100, 110, 120,
    };

    brotensor::Tensor Y;
    broimage::image_u8_to_f32_nhwc_to_nchw(src.data(), N, H, W, C,
                                           1.0f / 255.0f, 0.0f, Y);
    CHECK(Y.rows == N);
    CHECK(Y.cols == C * H * W);
    CHECK(Y.dtype == brotensor::Dtype::FP32);
    const float* Yp = Y.host_f32();
    CHECK(nearf(Yp[0], 10.0f / 255.0f));
    CHECK(nearf(Yp[1], 40.0f / 255.0f));

    // image_normalize via adapter. mean/std are (C,1) FP32.
    brotensor::Tensor mean, std_;
    mean.resize(C, 1, brotensor::Dtype::FP32);
    std_.resize(C, 1, brotensor::Dtype::FP32);
    float* mp = mean.host_f32_mut();
    float* sp = std_.host_f32_mut();
    mp[0] = 0.0f; mp[1] = 0.0f; mp[2] = 0.0f;
    sp[0] = 1.0f; sp[1] = 1.0f; sp[2] = 1.0f;
    brotensor::Tensor Yn;
    broimage::image_normalize(Y, mean, std_, N, C, H, W, Yn);
    CHECK(Yn.rows == N);
    CHECK(Yn.cols == C * H * W);
    const float* YnP = Yn.host_f32();
    // mean=0, std=1 is identity.
    for (int i = 0; i < N * C * H * W; ++i) CHECK(nearf(YnP[i], Yp[i]));

    return g_failed == 0 ? 0 : 1;
}
