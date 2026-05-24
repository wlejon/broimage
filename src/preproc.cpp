#include "broimage/preproc.h"

#include <cstdint>

namespace broimage {

void u8_nhwc_to_f32_nchw(const uint8_t* src,
                         int N, int H, int W, int C,
                         float scale, float bias,
                         float* Y) {
    const int spatial = H * W;
    for (int n = 0; n < N; ++n) {
        const uint8_t* src_n = src + n * spatial * C;
        float*         y_n   = Y   + n * C * spatial;
        for (int c = 0; c < C; ++c) {
            float* y_chan = y_n + c * spatial;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    const uint8_t v = src_n[(h * W + w) * C + c];
                    y_chan[h * W + w] = static_cast<float>(v) * scale + bias;
                }
            }
        }
    }
}

void nhwc_to_nchw_f32(const float* src,
                      int N, int H, int W, int C,
                      float* Y) {
    const int spatial = H * W;
    for (int n = 0; n < N; ++n) {
        const float* src_n = src + n * spatial * C;
        float*       y_n   = Y   + n * C * spatial;
        for (int c = 0; c < C; ++c) {
            float* y_chan = y_n + c * spatial;
            for (int i = 0; i < spatial; ++i) {
                y_chan[i] = src_n[i * C + c];
            }
        }
    }
}

void nchw_to_nhwc_f32(const float* src,
                      int N, int C, int H, int W,
                      float* Y) {
    const int spatial = H * W;
    for (int n = 0; n < N; ++n) {
        const float* src_n = src + n * C * spatial;
        float*       y_n   = Y   + n * spatial * C;
        for (int i = 0; i < spatial; ++i) {
            for (int c = 0; c < C; ++c) {
                y_n[i * C + c] = src_n[c * spatial + i];
            }
        }
    }
}

void f32_nchw_to_u8_nhwc(const float* src,
                         int N, int C, int H, int W,
                         float scale, float bias,
                         uint8_t* Y) {
    const int spatial = H * W;
    for (int n = 0; n < N; ++n) {
        const float* src_n = src + n * C * spatial;
        uint8_t*     y_n   = Y   + n * spatial * C;
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                uint8_t* dp = y_n + (h * W + w) * C;
                for (int c = 0; c < C; ++c) {
                    float v = src_n[c * spatial + h * W + w] * scale + bias;
                    if (v < 0.0f) v = 0.0f;
                    if (v > 255.0f) v = 255.0f;
                    dp[c] = static_cast<uint8_t>(v + 0.5f);
                }
            }
        }
    }
}

} // namespace broimage
