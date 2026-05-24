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

} // namespace broimage
