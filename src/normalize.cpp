#include "broimage/normalize.h"

namespace broimage {

void image_normalize_nchw_f32(const float* X,
                              const float* mean,
                              const float* std_,
                              int N, int C, int H, int W,
                              float* Y) {
    const int spatial = H * W;
    for (int c = 0; c < C; ++c) {
        const float mu = mean[c];
        const float s  = std_[c];
        const float inv = (s != 0.0f) ? (1.0f / s) : 0.0f;
        for (int n = 0; n < N; ++n) {
            const float* x_chan = X + (n * C + c) * spatial;
            float*       y_chan = Y + (n * C + c) * spatial;
            for (int i = 0; i < spatial; ++i) {
                y_chan[i] = (x_chan[i] - mu) * inv;
            }
        }
    }
}

} // namespace broimage
