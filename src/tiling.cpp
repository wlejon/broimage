#include "broimage/tiling.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace broimage {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Raised-cosine ramp on [0,1]: 0 at t<=0, 1 at t>=1, smooth (C1) between.
inline float cos_ramp(double t) {
    if (t <= 0.0) return 0.0f;
    if (t >= 1.0) return 1.0f;
    return static_cast<float>(0.5 - 0.5 * std::cos(kPi * t));
}

// Fill a 1D axis weight of length n: ramp up across `a` leading pixels, hold at
// 1 in the middle, ramp down across `b` trailing pixels. Sampling at pixel
// centres ((i+0.5)/a) keeps the ramp strictly positive at the very edge.
void axis_ramp(std::vector<float>& w, int n, int a, int b) {
    w.assign(static_cast<std::size_t>(std::max(n, 0)), 1.0f);
    if (n <= 0) return;
    a = std::clamp(a, 0, n);
    b = std::clamp(b, 0, n - a);   // if both overlaps collide, trailing yields
    for (int i = 0; i < a; ++i)
        w[i] = cos_ramp((i + 0.5) / a);
    for (int j = 0; j < b; ++j)
        w[n - b + j] = cos_ramp((b - j - 0.5) / b);
}

}  // namespace

void feather_window_f32(float* w, int tw, int th,
                        int ov_l, int ov_r, int ov_t, int ov_b) {
    if (!w || tw <= 0 || th <= 0) return;

    std::vector<float> rx, ry;
    axis_ramp(rx, tw, ov_l, ov_r);
    axis_ramp(ry, th, ov_t, ov_b);

    for (int y = 0; y < th; ++y) {
        const float wy = ry[y];
        float* row = w + static_cast<std::size_t>(y) * tw;
        for (int x = 0; x < tw; ++x)
            row[x] = rx[x] * wy;
    }
}

void accumulate_tile_f32(float* acc, float* wacc,
                         int full_w, int full_h, int channels,
                         const float* tile, int tw, int th,
                         int dst_x, int dst_y, const float* window) {
    if (!acc || !wacc || !tile || !window) return;
    if (full_w <= 0 || full_h <= 0 || channels <= 0 || tw <= 0 || th <= 0) return;

    for (int ty = 0; ty < th; ++ty) {
        const int fy = dst_y + ty;
        if (fy < 0 || fy >= full_h) continue;
        for (int tx = 0; tx < tw; ++tx) {
            const int fx = dst_x + tx;
            if (fx < 0 || fx >= full_w) continue;

            const std::size_t tpix = static_cast<std::size_t>(ty) * tw + tx;
            const float win = window[tpix];
            const std::size_t fpix = static_cast<std::size_t>(fy) * full_w + fx;

            const float* tsrc = tile + tpix * channels;
            float* adst = acc + fpix * channels;
            for (int c = 0; c < channels; ++c)
                adst[c] += tsrc[c] * win;
            wacc[fpix] += win;
        }
    }
}

void normalize_accumulator_f32(float* acc, const float* wacc,
                               int n_pixels, int channels, float eps) {
    if (!acc || !wacc || n_pixels <= 0 || channels <= 0) return;
    for (int i = 0; i < n_pixels; ++i) {
        const float inv = 1.0f / std::max(wacc[i], eps);
        float* a = acc + static_cast<std::size_t>(i) * channels;
        for (int c = 0; c < channels; ++c)
            a[c] *= inv;
    }
}

}  // namespace broimage
