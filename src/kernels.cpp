#include "broimage/kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace broimage {

// ----- gradient --------------------------------------------------------------

void gradient(const GradientStop* stops, int stop_count, int n,
              std::vector<uint8_t>& out) {
    out.assign(static_cast<std::size_t>(n) * 4, 0);
    if (stop_count < 2 || n < 2) return;

    // Per the header contract: a < 0 means "fully opaque" (255).
    auto alpha_of = [](const GradientStop& s) {
        return (s.a < 0.0f) ? 255.0f : s.a;
    };

    int cur = 0;
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(n - 1);
        while (cur + 1 < stop_count - 1 && stops[cur + 1].t < t) ++cur;

        // Clamp to endpoints outside the explicit stop range.
        if (t >= stops[stop_count - 1].t) {
            const GradientStop& last = stops[stop_count - 1];
            out[i * 4 + 0] = static_cast<uint8_t>(std::clamp(last.r, 0.0f, 255.0f));
            out[i * 4 + 1] = static_cast<uint8_t>(std::clamp(last.g, 0.0f, 255.0f));
            out[i * 4 + 2] = static_cast<uint8_t>(std::clamp(last.b, 0.0f, 255.0f));
            out[i * 4 + 3] = static_cast<uint8_t>(std::clamp(alpha_of(last), 0.0f, 255.0f));
            continue;
        }

        const GradientStop& s0 = stops[cur];
        const GradientStop& s1 = stops[cur + 1];
        const float span = s1.t - s0.t;
        float u = (span > 1e-9f) ? (t - s0.t) / span : 0.0f;
        if (t <= stops[0].t) u = 0.0f;
        u = std::clamp(u, 0.0f, 1.0f);

        const float a0 = alpha_of(s0);
        const float a1 = alpha_of(s1);
        const float r = s0.r + (s1.r - s0.r) * u;
        const float g = s0.g + (s1.g - s0.g) * u;
        const float b = s0.b + (s1.b - s0.b) * u;
        const float a = a0    + (a1    - a0)    * u;
        out[i * 4 + 0] = static_cast<uint8_t>(std::clamp(r, 0.0f, 255.0f));
        out[i * 4 + 1] = static_cast<uint8_t>(std::clamp(g, 0.0f, 255.0f));
        out[i * 4 + 2] = static_cast<uint8_t>(std::clamp(b, 0.0f, 255.0f));
        out[i * 4 + 3] = static_cast<uint8_t>(std::clamp(a, 0.0f, 255.0f));
    }
}

// ----- lookup ----------------------------------------------------------------

void lookup_f32(const float* src, int n_pixels,
                const uint8_t* lut, int lut_n,
                uint8_t* dst,
                float lo, float hi,
                LookupEdge edge) {
    const float inv_span = (hi > lo) ? (1.0f / (hi - lo)) : 0.0f;
    const float idx_max  = static_cast<float>(lut_n - 1);
    const bool wrap = (edge == LookupEdge::Wrap);

    for (int i = 0; i < n_pixels; ++i) {
        const float t  = (src[i] - lo) * inv_span;
        float       fi = t * idx_max;
        int idx;
        if (wrap) {
            const float lf = static_cast<float>(lut_n);
            fi = std::fmod(fi, lf);
            if (fi < 0.0f) fi += lf;
            idx = static_cast<int>(fi);
            if (idx >= lut_n) idx = lut_n - 1;
        } else {
            if (fi < 0.0f) fi = 0.0f;
            if (fi > idx_max) fi = idx_max;
            idx = static_cast<int>(fi);
        }
        const uint8_t* lp = lut + idx * 4;
        dst[i * 4 + 0] = lp[0];
        dst[i * 4 + 1] = lp[1];
        dst[i * 4 + 2] = lp[2];
        dst[i * 4 + 3] = lp[3];
    }
}

// ----- reduce ----------------------------------------------------------------

MinMax reduce_minmax_f32(const float* src, int n, int stride) {
    if (stride < 1) stride = 1;
    float mn = src[0], mx = src[0];
    for (int i = stride; i < n; i += stride) {
        const float v = src[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    return { mn, mx };
}

double reduce_sum_f32(const float* src, int n, int stride) {
    if (stride < 1) stride = 1;
    double sum = 0.0;
    for (int i = 0; i < n; i += stride) sum += src[i];
    return sum;
}

double reduce_mean_f32(const float* src, int n, int stride) {
    if (stride < 1) stride = 1;
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < n; i += stride) { sum += src[i]; ++count; }
    return (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
}

void reduce_histogram_f32(const float* src, int n,
                          int bins, float lo, float hi,
                          uint32_t* out_counts,
                          int stride) {
    if (stride < 1) stride = 1;
    for (int b = 0; b < bins; ++b) out_counts[b] = 0;
    if (bins < 1 || hi <= lo) return;
    const float inv_span = 1.0f / (hi - lo);
    for (int i = 0; i < n; i += stride) {
        const float t = (src[i] - lo) * inv_span;
        const int idx = static_cast<int>(t * static_cast<float>(bins));
        if (idx < 0 || idx >= bins) continue;
        out_counts[idx]++;
    }
}

// ----- map -------------------------------------------------------------------

void map_affine_f32(const float* src, float* dst, int n, float a, float b) {
    for (int i = 0; i < n; ++i) dst[i] = src[i] * a + b;
}

void map_affine_clamp_f32(const float* src, float* dst, int n,
                          float a, float b, float clamp_lo, float clamp_hi) {
    for (int i = 0; i < n; ++i) {
        float v = src[i] * a + b;
        if (v < clamp_lo) v = clamp_lo;
        if (v > clamp_hi) v = clamp_hi;
        dst[i] = v;
    }
}

void map_abs_f32(const float* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = std::fabs(src[i]);
}

void map_log_f32(const float* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = std::log(src[i]);
}

void map_sqrt_f32(const float* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = std::sqrt(src[i]);
}

void map_exp_f32(const float* src, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = std::exp(src[i]);
}

void map_pow_f32(const float* src, float* dst, int n, float exponent) {
    for (int i = 0; i < n; ++i) dst[i] = std::pow(src[i], exponent);
}

// ----- combine ---------------------------------------------------------------

void combine_add_f32(const float* a, const float* b, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = a[i] + b[i];
}
void combine_sub_f32(const float* a, const float* b, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = a[i] - b[i];
}
void combine_mul_f32(const float* a, const float* b, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = a[i] * b[i];
}
void combine_min_f32(const float* a, const float* b, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = std::min(a[i], b[i]);
}
void combine_max_f32(const float* a, const float* b, float* dst, int n) {
    for (int i = 0; i < n; ++i) dst[i] = std::max(a[i], b[i]);
}
void combine_lerp_f32(const float* a, const float* b, float* dst, int n, float t) {
    const float omt = 1.0f - t;
    for (int i = 0; i < n; ++i) dst[i] = a[i] * omt + b[i] * t;
}
void combine_wsum_f32(const float* a, const float* b, float* dst, int n,
                      float wa, float wb) {
    for (int i = 0; i < n; ++i) dst[i] = a[i] * wa + b[i] * wb;
}

// ----- stencil ---------------------------------------------------------------

void stencil_f32(const float* src, float* dst,
                 int src_w, int src_h,
                 const float* kernel, int kw, int kh,
                 float divisor, float bias,
                 StencilEdge edge) {
    const float inv_div = (divisor != 0.0f) ? (1.0f / divisor) : 1.0f;
    const int hkw = kw / 2;
    const int hkh = kh / 2;

    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
            float acc = 0.0f;
            for (int ky = 0; ky < kh; ++ky) {
                int sy = y + ky - hkh;
                int syy = 0;
                bool zero_row = false;
                if (edge == StencilEdge::Clamp) {
                    if (sy < 0) sy = 0;
                    if (sy >= src_h) sy = src_h - 1;
                    syy = sy;
                } else if (edge == StencilEdge::Wrap) {
                    syy = ((sy % src_h) + src_h) % src_h;
                } else {
                    if (sy < 0 || sy >= src_h) zero_row = true;
                    else syy = sy;
                }
                for (int kx = 0; kx < kw; ++kx) {
                    int sx = x + kx - hkw;
                    int sxx = 0;
                    bool zero = zero_row;
                    if (!zero) {
                        if (edge == StencilEdge::Clamp) {
                            if (sx < 0) sx = 0;
                            if (sx >= src_w) sx = src_w - 1;
                            sxx = sx;
                        } else if (edge == StencilEdge::Wrap) {
                            sxx = ((sx % src_w) + src_w) % src_w;
                        } else {
                            if (sx < 0 || sx >= src_w) zero = true;
                            else sxx = sx;
                        }
                    }
                    const float v = zero ? 0.0f : src[syy * src_w + sxx];
                    acc += v * kernel[ky * kw + kx];
                }
            }
            dst[y * src_w + x] = acc * inv_div + bias;
        }
    }
}

void stencil_hwc_f32(const float* src, float* dst,
                     int src_w, int src_h, int channels,
                     const float* kernel, int kw, int kh,
                     float divisor, float bias,
                     StencilEdge edge) {
    const float inv_div = (divisor != 0.0f) ? (1.0f / divisor) : 1.0f;
    const int hkw = kw / 2;
    const int hkh = kh / 2;

    for (int y = 0; y < src_h; ++y) {
        for (int x = 0; x < src_w; ++x) {
            // One accumulator per channel; keeps the kernel-traversal loops
            // outside, so the per-tap edge handling cost is paid once for all
            // channels of the pixel instead of once per (pixel * channel).
            float acc[8] = {0,0,0,0,0,0,0,0};
            float* accp = acc;
            std::vector<float> acc_heap;
            if (channels > 8) {
                acc_heap.assign(channels, 0.0f);
                accp = acc_heap.data();
            }
            for (int ky = 0; ky < kh; ++ky) {
                int sy = y + ky - hkh;
                int syy = 0;
                bool zero_row = false;
                if (edge == StencilEdge::Clamp) {
                    if (sy < 0) sy = 0;
                    if (sy >= src_h) sy = src_h - 1;
                    syy = sy;
                } else if (edge == StencilEdge::Wrap) {
                    syy = ((sy % src_h) + src_h) % src_h;
                } else {
                    if (sy < 0 || sy >= src_h) zero_row = true;
                    else syy = sy;
                }
                for (int kx = 0; kx < kw; ++kx) {
                    int sx = x + kx - hkw;
                    int sxx = 0;
                    bool zero = zero_row;
                    if (!zero) {
                        if (edge == StencilEdge::Clamp) {
                            if (sx < 0) sx = 0;
                            if (sx >= src_w) sx = src_w - 1;
                            sxx = sx;
                        } else if (edge == StencilEdge::Wrap) {
                            sxx = ((sx % src_w) + src_w) % src_w;
                        } else {
                            if (sx < 0 || sx >= src_w) zero = true;
                            else sxx = sx;
                        }
                    }
                    const float w = kernel[ky * kw + kx];
                    if (zero) continue;
                    const float* sp = src + (syy * src_w + sxx) * channels;
                    for (int c = 0; c < channels; ++c) accp[c] += w * sp[c];
                }
            }
            float* dp = dst + (y * src_w + x) * channels;
            for (int c = 0; c < channels; ++c) dp[c] = accp[c] * inv_div + bias;
        }
    }
}

// ----- resample --------------------------------------------------------------

void resample_f32(const float* src, int src_w, int src_h,
                  float*       dst, int dst_w, int dst_h,
                  int channels,
                  Filter filter) {
    // Delegates to the geometric resize so kernel-verb callers and geometric
    // callers share one implementation (and pick up any future improvements
    // like SIMD / area filter in one place).
    resize_hwc_f32(src, src_w, src_h, channels,
                   dst, dst_w, dst_h, filter);
}

} // namespace broimage
