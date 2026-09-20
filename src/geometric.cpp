#include "broimage/geometric.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace broimage {

namespace {

inline float cubic_weight(float t) {
    // Catmull-Rom (a = -0.5).
    const float a = -0.5f;
    const float at = std::fabs(t);
    if (at < 1.0f) {
        return (a + 2.0f) * at * at * at - (a + 3.0f) * at * at + 1.0f;
    }
    if (at < 2.0f) {
        return a * at * at * at - 5.0f * a * at * at + 8.0f * a * at - 4.0f * a;
    }
    return 0.0f;
}

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline int resolve_stride(int stride_bytes, int width, int channels) {
    return stride_bytes > 0 ? stride_bytes : width * channels;
}

inline float sinc(float x) {
    if (std::fabs(x) < 1e-8f) return 1.0f;
    const float px = 3.14159265358979323846f * x;
    return std::sin(px) / px;
}

inline float lanczos3_weight(float t) {
    const float at = std::fabs(t);
    if (at >= 3.0f) return 0.0f;
    return sinc(t) * sinc(t / 3.0f);
}

void resize_hwc_f32_nearest(const float* src, int sw, int sh, int ch,
                            float* dst, int dw, int dh) {
    for (int y = 0; y < dh; ++y) {
        int sy = static_cast<int>((y + 0.5f) * sh / dh);
        if (sy >= sh) sy = sh - 1;
        for (int x = 0; x < dw; ++x) {
            int sx = static_cast<int>((x + 0.5f) * sw / dw);
            if (sx >= sw) sx = sw - 1;
            const float* sp = src + (sy * sw + sx) * ch;
            float* dp = dst + (y * dw + x) * ch;
            for (int c = 0; c < ch; ++c) dp[c] = sp[c];
        }
    }
}

void resize_hwc_f32_bilinear(const float* src, int sw, int sh, int ch,
                             float* dst, int dw, int dh) {
    const float xs = static_cast<float>(sw) / static_cast<float>(dw);
    const float ys = static_cast<float>(sh) / static_cast<float>(dh);
    for (int y = 0; y < dh; ++y) {
        float fy = (y + 0.5f) * ys - 0.5f;
        int y0 = static_cast<int>(std::floor(fy));
        int y1 = y0 + 1;
        float ty = fy - y0;
        if (y0 < 0) { y0 = 0; ty = 0.0f; }
        if (y1 >= sh) { y1 = sh - 1; ty = 1.0f; }
        for (int x = 0; x < dw; ++x) {
            float fx = (x + 0.5f) * xs - 0.5f;
            int x0 = static_cast<int>(std::floor(fx));
            int x1 = x0 + 1;
            float tx = fx - x0;
            if (x0 < 0) { x0 = 0; tx = 0.0f; }
            if (x1 >= sw) { x1 = sw - 1; tx = 1.0f; }
            const float* p00 = src + (y0 * sw + x0) * ch;
            const float* p01 = src + (y0 * sw + x1) * ch;
            const float* p10 = src + (y1 * sw + x0) * ch;
            const float* p11 = src + (y1 * sw + x1) * ch;
            float* dp = dst + (y * dw + x) * ch;
            const float omtx = 1.0f - tx, omty = 1.0f - ty;
            for (int c = 0; c < ch; ++c) {
                const float top = p00[c] * omtx + p01[c] * tx;
                const float bot = p10[c] * omtx + p11[c] * tx;
                dp[c] = top * omty + bot * ty;
            }
        }
    }
}

void resize_hwc_f32_bicubic(const float* src, int sw, int sh, int ch,
                            float* dst, int dw, int dh) {
    const float xs = static_cast<float>(sw) / static_cast<float>(dw);
    const float ys = static_cast<float>(sh) / static_cast<float>(dh);
    for (int y = 0; y < dh; ++y) {
        const float fy = (y + 0.5f) * ys - 0.5f;
        const int   iy = static_cast<int>(std::floor(fy));
        const float ty = fy - iy;
        float wy[4] = {
            cubic_weight(-1.0f - ty),
            cubic_weight( 0.0f - ty),
            cubic_weight( 1.0f - ty),
            cubic_weight( 2.0f - ty),
        };
        for (int x = 0; x < dw; ++x) {
            const float fx = (x + 0.5f) * xs - 0.5f;
            const int   ix = static_cast<int>(std::floor(fx));
            const float tx = fx - ix;
            float wx[4] = {
                cubic_weight(-1.0f - tx),
                cubic_weight( 0.0f - tx),
                cubic_weight( 1.0f - tx),
                cubic_weight( 2.0f - tx),
            };
            float* dp = dst + (y * dw + x) * ch;
            for (int c = 0; c < ch; ++c) {
                float acc = 0.0f;
                for (int j = 0; j < 4; ++j) {
                    const int sy = clampi(iy + j - 1, 0, sh - 1);
                    float row = 0.0f;
                    for (int i = 0; i < 4; ++i) {
                        const int sx = clampi(ix + i - 1, 0, sw - 1);
                        row += wx[i] * src[(sy * sw + sx) * ch + c];
                    }
                    acc += wy[j] * row;
                }
                dp[c] = acc;
            }
        }
    }
}

// ---- Lanczos3 ---------------------------------------------------------------
//
// Separable two-pass: horizontal resize sw -> dw at original height, then
// vertical resize sh -> dh. O(ch * (dw*sw_kernel + dw*dh*sh_kernel)) instead
// of the O(ch * dw*dh*sw_kernel*sh_kernel) you'd get if we did it naively
// (also keeps the kernel weights cached and reused across rows / columns).
//
// The kernel half-width in source pixels is `max(1, 3 * scale)` where scale
// is the source-per-dst ratio. For downscales (scale > 1) we widen the window
// to preserve the low-pass behaviour — using a fixed radius-3 kernel at large
// reductions aliases just like bicubic.

void resize_axis_lanczos3(const float* src, int src_w, int src_h, int ch,
                          float* dst, int dst_w,
                          bool axis_x) {
    // axis_x == true:  src is (src_h, src_w, ch),  dst is (src_h, dst_w, ch)
    // axis_x == false: src is (src_h, src_w, ch),  dst is (dst_w, src_w, ch)
    //                  (here dst_w is actually the new height; the caller keeps
    //                  the naming consistent for the precomputed weight table).
    const int   sn    = axis_x ? src_w : src_h;
    const float scale = static_cast<float>(sn) / static_cast<float>(dst_w);
    const float fscale = std::max(1.0f, scale);
    const float radius = 3.0f * fscale;
    const int   ksz    = static_cast<int>(std::ceil(radius)) * 2;

    // Precompute the weight table for the output axis. taps[i] are source-index
    // contributions for dst index i, all normalized to sum to 1.
    std::vector<int>   first(dst_w);
    std::vector<float> weights(static_cast<std::size_t>(dst_w) * ksz, 0.0f);
    for (int i = 0; i < dst_w; ++i) {
        const float center = (i + 0.5f) * scale - 0.5f;
        const int   lo     = static_cast<int>(std::floor(center - radius)) + 1;
        first[i] = lo;
        float wsum = 0.0f;
        for (int k = 0; k < ksz; ++k) {
            const float t = (static_cast<float>(lo + k) - center) / fscale;
            const float w = lanczos3_weight(t);
            weights[i * ksz + k] = w;
            wsum += w;
        }
        if (wsum > 0.0f) {
            const float inv = 1.0f / wsum;
            for (int k = 0; k < ksz; ++k) weights[i * ksz + k] *= inv;
        }
    }

    // The row terms are widened to size_t before multiplying. In int, an offset
    // like (y * src_w + x) * ch overflows once the plane passes ~2^31 elements —
    // a 30k x 30k RGB image gets there — and that is UB in the pointer
    // arithmetic below, not merely a wrong index.
    if (axis_x) {
        for (int y = 0; y < src_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                const float* wrow = &weights[static_cast<size_t>(x) * ksz];
                const int    lo   = first[x];
                float* dp = dst + (static_cast<size_t>(y) * dst_w + x) * ch;
                for (int c = 0; c < ch; ++c) dp[c] = 0.0f;
                for (int k = 0; k < ksz; ++k) {
                    const int sx = clampi(lo + k, 0, src_w - 1);
                    const float w = wrow[k];
                    const float* sp = src + (static_cast<size_t>(y) * src_w + sx) * ch;
                    for (int c = 0; c < ch; ++c) dp[c] += sp[c] * w;
                }
            }
        }
    } else {
        // dst layout: (dst_w aka new height, src_w, ch).
        for (int y = 0; y < dst_w; ++y) {
            const float* wrow = &weights[static_cast<size_t>(y) * ksz];
            const int    lo   = first[y];
            for (int x = 0; x < src_w; ++x) {
                float* dp = dst + (static_cast<size_t>(y) * src_w + x) * ch;
                for (int c = 0; c < ch; ++c) dp[c] = 0.0f;
                for (int k = 0; k < ksz; ++k) {
                    const int sy = clampi(lo + k, 0, src_h - 1);
                    const float w = wrow[k];
                    const float* sp = src + (static_cast<size_t>(sy) * src_w + x) * ch;
                    for (int c = 0; c < ch; ++c) dp[c] += sp[c] * w;
                }
            }
        }
    }
}

void resize_hwc_f32_lanczos3(const float* src, int sw, int sh, int ch,
                             float* dst, int dw, int dh) {
    if (dw == sw && dh == sh) {
        std::memcpy(dst, src, static_cast<std::size_t>(sw) * sh * ch * sizeof(float));
        return;
    }
    // Horizontal pass: (sh, sw, ch) -> (sh, dw, ch). Skip if width unchanged.
    std::vector<float> mid;
    const float* h_out;
    int          h_out_w;
    if (dw == sw) {
        h_out = src; h_out_w = sw;
    } else {
        mid.resize(static_cast<std::size_t>(sh) * dw * ch);
        resize_axis_lanczos3(src, sw, sh, ch, mid.data(), dw, /*axis_x=*/true);
        h_out = mid.data(); h_out_w = dw;
    }
    // Vertical pass: (sh, h_out_w, ch) -> (dh, h_out_w, ch).
    if (dh == sh) {
        std::memcpy(dst, h_out,
                    static_cast<std::size_t>(sh) * h_out_w * ch * sizeof(float));
    } else {
        resize_axis_lanczos3(h_out, h_out_w, sh, ch, dst, dh, /*axis_x=*/false);
    }
}

// ---- Area / box ------------------------------------------------------------
//
// Separable box/area averaging for minification anti-aliasing.
// For any downscaled axis (dw < sw or dh < sh), each output pixel averages the
// continuous source interval it covers, eliminating moiré and aliasing.
// Any upscaled axis falls back to bilinear interpolation. Pure upscales
// (dw >= sw && dh >= sh) fall back directly to bilinear.

static void area_downsample_x(const float* src, int sw, int sh, int ch,
                              float* dst, int dw) {
    const float scale = static_cast<float>(sw) / static_cast<float>(dw);
    struct Tap {
        int start;
        int count;
    };
    std::vector<Tap> taps(dw);
    std::vector<float> weights;
    weights.reserve(static_cast<std::size_t>(dw) * (static_cast<std::size_t>(std::ceil(scale)) + 2));
    std::vector<std::size_t> weight_offsets(dw);

    for (int x = 0; x < dw; ++x) {
        float x0 = x * scale;
        float x1 = (x + 1) * scale;
        if (x == 0) x0 = 0.0f;
        if (x == dw - 1) x1 = static_cast<float>(sw);

        int ix0 = clampi(static_cast<int>(std::floor(x0)), 0, sw - 1);
        int ix1 = clampi(static_cast<int>(std::ceil(x1)), 0, sw);
        if (ix1 <= ix0) ix1 = ix0 + 1;

        taps[x].start = ix0;
        taps[x].count = ix1 - ix0;
        weight_offsets[x] = weights.size();

        float wtot = 0.0f;
        const std::size_t w_start = weights.size();
        for (int i = ix0; i < ix1; ++i) {
            float ix0f = std::max(static_cast<float>(i), x0);
            float ix1f = std::min(static_cast<float>(i + 1), x1);
            float w = std::max(0.0f, ix1f - ix0f);
            weights.push_back(w);
            wtot += w;
        }
        if (wtot > 0.0f) {
            float inv = 1.0f / wtot;
            for (std::size_t idx = w_start; idx < weights.size(); ++idx) {
                weights[idx] *= inv;
            }
        }
    }

    for (int y = 0; y < sh; ++y) {
        const float* srow = src + static_cast<std::size_t>(y) * sw * ch;
        float* drow = dst + static_cast<std::size_t>(y) * dw * ch;
        for (int x = 0; x < dw; ++x) {
            const int start = taps[x].start;
            const int count = taps[x].count;
            const float* wptr = weights.data() + weight_offsets[x];
            float* dp = drow + x * ch;
            for (int c = 0; c < ch; ++c) dp[c] = 0.0f;
            for (int k = 0; k < count; ++k) {
                int sx = clampi(start + k, 0, sw - 1);
                float w = wptr[k];
                const float* sp = srow + sx * ch;
                for (int c = 0; c < ch; ++c) dp[c] += sp[c] * w;
            }
        }
    }
}

static void area_downsample_y(const float* src, int w, int sh, int ch,
                              float* dst, int dh) {
    const float scale = static_cast<float>(sh) / static_cast<float>(dh);
    struct Tap {
        int start;
        int count;
    };
    std::vector<Tap> taps(dh);
    std::vector<float> weights;
    weights.reserve(static_cast<std::size_t>(dh) * (static_cast<std::size_t>(std::ceil(scale)) + 2));
    std::vector<std::size_t> weight_offsets(dh);

    for (int y = 0; y < dh; ++y) {
        float y0 = y * scale;
        float y1 = (y + 1) * scale;
        if (y == 0) y0 = 0.0f;
        if (y == dh - 1) y1 = static_cast<float>(sh);

        int iy0 = clampi(static_cast<int>(std::floor(y0)), 0, sh - 1);
        int iy1 = clampi(static_cast<int>(std::ceil(y1)), 0, sh);
        if (iy1 <= iy0) iy1 = iy0 + 1;

        taps[y].start = iy0;
        taps[y].count = iy1 - iy0;
        weight_offsets[y] = weights.size();

        float wtot = 0.0f;
        const std::size_t w_start = weights.size();
        for (int j = iy0; j < iy1; ++j) {
            float jy0 = std::max(static_cast<float>(j), y0);
            float jy1 = std::min(static_cast<float>(j + 1), y1);
            float w = std::max(0.0f, jy1 - jy0);
            weights.push_back(w);
            wtot += w;
        }
        if (wtot > 0.0f) {
            float inv = 1.0f / wtot;
            for (std::size_t idx = w_start; idx < weights.size(); ++idx) {
                weights[idx] *= inv;
            }
        }
    }

    const std::size_t row_floats = static_cast<std::size_t>(w) * ch;
    for (int y = 0; y < dh; ++y) {
        const int start = taps[y].start;
        const int count = taps[y].count;
        const float* wptr = weights.data() + weight_offsets[y];
        float* drow = dst + static_cast<std::size_t>(y) * row_floats;

        for (std::size_t i = 0; i < row_floats; ++i) drow[i] = 0.0f;

        for (int k = 0; k < count; ++k) {
            int sy = clampi(start + k, 0, sh - 1);
            float w = wptr[k];
            const float* srow = src + static_cast<std::size_t>(sy) * row_floats;
            for (std::size_t i = 0; i < row_floats; ++i) {
                drow[i] += srow[i] * w;
            }
        }
    }
}

void resize_hwc_f32_area(const float* src, int sw, int sh, int ch,
                         float* dst, int dw, int dh) {
    if (dw == sw && dh == sh) {
        std::memcpy(dst, src, static_cast<std::size_t>(sw) * sh * ch * sizeof(float));
        return;
    }

    if (dw >= sw && dh >= sh) {
        // Pure upsampling or identity: fall back to bilinear
        resize_hwc_f32_bilinear(src, sw, sh, ch, dst, dw, dh);
        return;
    }

    // Minification on at least one axis (dw < sw or dh < sh)
    if (dw < sw && dh < sh) {
        std::vector<float> mid(static_cast<std::size_t>(dw) * sh * ch);
        area_downsample_x(src, sw, sh, ch, mid.data(), dw);
        area_downsample_y(mid.data(), dw, sh, ch, dst, dh);
        return;
    }

    if (dw < sw && dh == sh) {
        area_downsample_x(src, sw, sh, ch, dst, dw);
        return;
    }

    if (dw == sw && dh < sh) {
        area_downsample_y(src, sw, sh, ch, dst, dh);
        return;
    }

    if (dw < sw && dh > sh) {
        std::vector<float> mid(static_cast<std::size_t>(dw) * sh * ch);
        area_downsample_x(src, sw, sh, ch, mid.data(), dw);
        resize_hwc_f32_bilinear(mid.data(), dw, sh, ch, dst, dw, dh);
        return;
    }

    if (dw > sw && dh < sh) {
        std::vector<float> mid(static_cast<std::size_t>(sw) * dh * ch);
        area_downsample_y(src, sw, sh, ch, mid.data(), dh);
        resize_hwc_f32_bilinear(mid.data(), sw, dh, ch, dst, dw, dh);
        return;
    }
}

} // namespace

void resize_hwc_f32(const float* src, int sw, int sh, int ch,
                    float* dst, int dw, int dh, Filter filter) {
    switch (filter) {
        case Filter::Nearest:  resize_hwc_f32_nearest(src, sw, sh, ch, dst, dw, dh); break;
        case Filter::Bicubic:  resize_hwc_f32_bicubic(src, sw, sh, ch, dst, dw, dh); break;
        case Filter::Lanczos3: resize_hwc_f32_lanczos3(src, sw, sh, ch, dst, dw, dh); break;
        case Filter::Area:     resize_hwc_f32_area(src, sw, sh, ch, dst, dw, dh); break;
        case Filter::Bilinear:
        default:               resize_hwc_f32_bilinear(src, sw, sh, ch, dst, dw, dh); break;
    }
}

void resize_chw_f32(const float* src, int sw, int sh, int ch,
                    float* dst, int dw, int dh, Filter filter) {
    const int src_plane = sw * sh;
    const int dst_plane = dw * dh;
    for (int c = 0; c < ch; ++c) {
        resize_hwc_f32(src + c * src_plane, sw, sh, 1,
                       dst + c * dst_plane, dw, dh, filter);
    }
}

void resize_hwc_u8(const uint8_t* src, int sw, int sh, int ch,
                   uint8_t* dst, int dw, int dh, Filter filter,
                   int src_stride_bytes, int dst_stride_bytes) {
    // Promote to float, resize, round back. Keeps a single bilinear/bicubic
    // implementation; the working-set is bounded by the larger of src or dst.
    const int src_row = resolve_stride(src_stride_bytes, sw, ch);
    const int dst_row = resolve_stride(dst_stride_bytes, dw, ch);
    const std::size_t src_n = static_cast<std::size_t>(sw) * sh * ch;
    const std::size_t dst_n = static_cast<std::size_t>(dw) * dh * ch;
    std::vector<float> sf(src_n);
    std::vector<float> df(dst_n);
    for (int y = 0; y < sh; ++y) {
        const uint8_t* srow = src + y * src_row;
        float* fp = sf.data() + std::size_t(y) * sw * ch;
        for (int i = 0; i < sw * ch; ++i) fp[i] = static_cast<float>(srow[i]);
    }
    resize_hwc_f32(sf.data(), sw, sh, ch, df.data(), dw, dh, filter);
    for (int y = 0; y < dh; ++y) {
        uint8_t* drow = dst + y * dst_row;
        const float* fp = df.data() + std::size_t(y) * dw * ch;
        for (int i = 0; i < dw * ch; ++i) {
            float v = fp[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 255.0f) v = 255.0f;
            drow[i] = static_cast<uint8_t>(v + 0.5f);
        }
    }
}

void crop_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                 uint8_t* dst, int x, int y, int w, int h,
                 int src_stride_bytes, int dst_stride_bytes) {
    const int src_row = resolve_stride(src_stride_bytes, src_w, channels);
    const int dst_row = resolve_stride(dst_stride_bytes, w,     channels);
    for (int j = 0; j < h; ++j) {
        const int sy = clampi(y + j, 0, src_h - 1);
        uint8_t* drow = dst + j * dst_row;
        const uint8_t* srow = src + sy * src_row;
        for (int i = 0; i < w; ++i) {
            const int sx = clampi(x + i, 0, src_w - 1);
            const uint8_t* sp = srow + sx * channels;
            uint8_t* dp = drow + i * channels;
            for (int c = 0; c < channels; ++c) dp[c] = sp[c];
        }
    }
}

void center_crop_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                        uint8_t* dst, int crop_w, int crop_h,
                        int src_stride_bytes, int dst_stride_bytes) {
    const int x = (src_w - crop_w) / 2;
    const int y = (src_h - crop_h) / 2;
    crop_hwc_u8(src, src_w, src_h, channels, dst, x, y, crop_w, crop_h,
                src_stride_bytes, dst_stride_bytes);
}

void pad_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                uint8_t* dst, int dst_w, int dst_h,
                int off_x, int off_y,
                uint8_t pad_r, uint8_t pad_g, uint8_t pad_b, uint8_t pad_a,
                int src_stride_bytes, int dst_stride_bytes) {
    const int src_row = resolve_stride(src_stride_bytes, src_w, channels);
    const int dst_row = resolve_stride(dst_stride_bytes, dst_w, channels);
    const uint8_t pad[4] = { pad_r, pad_g, pad_b, pad_a };
    for (int y = 0; y < dst_h; ++y) {
        uint8_t* drow = dst + y * dst_row;
        const int sy = y - off_y;
        const uint8_t* srow = (sy >= 0 && sy < src_h) ? src + sy * src_row : nullptr;
        for (int x = 0; x < dst_w; ++x) {
            uint8_t* dp = drow + x * channels;
            const int sx = x - off_x;
            if (srow && sx >= 0 && sx < src_w) {
                const uint8_t* sp = srow + sx * channels;
                for (int c = 0; c < channels; ++c) dp[c] = sp[c];
            } else {
                for (int c = 0; c < channels; ++c) {
                    dp[c] = (c < 4) ? pad[c] : 0;
                }
            }
        }
    }
}

void letterbox_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                      uint8_t* dst, int dst_w, int dst_h,
                      uint8_t pad_r, uint8_t pad_g, uint8_t pad_b, uint8_t pad_a,
                      Filter filter,
                      int* out_x, int* out_y, int* out_w, int* out_h) {
    const float sx = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float sy = static_cast<float>(dst_h) / static_cast<float>(src_h);
    const float s = std::min(sx, sy);
    int new_w = std::max(1, static_cast<int>(std::round(src_w * s)));
    int new_h = std::max(1, static_cast<int>(std::round(src_h * s)));
    if (new_w > dst_w) new_w = dst_w;
    if (new_h > dst_h) new_h = dst_h;
    const int off_x = (dst_w - new_w) / 2;
    const int off_y = (dst_h - new_h) / 2;

    // Resize source into a scratch buffer at the target content rect, then pad.
    std::vector<uint8_t> scratch(
        static_cast<std::size_t>(new_w) * new_h * channels);
    resize_hwc_u8(src, src_w, src_h, channels, scratch.data(), new_w, new_h, filter);
    pad_hwc_u8(scratch.data(), new_w, new_h, channels, dst, dst_w, dst_h,
               off_x, off_y, pad_r, pad_g, pad_b, pad_a);

    if (out_x) *out_x = off_x;
    if (out_y) *out_y = off_y;
    if (out_w) *out_w = new_w;
    if (out_h) *out_h = new_h;
}

void flip_horizontal_hwc_u8(const uint8_t* src, uint8_t* dst,
                            int w, int h, int channels,
                            int src_stride_bytes, int dst_stride_bytes) {
    const int src_row = resolve_stride(src_stride_bytes, w, channels);
    const int dst_row = resolve_stride(dst_stride_bytes, w, channels);
    for (int y = 0; y < h; ++y) {
        const uint8_t* srow = src + y * src_row;
        uint8_t*       drow = dst + y * dst_row;
        for (int x = 0; x < w; ++x) {
            const uint8_t* sp = srow + (w - 1 - x) * channels;
            uint8_t* dp = drow + x * channels;
            for (int c = 0; c < channels; ++c) dp[c] = sp[c];
        }
    }
}

void flip_vertical_hwc_u8(const uint8_t* src, uint8_t* dst,
                          int w, int h, int channels,
                          int src_stride_bytes, int dst_stride_bytes) {
    const int src_row = resolve_stride(src_stride_bytes, w, channels);
    const int dst_row = resolve_stride(dst_stride_bytes, w, channels);
    const std::size_t copy_bytes = static_cast<std::size_t>(w) * channels;
    for (int y = 0; y < h; ++y) {
        std::memcpy(dst + y * dst_row,
                    src + (h - 1 - y) * src_row,
                    copy_bytes);
    }
}

void rotate_90_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                      uint8_t* dst, int turns,
                      int src_stride_bytes, int dst_stride_bytes) {
    turns = ((turns % 4) + 4) % 4;
    const int src_row = resolve_stride(src_stride_bytes, src_w, channels);
    const int dst_w_dims = (turns % 2 == 0) ? src_w : src_h;
    const int dst_row = resolve_stride(dst_stride_bytes, dst_w_dims, channels);

    if (turns == 0) {
        const std::size_t copy_bytes =
            static_cast<std::size_t>(src_w) * channels;
        for (int y = 0; y < src_h; ++y) {
            std::memcpy(dst + y * dst_row, src + y * src_row, copy_bytes);
        }
        return;
    }
    if (turns == 2) {
        for (int y = 0; y < src_h; ++y) {
            const uint8_t* srow = src + (src_h - 1 - y) * src_row;
            uint8_t*       drow = dst + y * dst_row;
            for (int x = 0; x < src_w; ++x) {
                const uint8_t* sp = srow + (src_w - 1 - x) * channels;
                uint8_t* dp = drow + x * channels;
                for (int c = 0; c < channels; ++c) dp[c] = sp[c];
            }
        }
        return;
    }
    // dst dims: (dh = src_w, dw = src_h).
    //   turns == 1 (90 CCW):  src[x, src_w-1-y]
    //   turns == 3 (90 CW):   src[src_h-1-x, y]
    const int dw = src_h, dh = src_w;
    for (int y = 0; y < dh; ++y) {
        uint8_t* drow = dst + y * dst_row;
        for (int x = 0; x < dw; ++x) {
            int sx, sy;
            if (turns == 1) { sx = src_w - 1 - y; sy = x;             }
            else            { sx = y;             sy = src_h - 1 - x; }
            const uint8_t* sp = src + sy * src_row + sx * channels;
            uint8_t* dp = drow + x * channels;
            for (int c = 0; c < channels; ++c) dp[c] = sp[c];
        }
    }
}

} // namespace broimage
