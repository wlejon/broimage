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

} // namespace

void resize_hwc_f32(const float* src, int sw, int sh, int ch,
                    float* dst, int dw, int dh, Filter filter) {
    switch (filter) {
        case Filter::Nearest:  resize_hwc_f32_nearest(src, sw, sh, ch, dst, dw, dh); break;
        case Filter::Bicubic:  resize_hwc_f32_bicubic(src, sw, sh, ch, dst, dw, dh); break;
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
                   uint8_t* dst, int dw, int dh, Filter filter) {
    // Promote to float, resize, round back. Keeps a single bilinear/bicubic
    // implementation; the working-set is bounded by the larger of src or dst.
    const std::size_t src_n = static_cast<std::size_t>(sw) * sh * ch;
    const std::size_t dst_n = static_cast<std::size_t>(dw) * dh * ch;
    std::vector<float> sf(src_n);
    std::vector<float> df(dst_n);
    for (std::size_t i = 0; i < src_n; ++i) sf[i] = static_cast<float>(src[i]);
    resize_hwc_f32(sf.data(), sw, sh, ch, df.data(), dw, dh, filter);
    for (std::size_t i = 0; i < dst_n; ++i) {
        float v = df[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        dst[i] = static_cast<uint8_t>(v + 0.5f);
    }
}

void crop_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                 uint8_t* dst, int x, int y, int w, int h) {
    for (int j = 0; j < h; ++j) {
        const int sy = clampi(y + j, 0, src_h - 1);
        for (int i = 0; i < w; ++i) {
            const int sx = clampi(x + i, 0, src_w - 1);
            const uint8_t* sp = src + (sy * src_w + sx) * channels;
            uint8_t* dp = dst + (j * w + i) * channels;
            for (int c = 0; c < channels; ++c) dp[c] = sp[c];
        }
    }
}

void center_crop_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                        uint8_t* dst, int crop_w, int crop_h) {
    const int x = (src_w - crop_w) / 2;
    const int y = (src_h - crop_h) / 2;
    crop_hwc_u8(src, src_w, src_h, channels, dst, x, y, crop_w, crop_h);
}

void pad_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                uint8_t* dst, int dst_w, int dst_h,
                int off_x, int off_y,
                uint8_t pad_r, uint8_t pad_g, uint8_t pad_b, uint8_t pad_a) {
    const uint8_t pad[4] = { pad_r, pad_g, pad_b, pad_a };
    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            uint8_t* dp = dst + (y * dst_w + x) * channels;
            const int sx = x - off_x;
            const int sy = y - off_y;
            if (sx >= 0 && sx < src_w && sy >= 0 && sy < src_h) {
                const uint8_t* sp = src + (sy * src_w + sx) * channels;
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
                            int w, int h, int channels) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t* sp = src + (y * w + (w - 1 - x)) * channels;
            uint8_t* dp = dst + (y * w + x) * channels;
            for (int c = 0; c < channels; ++c) dp[c] = sp[c];
        }
    }
}

void flip_vertical_hwc_u8(const uint8_t* src, uint8_t* dst,
                          int w, int h, int channels) {
    const int row = w * channels;
    for (int y = 0; y < h; ++y) {
        std::memcpy(dst + y * row, src + (h - 1 - y) * row,
                    static_cast<std::size_t>(row));
    }
}

void rotate_90_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                      uint8_t* dst, int turns) {
    turns = ((turns % 4) + 4) % 4;
    if (turns == 0) {
        std::memcpy(dst, src,
                    static_cast<std::size_t>(src_w) * src_h * channels);
        return;
    }
    if (turns == 2) {
        for (int y = 0; y < src_h; ++y) {
            for (int x = 0; x < src_w; ++x) {
                const uint8_t* sp = src + ((src_h - 1 - y) * src_w + (src_w - 1 - x)) * channels;
                uint8_t* dp = dst + (y * src_w + x) * channels;
                for (int c = 0; c < channels; ++c) dp[c] = sp[c];
            }
        }
        return;
    }
    // dst dims: (dh = src_w, dw = src_h). For dst[y, x]:
    //   90 CCW (turns=1):  src[x, src_w-1-y]   — top row of dst = right column of src.
    //   90 CW  (turns=3):  src[src_h-1-x, y]   — top row of dst = left column of src reversed.
    const int dw = src_h, dh = src_w;
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            int sx, sy;
            if (turns == 1) { sx = src_w - 1 - y; sy = x;             }
            else            { sx = y;             sy = src_h - 1 - x; }
            const uint8_t* sp = src + (sy * src_w + sx) * channels;
            uint8_t* dp = dst + (y * dw + x) * channels;
            for (int c = 0; c < channels; ++c) dp[c] = sp[c];
        }
    }
}

} // namespace broimage
