#include "broimage/color.h"

#include <cmath>
#include <cstdint>

namespace broimage {

void rgba_to_rgb_u8(const uint8_t* src, uint8_t* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        dst[i * 3 + 0] = src[i * 4 + 0];
        dst[i * 3 + 1] = src[i * 4 + 1];
        dst[i * 3 + 2] = src[i * 4 + 2];
    }
}

void rgb_to_rgba_u8(const uint8_t* src, uint8_t* dst, int pixel_count,
                    uint8_t alpha) {
    for (int i = 0; i < pixel_count; ++i) {
        dst[i * 4 + 0] = src[i * 3 + 0];
        dst[i * 4 + 1] = src[i * 3 + 1];
        dst[i * 4 + 2] = src[i * 3 + 2];
        dst[i * 4 + 3] = alpha;
    }
}

void rgba_to_gray_u8(const uint8_t* src, uint8_t* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        const int r = src[i * 4 + 0];
        const int g = src[i * 4 + 1];
        const int b = src[i * 4 + 2];
        // Rec.601 with stb_image's integer weights (77 R + 150 G + 29 B) / 256.
        dst[i] = static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
    }
}

void rgb_to_gray_u8(const uint8_t* src, uint8_t* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        const int r = src[i * 3 + 0];
        const int g = src[i * 3 + 1];
        const int b = src[i * 3 + 2];
        dst[i] = static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
    }
}

void hwc_to_chw_f32(const float* src, float* dst,
                    int width, int height, int channels) {
    const int spatial = width * height;
    for (int c = 0; c < channels; ++c) {
        float* dchan = dst + c * spatial;
        for (int i = 0; i < spatial; ++i) {
            dchan[i] = src[i * channels + c];
        }
    }
}

void chw_to_hwc_f32(const float* src, float* dst,
                    int width, int height, int channels) {
    const int spatial = width * height;
    for (int i = 0; i < spatial; ++i) {
        for (int c = 0; c < channels; ++c) {
            dst[i * channels + c] = src[c * spatial + i];
        }
    }
}

void apply_gamma_f32(const float* src, float* dst, int element_count,
                     float gamma) {
    const float inv = (gamma != 0.0f) ? (1.0f / gamma) : 1.0f;
    for (int i = 0; i < element_count; ++i) {
        const float v = src[i];
        dst[i] = (v > 0.0f) ? std::pow(v, inv) : v;
    }
}

namespace {

inline float srgb_to_linear_one(float v) {
    if (v <= 0.04045f) return v / 12.92f;
    return std::pow((v + 0.055f) / 1.055f, 2.4f);
}

inline float linear_to_srgb_one(float v) {
    if (v <= 0.0031308f) return v * 12.92f;
    return 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
}

} // namespace

void srgb_to_linear_f32(const float* src, float* dst, int element_count) {
    for (int i = 0; i < element_count; ++i) dst[i] = srgb_to_linear_one(src[i]);
}

void linear_to_srgb_f32(const float* src, float* dst, int element_count) {
    for (int i = 0; i < element_count; ++i) dst[i] = linear_to_srgb_one(src[i]);
}

void srgb_to_linear_u8_to_f32(const uint8_t* src, float* dst, int element_count) {
    for (int i = 0; i < element_count; ++i) {
        dst[i] = srgb_to_linear_one(static_cast<float>(src[i]) * (1.0f / 255.0f));
    }
}

void linear_f32_to_srgb_u8(const float* src, uint8_t* dst, int element_count) {
    for (int i = 0; i < element_count; ++i) {
        float v = linear_to_srgb_one(src[i]) * 255.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        dst[i] = static_cast<uint8_t>(v + 0.5f);
    }
}

// ----- HSV / HSL ------------------------------------------------------------

namespace {

inline void rgb_to_hsv_one(float r, float g, float b,
                           float& h, float& s, float& v) {
    const float mx = std::fmax(r, std::fmax(g, b));
    const float mn = std::fmin(r, std::fmin(g, b));
    const float d  = mx - mn;
    v = mx;
    s = (mx > 0.0f) ? (d / mx) : 0.0f;
    if (d <= 0.0f) { h = 0.0f; return; }
    float hh;
    if      (mx == r) hh = (g - b) / d + (g < b ? 6.0f : 0.0f);
    else if (mx == g) hh = (b - r) / d + 2.0f;
    else              hh = (r - g) / d + 4.0f;
    h = hh / 6.0f;
}

inline void hsv_to_rgb_one(float h, float s, float v,
                           float& r, float& g, float& b) {
    if (s <= 0.0f) { r = g = b = v; return; }
    h = h - std::floor(h); // wrap into [0, 1)
    const float hh = h * 6.0f;
    const int   i  = static_cast<int>(std::floor(hh));
    const float f  = hh - i;
    const float p  = v * (1.0f - s);
    const float q  = v * (1.0f - s * f);
    const float t  = v * (1.0f - s * (1.0f - f));
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default:r = v; g = p; b = q; break;
    }
}

inline void rgb_to_hsl_one(float r, float g, float b,
                           float& h, float& s, float& l) {
    const float mx = std::fmax(r, std::fmax(g, b));
    const float mn = std::fmin(r, std::fmin(g, b));
    const float d  = mx - mn;
    l = 0.5f * (mx + mn);
    if (d <= 0.0f) { h = 0.0f; s = 0.0f; return; }
    s = (l > 0.5f) ? (d / (2.0f - mx - mn)) : (d / (mx + mn));
    float hh;
    if      (mx == r) hh = (g - b) / d + (g < b ? 6.0f : 0.0f);
    else if (mx == g) hh = (b - r) / d + 2.0f;
    else              hh = (r - g) / d + 4.0f;
    h = hh / 6.0f;
}

inline float hsl_hue_to_channel(float p, float q, float t) {
    if (t < 0.0f) t += 1.0f;
    if (t > 1.0f) t -= 1.0f;
    if (t < 1.0f / 6.0f) return p + (q - p) * 6.0f * t;
    if (t < 1.0f / 2.0f) return q;
    if (t < 2.0f / 3.0f) return p + (q - p) * (2.0f / 3.0f - t) * 6.0f;
    return p;
}

inline void hsl_to_rgb_one(float h, float s, float l,
                           float& r, float& g, float& b) {
    if (s <= 0.0f) { r = g = b = l; return; }
    h = h - std::floor(h);
    const float q = (l < 0.5f) ? (l * (1.0f + s)) : (l + s - l * s);
    const float p = 2.0f * l - q;
    r = hsl_hue_to_channel(p, q, h + 1.0f / 3.0f);
    g = hsl_hue_to_channel(p, q, h);
    b = hsl_hue_to_channel(p, q, h - 1.0f / 3.0f);
}

} // namespace

void rgb_to_hsv_f32(const float* src, float* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        float h, s, v;
        rgb_to_hsv_one(src[i * 3 + 0], src[i * 3 + 1], src[i * 3 + 2], h, s, v);
        dst[i * 3 + 0] = h; dst[i * 3 + 1] = s; dst[i * 3 + 2] = v;
    }
}

void hsv_to_rgb_f32(const float* src, float* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        float r, g, b;
        hsv_to_rgb_one(src[i * 3 + 0], src[i * 3 + 1], src[i * 3 + 2], r, g, b);
        dst[i * 3 + 0] = r; dst[i * 3 + 1] = g; dst[i * 3 + 2] = b;
    }
}

void rgb_to_hsl_f32(const float* src, float* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        float h, s, l;
        rgb_to_hsl_one(src[i * 3 + 0], src[i * 3 + 1], src[i * 3 + 2], h, s, l);
        dst[i * 3 + 0] = h; dst[i * 3 + 1] = s; dst[i * 3 + 2] = l;
    }
}

void hsl_to_rgb_f32(const float* src, float* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        float r, g, b;
        hsl_to_rgb_one(src[i * 3 + 0], src[i * 3 + 1], src[i * 3 + 2], r, g, b);
        dst[i * 3 + 0] = r; dst[i * 3 + 1] = g; dst[i * 3 + 2] = b;
    }
}

// ----- Color matrix ---------------------------------------------------------

void apply_color_matrix_3x3_f32(const float* src, float* dst, int pixel_count,
                                int channels, const float m[9]) {
    for (int i = 0; i < pixel_count; ++i) {
        const float r = src[i * channels + 0];
        const float g = src[i * channels + 1];
        const float b = src[i * channels + 2];
        const float a = (channels == 4) ? src[i * channels + 3] : 0.0f;
        const float r2 = m[0] * r + m[1] * g + m[2] * b;
        const float g2 = m[3] * r + m[4] * g + m[5] * b;
        const float b2 = m[6] * r + m[7] * g + m[8] * b;
        dst[i * channels + 0] = r2;
        dst[i * channels + 1] = g2;
        dst[i * channels + 2] = b2;
        if (channels == 4) dst[i * channels + 3] = a;
    }
}

void apply_color_matrix_3x4_f32(const float* src, float* dst, int pixel_count,
                                int channels, const float m[12]) {
    for (int i = 0; i < pixel_count; ++i) {
        const float r = src[i * channels + 0];
        const float g = src[i * channels + 1];
        const float b = src[i * channels + 2];
        const float a = (channels == 4) ? src[i * channels + 3] : 0.0f;
        const float r2 = m[0] * r + m[1] * g + m[2]  * b + m[3];
        const float g2 = m[4] * r + m[5] * g + m[6]  * b + m[7];
        const float b2 = m[8] * r + m[9] * g + m[10] * b + m[11];
        dst[i * channels + 0] = r2;
        dst[i * channels + 1] = g2;
        dst[i * channels + 2] = b2;
        if (channels == 4) dst[i * channels + 3] = a;
    }
}

} // namespace broimage
