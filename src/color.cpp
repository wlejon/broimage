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

} // namespace broimage
