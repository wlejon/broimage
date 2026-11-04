// TIFF samples to RGBA (tiff_convert.h).

#include "tiff_convert.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace broimage::detail {

namespace {

template <class S, class T>
void convert(const TiffLayout& l, const S* in, T* out) {
    const uint64_t in_max = (uint64_t(1) << l.bits) - 1;
    const uint64_t out_max = sizeof(T) == 1 ? 255 : 65535;
    auto scale = [&](uint64_t v) { return T((v * out_max + in_max / 2) / in_max); };
    const std::size_t pixels = std::size_t(l.width) * l.height;
    const uint32_t spp = l.samples;
    const std::size_t palette_n = std::size_t(1) << l.bits;
    for (std::size_t i = 0; i < pixels; ++i) {
        const S* s = in + i * spp;
        T* d = out + i * 4;
        const uint64_t a = l.alpha >= 0 ? s[l.alpha] : in_max;
        // Associated alpha: the colour samples were multiplied by it.
        auto straight = [&](uint64_t c) {
            if (!l.premultiplied || l.alpha < 0 || a == 0) return c;
            return std::min(in_max, (c * in_max + a / 2) / a);
        };
        switch (l.photometric) {
            case 0:    // WhiteIsZero
            case 1: {  // BlackIsZero
                uint64_t g = straight(s[0]);
                if (l.photometric == 0) g = in_max - std::min<uint64_t>(g, in_max);
                d[0] = d[1] = d[2] = scale(g);
                break;
            }
            case 3: {  // palette: 16-bit colours
                const std::size_t k = s[0];
                for (int c = 0; c < 3; ++c) {
                    const uint64_t v = k < palette_n ? l.colormap[std::size_t(c) * palette_n + k] : 0;
                    d[c] = T((v * out_max + 32767) / 65535);
                }
                break;
            }
            case 5: {  // CMYK
                const uint64_t k = in_max - straight(s[3]);
                for (int c = 0; c < 3; ++c) {
                    const uint64_t v = (in_max - straight(s[c])) * k;
                    d[c] = T((v * out_max + (in_max * in_max) / 2) / (in_max * in_max));
                }
                break;
            }
            default:  // RGB, and YCbCr the JPEG decoder already turned into RGB
                for (int c = 0; c < 3; ++c) d[c] = scale(straight(s[c]));
                break;
        }
        d[3] = scale(a);
    }
}

template <class S>
void convert_uint_to_f32(const TiffLayout& l, const S* in, float* out) {
    const double in_max = double((uint64_t(1) << l.bits) - 1);
    const std::size_t pixels = std::size_t(l.width) * l.height;
    const uint32_t spp = l.samples;
    const std::size_t palette_n = std::size_t(1) << l.bits;
    for (std::size_t i = 0; i < pixels; ++i) {
        const S* s = in + i * spp;
        float* d = out + i * 4;
        const double a = l.alpha >= 0 ? double(s[l.alpha]) / in_max : 1.0;
        auto straight = [&](double c) {
            if (!l.premultiplied || l.alpha < 0 || a <= 0.0) return c;
            return std::min(1.0, c / a);
        };
        switch (l.photometric) {
            case 0:
            case 1: {
                double g = straight(double(s[0]) / in_max);
                if (l.photometric == 0) g = 1.0 - std::clamp(g, 0.0, 1.0);
                d[0] = d[1] = d[2] = float(g);
                break;
            }
            case 3: {
                const std::size_t k = s[0];
                for (int c = 0; c < 3; ++c) {
                    const double v = k < palette_n ? l.colormap[std::size_t(c) * palette_n + k] : 0.0;
                    d[c] = float(v / 65535.0);
                }
                break;
            }
            case 5: {
                const double k = 1.0 - straight(double(s[3]) / in_max);
                for (int c = 0; c < 3; ++c) {
                    d[c] = float((1.0 - straight(double(s[c]) / in_max)) * k);
                }
                break;
            }
            default:
                for (int c = 0; c < 3; ++c) d[c] = float(straight(double(s[c]) / in_max));
                break;
        }
        d[3] = float(a);
    }
}

void convert_f32_to_f32(const TiffLayout& l, const float* in, float* out) {
    const std::size_t pixels = std::size_t(l.width) * l.height;
    const uint32_t spp = l.samples;
    for (std::size_t i = 0; i < pixels; ++i) {
        const float* s = in + i * spp;
        float* d = out + i * 4;
        const float a = l.alpha >= 0 ? s[l.alpha] : 1.0f;
        auto straight = [&](float c) {
            if (!l.premultiplied || l.alpha < 0 || a == 0.0f) return c;
            return c / a;
        };
        switch (l.photometric) {
            case 0:
            case 1: {
                float g = straight(s[0]);
                if (l.photometric == 0 && l.sample_format == 1) g = 1.0f - g;
                d[0] = d[1] = d[2] = g;
                break;
            }
            case 5: {
                const float k = 1.0f - straight(s[3]);
                for (int c = 0; c < 3; ++c) d[c] = (1.0f - straight(s[c])) * k;
                break;
            }
            default:
                for (int c = 0; c < 3; ++c) d[c] = straight(s[c]);
                break;
        }
        d[3] = a;
    }
}

template <class T>
void convert_f32_to_int(const TiffLayout& l, const float* in, T* out) {
    const double out_max = sizeof(T) == 1 ? 255.0 : 65535.0;
    double norm_div = 1.0;
    if (l.sample_format == 1 && l.bits == 32) norm_div = double(0xFFFFFFFFull);
    else if (l.sample_format == 2) norm_div = double((uint64_t(1) << (l.bits - 1)) - 1);

    auto quant = [&](float v) -> T {
        if (!std::isfinite(v)) return 0;
        const double u = std::clamp(double(v) / norm_div, 0.0, 1.0);
        return T(std::llround(u * out_max));
    };

    const std::size_t pixels = std::size_t(l.width) * l.height;
    const uint32_t spp = l.samples;
    for (std::size_t i = 0; i < pixels; ++i) {
        const float* s = in + i * spp;
        T* d = out + i * 4;
        const float a = l.alpha >= 0 ? s[l.alpha] : float(norm_div);
        switch (l.photometric) {
            case 0:
            case 1: {
                T g = quant(s[0]);
                if (l.photometric == 0) g = T(out_max) - g;
                d[0] = d[1] = d[2] = g;
                break;
            }
            case 5: {
                const double k = 1.0 - std::clamp(double(s[3]) / norm_div, 0.0, 1.0);
                for (int c = 0; c < 3; ++c) {
                    const double v = (1.0 - std::clamp(double(s[c]) / norm_div, 0.0, 1.0)) * k;
                    d[c] = T(std::llround(v * out_max));
                }
                break;
            }
            default:
                for (int c = 0; c < 3; ++c) d[c] = quant(s[c]);
                break;
        }
        d[3] = l.alpha >= 0 ? quant(a) : T(out_max);
    }
}

} // namespace

void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, uint8_t* rgba) { convert(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, uint16_t* rgba) { convert(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, float* rgba) { convert_uint_to_f32(l, samples, rgba); }

void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, uint8_t* rgba) { convert(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, uint16_t* rgba) { convert(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, float* rgba) { convert_uint_to_f32(l, samples, rgba); }

void tiff_to_rgba(const TiffLayout& l, const float* samples, uint8_t* rgba) { convert_f32_to_int(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const float* samples, uint16_t* rgba) { convert_f32_to_int(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const float* samples, float* rgba) { convert_f32_to_f32(l, samples, rgba); }

} // namespace broimage::detail
