// TIFF samples to RGBA (tiff_convert.h).

#include "tiff_convert.h"

#include <algorithm>
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

} // namespace

void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, uint8_t* rgba) { convert(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, uint16_t* rgba) { convert(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, uint8_t* rgba) { convert(l, samples, rgba); }
void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, uint16_t* rgba) { convert(l, samples, rgba); }

} // namespace broimage::detail
