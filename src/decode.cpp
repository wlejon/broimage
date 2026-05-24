#define _CRT_SECURE_NO_WARNINGS // for fopen in the EXIF sniff path

#include "broimage/decode.h"

#include "broimage/buffer.h"

#include <stb_image.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace broimage {

namespace {

void set_fallback(Image& out) {
    out.width = 1;
    out.height = 1;
    out.channels = 4;
    out.pixels = { 255, 255, 255, 255 };
}

// ---- EXIF orientation reader ----------------------------------------------
//
// JPEG: SOI (FF D8), then a series of markers. APP1 (FF E1) may contain an
// EXIF block ("Exif\0\0" + a TIFF header). The orientation tag is tag 0x0112
// in IFD0; we only need that one field.

inline uint16_t read_u16(const uint8_t* p, bool be) {
    return be ? static_cast<uint16_t>((p[0] << 8) | p[1])
              : static_cast<uint16_t>((p[1] << 8) | p[0]);
}
inline uint32_t read_u32(const uint8_t* p, bool be) {
    return be ? (uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 |
                 uint32_t(p[2]) << 8  | uint32_t(p[3]))
              : (uint32_t(p[3]) << 24 | uint32_t(p[2]) << 16 |
                 uint32_t(p[1]) << 8  | uint32_t(p[0]));
}

ExifOrientation parse_exif_payload(const uint8_t* exif, std::size_t exif_size) {
    // exif points at the TIFF header (right after "Exif\0\0").
    if (exif_size < 8) return ExifOrientation::Normal;
    const bool be = (exif[0] == 'M' && exif[1] == 'M');
    const bool le = (exif[0] == 'I' && exif[1] == 'I');
    if (!be && !le) return ExifOrientation::Normal;
    if (read_u16(exif + 2, be) != 0x002A) return ExifOrientation::Normal;
    const uint32_t ifd0_off = read_u32(exif + 4, be);
    if (std::size_t(ifd0_off) + 2 > exif_size) return ExifOrientation::Normal;
    const uint16_t n_entries = read_u16(exif + ifd0_off, be);
    const std::size_t entries_off = std::size_t(ifd0_off) + 2;
    if (entries_off + std::size_t(n_entries) * 12 > exif_size) {
        return ExifOrientation::Normal;
    }
    for (uint16_t i = 0; i < n_entries; ++i) {
        const uint8_t* e = exif + entries_off + std::size_t(i) * 12;
        const uint16_t tag  = read_u16(e + 0, be);
        const uint16_t type = read_u16(e + 2, be);
        if (tag == 0x0112 && type == 3) { // SHORT
            // Value lives in the first 2 bytes of the "value offset" field.
            const uint16_t v = read_u16(e + 8, be);
            if (v >= 1 && v <= 8) return static_cast<ExifOrientation>(v);
            return ExifOrientation::Normal;
        }
    }
    return ExifOrientation::Normal;
}

void mirror_h_inplace(Image& img) {
    const int w = img.width, h = img.height, c = img.channels;
    for (int y = 0; y < h; ++y) {
        uint8_t* row = img.pixels.data() + std::size_t(y) * w * c;
        for (int x = 0; x < w / 2; ++x) {
            uint8_t* a = row + std::size_t(x) * c;
            uint8_t* b = row + std::size_t(w - 1 - x) * c;
            for (int k = 0; k < c; ++k) std::swap(a[k], b[k]);
        }
    }
}

void mirror_v_inplace(Image& img) {
    const int w = img.width, h = img.height, c = img.channels;
    const std::size_t row_bytes = std::size_t(w) * c;
    std::vector<uint8_t> tmp(row_bytes);
    for (int y = 0; y < h / 2; ++y) {
        uint8_t* a = img.pixels.data() + std::size_t(y) * row_bytes;
        uint8_t* b = img.pixels.data() + std::size_t(h - 1 - y) * row_bytes;
        std::memcpy(tmp.data(), a, row_bytes);
        std::memcpy(a, b, row_bytes);
        std::memcpy(b, tmp.data(), row_bytes);
    }
}

// Build a new buffer by mapping each dst (x, y) to a src coordinate.
template <typename Map>
void remap_into(Image& img, int new_w, int new_h, Map map) {
    const int c = img.channels;
    std::vector<uint8_t> out(std::size_t(new_w) * new_h * c);
    for (int y = 0; y < new_h; ++y) {
        for (int x = 0; x < new_w; ++x) {
            int sx, sy;
            map(x, y, sx, sy);
            const uint8_t* sp = img.pixels.data() +
                                (std::size_t(sy) * img.width + sx) * c;
            uint8_t* dp = out.data() + (std::size_t(y) * new_w + x) * c;
            for (int k = 0; k < c; ++k) dp[k] = sp[k];
        }
    }
    img.width = new_w;
    img.height = new_h;
    img.pixels = std::move(out);
}

} // namespace

ExifOrientation read_exif_orientation(const uint8_t* data, std::size_t size) {
    if (!data || size < 4) return ExifOrientation::Normal;
    if (data[0] != 0xFF || data[1] != 0xD8) return ExifOrientation::Normal; // SOI
    std::size_t i = 2;
    while (i + 4 <= size) {
        if (data[i] != 0xFF) return ExifOrientation::Normal;
        while (i + 1 < size && data[i + 1] == 0xFF) ++i; // fill bytes
        if (i + 4 > size) break;
        const uint8_t marker = data[i + 1];
        if (marker == 0xDA) return ExifOrientation::Normal; // SOS: scan data starts
        if (marker == 0xD8 || marker == 0xD9 ||
            (marker >= 0xD0 && marker <= 0xD7)) {
            i += 2;
            continue;
        }
        const uint16_t seg_len = static_cast<uint16_t>(
            (data[i + 2] << 8) | data[i + 3]);
        if (seg_len < 2 || i + 2 + std::size_t(seg_len) > size) {
            return ExifOrientation::Normal;
        }
        if (marker == 0xE1 && seg_len >= 8 &&
            data[i + 4] == 'E' && data[i + 5] == 'x' &&
            data[i + 6] == 'i' && data[i + 7] == 'f' &&
            data[i + 8] == 0x00 && data[i + 9] == 0x00) {
            const uint8_t* exif = data + i + 10;
            const std::size_t exif_size = std::size_t(seg_len) - 8;
            return parse_exif_payload(exif, exif_size);
        }
        i += 2 + std::size_t(seg_len);
    }
    return ExifOrientation::Normal;
}

ExifOrientation read_exif_orientation_file(const std::string& path) {
    // 64 KB is more than enough — phone JPEGs put APP1 right after SOI.
    constexpr std::size_t k_sniff = 64 * 1024;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return ExifOrientation::Normal;
    std::vector<uint8_t> buf(k_sniff);
    const std::size_t got = std::fread(buf.data(), 1, k_sniff, f);
    std::fclose(f);
    return read_exif_orientation(buf.data(), got);
}

void apply_exif_orientation(Image& img, ExifOrientation orient) {
    if (img.empty() || orient == ExifOrientation::Normal) return;
    const int W = img.width, H = img.height;
    switch (orient) {
        case ExifOrientation::Normal: return;
        case ExifOrientation::FlipH:     mirror_h_inplace(img); return;
        case ExifOrientation::Rotate180:
            mirror_h_inplace(img); mirror_v_inplace(img); return;
        case ExifOrientation::FlipV:     mirror_v_inplace(img); return;
        case ExifOrientation::Transpose:
            remap_into(img, H, W, [](int x, int y, int& sx, int& sy) {
                sx = y; sy = x;
            });
            return;
        case ExifOrientation::Rotate90CW:
            remap_into(img, H, W, [H](int x, int y, int& sx, int& sy) {
                sx = y; sy = H - 1 - x;
            });
            return;
        case ExifOrientation::Transverse:
            remap_into(img, H, W, [H, W](int x, int y, int& sx, int& sy) {
                sx = W - 1 - y; sy = H - 1 - x;
            });
            return;
        case ExifOrientation::Rotate90CCW:
            remap_into(img, H, W, [W](int x, int y, int& sx, int& sy) {
                sx = W - 1 - y; sy = x;
            });
            return;
    }
}

bool decode_file(const std::string& path, Image& out, std::string* error) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_load failed";
        set_fallback(out);
        return false;
    }
    out.width = w;
    out.height = h;
    out.channels = 4;
    out.pixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool decode_memory(const uint8_t* data, std::size_t size, Image& out,
                   std::string* error) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load_from_memory(data, static_cast<int>(size),
                                                  &w, &h, &channels, 4);
    if (!pixels) {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_load_from_memory failed";
        set_fallback(out);
        return false;
    }
    out.width = w;
    out.height = h;
    out.channels = 4;
    out.pixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool probe_dimensions_memory(const uint8_t* data, std::size_t size,
                             int* width, int* height, int* channels) {
    int w = 0, h = 0, c = 0;
    if (!stbi_info_from_memory(data, static_cast<int>(size), &w, &h, &c))
        return false;
    if (width)    *width    = w;
    if (height)   *height   = h;
    if (channels) *channels = c;
    return true;
}

// ---- 16-bit / HDR decode ---------------------------------------------------

bool decode_file_u16(const std::string& path, ImageU16& out, std::string* error) {
    int w = 0, h = 0, channels = 0;
    stbi_us* pixels = stbi_load_16(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_load_16 failed";
        out = {};
        return false;
    }
    out.width = w; out.height = h; out.channels = 4;
    out.pixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool decode_memory_u16(const uint8_t* data, std::size_t size, ImageU16& out,
                       std::string* error) {
    int w = 0, h = 0, channels = 0;
    stbi_us* pixels = stbi_load_16_from_memory(data, static_cast<int>(size),
                                               &w, &h, &channels, 4);
    if (!pixels) {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_load_16_from_memory failed";
        out = {};
        return false;
    }
    out.width = w; out.height = h; out.channels = 4;
    out.pixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool decode_file_f32(const std::string& path, ImageF32& out, std::string* error) {
    int w = 0, h = 0, channels = 0;
    float* pixels = stbi_loadf(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_loadf failed";
        out = {};
        return false;
    }
    out.width = w; out.height = h; out.channels = 4;
    out.pixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool decode_memory_f32(const uint8_t* data, std::size_t size, ImageF32& out,
                       std::string* error) {
    int w = 0, h = 0, channels = 0;
    float* pixels = stbi_loadf_from_memory(data, static_cast<int>(size),
                                           &w, &h, &channels, 4);
    if (!pixels) {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_loadf_from_memory failed";
        out = {};
        return false;
    }
    out.width = w; out.height = h; out.channels = 4;
    out.pixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool decode_file_oriented(const std::string& path, Image& out,
                          std::string* error) {
    const ExifOrientation orient = read_exif_orientation_file(path);
    if (!decode_file(path, out, error)) return false;
    apply_exif_orientation(out, orient);
    return true;
}

bool decode_memory_oriented(const uint8_t* data, std::size_t size, Image& out,
                            std::string* error) {
    const ExifOrientation orient = read_exif_orientation(data, size);
    if (!decode_memory(data, size, out, error)) return false;
    apply_exif_orientation(out, orient);
    return true;
}

} // namespace broimage
