#include "broimage/decode.h"

#include "broimage/buffer.h"

#include <stb_image.h>

#include "decode_ico.h"
#include "decode_tiff.h"
#include "file_bytes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace broimage {

namespace {

// ---- broimage's own decoders ------------------------------------------------
//
// Formats stb_image has no reader for, tried before it. Each is recognised
// by its signature; one that recognises the bytes but cannot decode them
// still lets stb try, and its reason is the one reported if stb fails too.
// A new format is a row in kOwn.

struct OwnDecoder {
    bool (*sniff)(const uint8_t*, std::size_t);
    bool (*probe)(const uint8_t*, std::size_t, int& w, int& h, int& channels);
    bool (*decode8)(const uint8_t*, std::size_t, Image& out, std::string* why);
    bool (*decode16)(const uint8_t*, std::size_t, ImageU16& out, std::string* why);  // may be null
};

bool ico_probe(const uint8_t* d, std::size_t n, int& w, int& h, int& c) {
    c = 4;
    return detail::ico_probe(d, n, w, h);
}

bool ico_decode(const uint8_t* d, std::size_t n, Image& out, std::string* why) {
    int w = 0, h = 0;
    std::vector<uint8_t> rgba;
    if (!detail::ico_decode(d, n, w, h, rgba)) {
        if (why) *why = "ICO image could not be decoded";
        return false;
    }
    out.width = w;
    out.height = h;
    out.channels = 4;
    out.pixels = std::move(rgba);
    return true;
}

bool tiff_probe(const uint8_t* d, std::size_t n, int& w, int& h, int& c) {
    detail::TiffLayout l;
    if (!detail::tiff_layout(d, n, l)) return false;
    w = int(l.width);
    h = int(l.height);
    c = l.channels();
    return true;
}

template <class Img>
bool tiff_decode_into(const uint8_t* d, std::size_t n, Img& out, std::string* why) {
    detail::TiffLayout l;
    if (!detail::tiff_layout(d, n, l, why) || !detail::tiff_decode(d, n, l, out.pixels, why)) return false;
    out.width = int(l.width);
    out.height = int(l.height);
    out.channels = 4;
    return true;
}

bool tiff_decode8(const uint8_t* d, std::size_t n, Image& out, std::string* why) {
    return tiff_decode_into(d, n, out, why);
}

bool tiff_decode16(const uint8_t* d, std::size_t n, ImageU16& out, std::string* why) {
    return tiff_decode_into(d, n, out, why);
}

const OwnDecoder kOwn[] = {
    {detail::is_ico, ico_probe, ico_decode, nullptr},
    {detail::is_tiff, tiff_probe, tiff_decode8, tiff_decode16},
};

const OwnDecoder* own_decoder(const uint8_t* data, std::size_t size) {
    if (!data) return nullptr;
    for (const OwnDecoder& d : kOwn)
        if (d.sniff(data, size)) return &d;
    return nullptr;
}

bool fits_int(std::size_t size) { return size <= std::size_t(std::numeric_limits<int>::max()); }

template <class Img>
bool fail_decode(Img& out, std::string* error, const std::string& own_why, const char* stb_default) {
    if (error) {
        const char* r = stbi_failure_reason();
        *error = !own_why.empty() ? own_why : (r ? r : stb_default);
    }
    out = Img{};
    if constexpr (std::is_same_v<Img, Image>) out.channels = 0;
    return false;
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
    std::vector<uint8_t> bytes;
    if (!detail::read_file_bytes(path, bytes)) {
        if (error) *error = "can't fopen";
        out = Image{};
        out.channels = 0;
        return false;
    }
    return decode_memory(bytes.data(), bytes.size(), out, error);
}

bool decode_memory(const uint8_t* data, std::size_t size, Image& out,
                   std::string* error) {
    std::string own_why;
    if (const OwnDecoder* own = own_decoder(data, size)) {
        if (own->decode8(data, size, out, &own_why)) return true;
    }
    if (!data || !fits_int(size)) return fail_decode(out, error, own_why, "no data");
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load_from_memory(data, static_cast<int>(size),
                                                  &w, &h, &channels, 4);
    if (!pixels) return fail_decode(out, error, own_why, "stbi_load_from_memory failed");
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
    bool ok = false;
    if (const OwnDecoder* own = own_decoder(data, size)) ok = own->probe(data, size, w, h, c);
    if (!ok && (!data || !fits_int(size) || !stbi_info_from_memory(data, static_cast<int>(size), &w, &h, &c)))
        return false;
    if (width)    *width    = w;
    if (height)   *height   = h;
    if (channels) *channels = c;
    return true;
}

// ---- 16-bit / HDR decode ---------------------------------------------------

bool decode_file_u16(const std::string& path, ImageU16& out, std::string* error) {
    std::vector<uint8_t> bytes;
    if (!detail::read_file_bytes(path, bytes)) {
        if (error) *error = "can't fopen";
        out = {};
        return false;
    }
    return decode_memory_u16(bytes.data(), bytes.size(), out, error);
}

bool decode_memory_u16(const uint8_t* data, std::size_t size, ImageU16& out,
                       std::string* error) {
    std::string own_why;
    if (const OwnDecoder* own = own_decoder(data, size); own && own->decode16) {
        if (own->decode16(data, size, out, &own_why)) return true;
    }
    if (!data || !fits_int(size)) return fail_decode(out, error, own_why, "no data");
    int w = 0, h = 0, channels = 0;
    stbi_us* pixels = stbi_load_16_from_memory(data, static_cast<int>(size),
                                               &w, &h, &channels, 4);
    if (!pixels) return fail_decode(out, error, own_why, "stbi_load_16_from_memory failed");
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
    std::vector<uint8_t> bytes;
    if (!detail::read_file_bytes(path, bytes)) {
        if (error) *error = "can't fopen";
        out = Image{};
        out.channels = 0;
        return false;
    }
    return decode_memory_oriented(bytes.data(), bytes.size(), out, error);
}

bool decode_memory_oriented(const uint8_t* data, std::size_t size, Image& out,
                            std::string* error) {
    const ExifOrientation orient = read_exif_orientation(data, size);
    if (!decode_memory(data, size, out, error)) return false;
    apply_exif_orientation(out, orient);
    return true;
}

} // namespace broimage
