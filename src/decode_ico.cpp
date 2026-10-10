#include "decode_ico.h"

#include <stb_image.h>

#include <cstring>

namespace broimage::detail {

namespace {

uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

constexpr uint8_t kPngSig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
constexpr int kMaxSide = 1 << 14;  // far beyond any real icon; bounds the allocation

struct Entry {
    const uint8_t* data = nullptr;
    std::size_t size = 0;
    bool png = false;
    int width = 0, height = 0, bpp = 0;
};

// The DIB of one entry: its header fields, checked against the bytes present.
struct Dib {
    int width = 0, height = 0;  // height of the colour image (half the stored one)
    int bpp = 0;
    uint32_t headerSize = 0, compression = 0, colors = 0;
};

bool read_dib(const uint8_t* p, std::size_t n, Dib& d) {
    if (n < 40) return false;
    d.headerSize = rd32(p);
    if (d.headerSize < 40 || d.headerSize > n) return false;
    const int32_t w = int32_t(rd32(p + 4));
    const int32_t h2 = int32_t(rd32(p + 8));
    d.bpp = rd16(p + 14);
    d.compression = rd32(p + 16);
    d.colors = rd32(p + 32);
    // Icons are stored bottom-up with the AND mask's rows counted in.
    if (w <= 0 || w > kMaxSide || h2 <= 0 || h2 / 2 > kMaxSide) return false;
    d.width = w;
    d.height = h2 / 2;
    if (d.height <= 0) return false;
    if (d.bpp != 1 && d.bpp != 4 && d.bpp != 8 && d.bpp != 24 && d.bpp != 32) return false;
    // BI_RGB, or BI_BITFIELDS with the standard BGRA masks for 32 bpp.
    if (d.compression != 0 && !(d.compression == 3 && d.bpp == 32)) return false;
    return true;
}

bool entry_size(Entry& e) {
    if (e.size >= 8 && std::memcmp(e.data, kPngSig, 8) == 0) {
        int w = 0, h = 0, c = 0;
        if (!stbi_info_from_memory(e.data, int(e.size), &w, &h, &c)) return false;
        e.png = true;
        e.width = w;
        e.height = h;
        e.bpp = 32;
        return true;
    }
    Dib d;
    if (!read_dib(e.data, e.size, d)) return false;
    e.width = d.width;
    e.height = d.height;
    e.bpp = d.bpp;
    return true;
}

// The best entry: the most pixels, ties going to the deepest colour. The
// directory's own width/height bytes stop at 256 (0 means 256) and are not
// always right, so the size compared is the embedded image's.
//
// `partial` is the probe's case: it may hold only the head of the file, so an
// entry whose bytes run past the end is sized from what is there (a PNG's or
// DIB's header), or failing that from the directory, rather than skipped —
// skipping it would report a smaller entry than the decode then produces.
bool best_entry(const uint8_t* data, std::size_t size, Entry& best, bool partial) {
    if (!is_ico(data, size)) return false;
    const unsigned count = rd16(data + 4);
    bool found = false;
    for (unsigned i = 0; i < count; ++i) {
        const uint8_t* dir = data + 6 + 16 * i;
        const uint32_t len = rd32(dir + 8);
        const uint32_t off = rd32(dir + 12);
        if (len == 0 || off < 6 + 16u * count) continue;
        Entry e;
        const bool complete = off < size && len <= size - off;
        if (complete) {
            e.data = data + off;
            e.size = len;
            if (!entry_size(e)) continue;
        } else {
            if (!partial) continue;
            e.data = off < size ? data + off : nullptr;
            e.size = off < size ? size - off : 0;
            if (!e.data || !entry_size(e)) {
                e.width = dir[0] ? dir[0] : 256;
                e.height = dir[1] ? dir[1] : 256;
                e.bpp = rd16(dir + 6);
            }
        }
        const long long px = (long long)e.width * e.height;
        const long long bestPx = (long long)best.width * best.height;
        if (!found || px > bestPx || (px == bestPx && e.bpp > best.bpp)) {
            best = e;
            found = true;
        }
    }
    return found;
}

bool decode_dib(const uint8_t* p, std::size_t n, int& outW, int& outH,
                std::vector<uint8_t>& rgba) {
    Dib d;
    if (!read_dib(p, n, d)) return false;
    const int W = d.width, H = d.height;

    std::size_t pos = d.headerSize;
    // BI_BITFIELDS in a 40-byte header puts its three masks after it.
    if (d.compression == 3 && d.headerSize == 40) pos += 12;

    const uint8_t* palette = nullptr;
    uint32_t paletteCount = 0;
    if (d.bpp <= 8) {
        paletteCount = d.colors ? d.colors : (1u << d.bpp);
        if (paletteCount > 256) return false;
        if (pos + std::size_t(paletteCount) * 4 > n) return false;
        palette = p + pos;
        pos += std::size_t(paletteCount) * 4;
    }

    const std::size_t xorStride = ((std::size_t(W) * d.bpp + 31) / 32) * 4;
    const std::size_t andStride = ((std::size_t(W) + 31) / 32) * 4;
    if (pos + xorStride * H > n) return false;
    const uint8_t* xorBits = p + pos;
    pos += xorStride * H;
    // Some writers leave the mask off a 32-bit image; its alpha is enough then.
    const uint8_t* andBits = (pos + andStride * H <= n) ? p + pos : nullptr;
    if (!andBits && d.bpp != 32) return false;

    rgba.assign(std::size_t(W) * H * 4, 0);
    bool anyAlpha = false;
    for (int y = 0; y < H; ++y) {
        const uint8_t* row = xorBits + xorStride * (H - 1 - y);  // bottom-up
        uint8_t* out = rgba.data() + std::size_t(y) * W * 4;
        for (int x = 0; x < W; ++x) {
            uint8_t r = 0, g = 0, b = 0, a = 255;
            if (d.bpp == 32) {
                b = row[x * 4]; g = row[x * 4 + 1]; r = row[x * 4 + 2]; a = row[x * 4 + 3];
                if (a) anyAlpha = true;
            } else if (d.bpp == 24) {
                b = row[x * 3]; g = row[x * 3 + 1]; r = row[x * 3 + 2];
            } else {
                unsigned idx;
                if (d.bpp == 8) idx = row[x];
                else if (d.bpp == 4) idx = (row[x >> 1] >> ((x & 1) ? 0 : 4)) & 0x0F;
                else idx = (row[x >> 3] >> (7 - (x & 7))) & 1;
                if (idx < paletteCount) {
                    b = palette[idx * 4]; g = palette[idx * 4 + 1]; r = palette[idx * 4 + 2];
                }
            }
            out[x * 4] = r; out[x * 4 + 1] = g; out[x * 4 + 2] = b; out[x * 4 + 3] = a;
        }
    }

    // Transparency: a 32-bit image carries its own alpha; the AND mask (1 =
    // transparent) decides it for everything else, and for a 32-bit image
    // whose alpha channel is all zero (an old-style icon that never used it).
    if (andBits && (d.bpp != 32 || !anyAlpha)) {
        for (int y = 0; y < H; ++y) {
            const uint8_t* row = andBits + andStride * (H - 1 - y);
            uint8_t* out = rgba.data() + std::size_t(y) * W * 4;
            for (int x = 0; x < W; ++x) {
                const bool transparent = (row[x >> 3] >> (7 - (x & 7))) & 1;
                out[x * 4 + 3] = transparent ? 0 : 255;
            }
        }
    }
    outW = W;
    outH = H;
    return true;
}

} // namespace

bool is_ico(const uint8_t* data, std::size_t size) {
    if (!data || size < 6) return false;
    const uint16_t reserved = rd16(data), type = rd16(data + 2), count = rd16(data + 4);
    // type 1 = icon, 2 = cursor (same layout; hotspot where the icon has planes/bpp).
    if (reserved != 0 || (type != 1 && type != 2) || count == 0) return false;
    return size >= 6 + std::size_t(count) * 16;
}

bool ico_probe(const uint8_t* data, std::size_t size, int& width, int& height) {
    Entry e;
    if (!best_entry(data, size, e, /*partial=*/true)) return false;
    width = e.width;
    height = e.height;
    return true;
}

bool ico_decode(const uint8_t* data, std::size_t size, int& width, int& height,
                std::vector<uint8_t>& rgba) {
    Entry e;
    if (!best_entry(data, size, e, /*partial=*/false)) return false;
    if (e.png) {
        int w = 0, h = 0, c = 0;
        unsigned char* px = stbi_load_from_memory(e.data, int(e.size), &w, &h, &c, 4);
        if (!px) return false;
        rgba.assign(px, px + std::size_t(w) * h * 4);
        stbi_image_free(px);
        width = w;
        height = h;
        return true;
    }
    return decode_dib(e.data, e.size, width, height, rgba);
}

} // namespace broimage::detail
