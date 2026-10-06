// Baseline uncompressed TIFF (decode_tiff.h): what chafa sends through
// iTerm2's inline image protocol, which stb_image does not read.

#include "decode_tiff.h"

#include <algorithm>
#include <cstring>

namespace broimage::detail {

namespace {

struct Reader {
    const uint8_t* b;
    std::size_t n;
    bool le;
    uint32_t u16(std::size_t o) const {
        if (o + 2 > n) return 0;
        return le ? uint32_t(b[o]) | uint32_t(b[o + 1]) << 8 : uint32_t(b[o]) << 8 | uint32_t(b[o + 1]);
    }
    uint32_t u32(std::size_t o) const {
        if (o + 4 > n) return 0;
        return le ? u16(o) | u16(o + 2) << 16 : u16(o) << 16 | u16(o + 2);
    }
};

} // namespace

bool is_tiff(const uint8_t* data, std::size_t size) {
    return size >= 4 && (std::memcmp(data, "II*\0", 4) == 0 || std::memcmp(data, "MM\0*", 4) == 0);
}

bool tiff_header(const uint8_t* data, std::size_t size, TiffLayout& out) {
    out = TiffLayout{};
    if (!data || size < 8 || !is_tiff(data, size)) return false;
    const Reader r{data, size, data[0] == 'I'};
    const std::size_t ifd = r.u32(4);
    if (ifd < 8 || ifd + 2 > size) return false;
    const uint32_t entries = r.u16(ifd);
    if (ifd + 2 + std::size_t(entries) * 12 > size) return false;
    uint32_t comp = 1, bps = 8;
    for (uint32_t e = 0; e < entries; ++e) {
        const std::size_t at = ifd + 2 + std::size_t(e) * 12;
        const uint32_t tag = r.u16(at), type = r.u16(at + 2), count = r.u32(at + 4);
        if (type != 3 && type != 4) continue;  // SHORT / LONG only; the tags read here are those
        const uint32_t unit = type == 3 ? 2 : 4;
        if (count == 0 || count > 1u << 20) continue;
        const std::size_t vals = std::size_t(count) * unit <= 4 ? at + 8 : r.u32(at + 8);
        if (vals + std::size_t(count) * unit > size) return false;
        auto val = [&](uint32_t i) { return unit == 2 ? r.u16(vals + i * 2) : r.u32(vals + i * 4); };
        switch (tag) {
            case 256: out.width = val(0); break;
            case 257: out.height = val(0); break;
            case 258: bps = val(0); break;  // every sample the same depth in a baseline file
            case 259: comp = val(0); break;
            case 262: out.photometric = val(0); break;
            case 277: out.samples = val(0); break;
            case 338: out.extra = val(0); break;
            case 273:
                out.offsets.resize(count);
                for (uint32_t i = 0; i < count; ++i) out.offsets[i] = val(i);
                break;
            case 279:
                out.counts.resize(count);
                for (uint32_t i = 0; i < count; ++i) out.counts[i] = val(i);
                break;
            default: break;
        }
    }
    // Uncompressed, 8 bits per sample, chunky; RGB(A) or grayscale(+alpha).
    if (out.width == 0 || out.height == 0 || comp != 1 || bps != 8) return false;
    if (out.photometric == 2 ? (out.samples != 3 && out.samples != 4)
                             : (out.photometric > 1 || (out.samples != 1 && out.samples != 2)))
        return false;
    if (out.offsets.empty() || out.offsets.size() != out.counts.size()) return false;
    return true;
}

bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& l, std::vector<uint8_t>& rgba) {
    const std::size_t pixels = std::size_t(l.width) * l.height;
    const std::size_t need = pixels * l.samples;
    std::vector<uint8_t> raw;
    raw.reserve(need);
    for (std::size_t i = 0; i < l.offsets.size() && raw.size() < need; ++i) {
        if (l.offsets[i] > size || l.counts[i] > size - l.offsets[i]) return false;
        const std::size_t take = std::min<std::size_t>(l.counts[i], need - raw.size());
        raw.insert(raw.end(), data + l.offsets[i], data + l.offsets[i] + take);
    }
    if (raw.size() < need) return false;
    rgba.resize(pixels * 4);
    const bool gray = l.photometric <= 1;
    const bool alpha = l.samples == 2 || l.samples == 4;
    for (std::size_t i = 0; i < pixels; ++i) {
        const uint8_t* s = raw.data() + i * l.samples;
        uint8_t* d = rgba.data() + i * 4;
        const uint8_t a = alpha ? s[l.samples - 1] : 255;
        for (int c = 0; c < 3; ++c) {
            uint8_t v = gray ? s[0] : s[c];
            if (gray && l.photometric == 0) v = uint8_t(255 - v);  // WhiteIsZero
            // ExtraSamples 1: associated (premultiplied) alpha; straighten it.
            if (alpha && l.extra == 1 && a) v = uint8_t(std::min(255, (int(v) * 255 + a / 2) / a));
            d[c] = v;
        }
        d[3] = a;
    }
    return true;
}

} // namespace broimage::detail
