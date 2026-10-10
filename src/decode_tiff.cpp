// The TIFF image decoder (decode_tiff.h): page 0's layout from its IFD, each
// chunk decompressed (tiff_codecs) and unpacked into one sample buffer, then
// converted to RGBA (tiff_convert).

#include "decode_tiff.h"

#include "tiff_codecs.h"
#include "tiff_convert.h"
#include "tiff_ifd.h"

#include <algorithm>
#include <cstring>

namespace broimage::detail {

namespace {

// Size caps, checked before anything is allocated: 2^28 pixels (a 16384 x
// 16384 image; 1 GiB of RGBA8), and a compressed image may not claim more
// than 4096 bytes of samples per byte of file (LZW peaks near 1400:1,
// Deflate at 1032:1), so a few hundred hostile bytes cannot ask for a
// gigabyte.
constexpr uint64_t kMaxPixels = uint64_t(1) << 28;
constexpr uint64_t kMaxSide = uint64_t(1) << 24;
constexpr uint64_t kMaxExpansion = 4096;

enum Tag : uint16_t {
    kWidth = 256, kHeight = 257, kBitsPerSample = 258, kCompression = 259, kPhotometric = 262,
    kFillOrder = 266, kStripOffsets = 273, kSamplesPerPixel = 277, kRowsPerStrip = 278,
    kStripByteCounts = 279, kPlanarConfig = 284, kPredictor = 317, kColorMap = 320, kTileWidth = 322,
    kTileLength = 323, kTileOffsets = 324, kTileByteCounts = 325, kInkSet = 332, kExtraSamples = 338,
    kSampleFormat = 339, kJpegTables = 347,
};

bool fail(std::string* why, const char* msg) {
    if (why) *why = msg;
    return false;
}

uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

struct Grid {
    uint32_t across, down, planes, cs;  // chunks per row / column, planes, samples per chunk pixel
    uint64_t row_bytes;                 // of a stored chunk row
};

Grid grid_of(const TiffLayout& l) {
    Grid g;
    g.across = uint32_t(ceil_div(l.width, l.chunk_w));
    g.down = uint32_t(ceil_div(l.height, l.chunk_h));
    g.planes = l.planar == 2 ? l.samples : 1;
    g.cs = l.planar == 2 ? 1 : l.samples;
    g.row_bytes = ceil_div(uint64_t(l.chunk_w) * g.cs * l.bits, 8);
    return g;
}

// Rows stored in chunk row `cy`: a tile is always whole, the last strip
// holds only the rows left.
uint32_t stored_rows(const TiffLayout& l, uint32_t cy) {
    if (l.tiled) return l.chunk_h;
    return std::min<uint32_t>(l.chunk_h, l.height - cy * l.chunk_h);
}

bool read_array(const TiffReader& r, const TiffEntry* e, std::size_t n, std::vector<uint32_t>& out) {
    if (!e || e->count < n) return false;
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!r.uint_at(*e, uint32_t(i), out[i])) return false;
    return true;
}

template <class S>
bool decode_samples(const uint8_t* data, std::size_t size, const TiffLayout& l, std::vector<S>& samples,
                    std::string* why) {
    const Grid g = grid_of(l);
    const std::size_t W = l.width, spp = l.samples;
    samples.assign(W * l.height * spp, S(0));
    std::vector<uint8_t> raw;
    std::vector<uint32_t> row(std::size_t(l.chunk_w) * g.cs);
    const uint32_t mask = l.bits >= 32 ? 0xFFFFFFFFu : (1u << l.bits) - 1;
    for (uint32_t p = 0; p < g.planes; ++p) {
        for (uint32_t cy = 0; cy < g.down; ++cy) {
            for (uint32_t cx = 0; cx < g.across; ++cx) {
                const std::size_t idx = (std::size_t(p) * g.down + cy) * g.across + cx;
                const uint32_t x0 = cx * l.chunk_w, y0 = cy * l.chunk_h;
                const uint32_t vis_w = std::min(l.chunk_w, l.width - x0);
                const uint32_t vis_h = std::min(l.chunk_h, l.height - y0);
                const uint32_t off = l.offsets[idx], cnt = l.counts[idx];
                if (off > size || cnt > size - off) return fail(why, "a TIFF chunk lies outside the file");
                const uint8_t* src = data + off;

                if (l.compression == kTiffJpeg) {
                    std::vector<uint8_t> px;
                    int jw = 0, jh = 0;
                    if (!tiff_jpeg(l.jpeg_tables, l.jpeg_tables_n, src, cnt, l.photometric == 2, int(g.cs), px, jw,
                                   jh))
                        return fail(why, "a TIFF JPEG chunk did not decode");
                    if (uint32_t(jw) < vis_w || uint32_t(jh) < vis_h)
                        return fail(why, "a TIFF JPEG chunk is smaller than its place in the image");
                    for (uint32_t y = 0; y < vis_h; ++y)
                        for (uint32_t x = 0; x < vis_w; ++x)
                            for (uint32_t s = 0; s < g.cs; ++s)
                                samples[((y0 + y) * W + x0 + x) * spp + s] =
                                    S(px[(std::size_t(y) * jw + x) * g.cs + s]);
                    continue;
                }

                const uint32_t rows = stored_rows(l, cy);
                const std::size_t expected = std::size_t(g.row_bytes * rows);
                raw.resize(expected);
                bool ok = false;
                switch (l.compression) {
                    case kTiffNone:
                        ok = cnt >= expected;
                        if (ok) std::memcpy(raw.data(), src, expected);
                        break;
                    case kTiffLzw: ok = tiff_lzw(src, cnt, raw.data(), expected); break;
                    case kTiffPackBits: ok = tiff_packbits(src, cnt, raw.data(), expected); break;
                    case kTiffAdobeDeflate:
                    case kTiffDeflate: ok = tiff_inflate(src, cnt, raw.data(), expected); break;
                    default: break;
                }
                if (!ok) return fail(why, "a TIFF chunk is truncated or corrupt");
                if (l.fill_order == 2)
                    for (uint8_t& b : raw) {
                        b = uint8_t((b & 0xF0) >> 4 | (b & 0x0F) << 4);
                        b = uint8_t((b & 0xCC) >> 2 | (b & 0x33) << 2);
                        b = uint8_t((b & 0xAA) >> 1 | (b & 0x55) << 1);
                    }

                const std::size_t n = row.size();
                for (uint32_t y = 0; y < vis_h; ++y) {
                    const uint8_t* rp = raw.data() + std::size_t(g.row_bytes) * y;
                    if (l.bits == 8) {
                        for (std::size_t i = 0; i < n; ++i) row[i] = rp[i];
                    } else if (l.bits == 16) {
                        const bool le = data[0] == 'I';
                        for (std::size_t i = 0; i < n; ++i)
                            row[i] = le ? uint32_t(rp[2 * i] | rp[2 * i + 1] << 8)
                                        : uint32_t(rp[2 * i] << 8 | rp[2 * i + 1]);
                    } else {
                        // 1, 2 or 4 bits: MSB first, each row starting on a byte.
                        std::size_t bit = 0;
                        for (std::size_t i = 0; i < n; ++i, bit += l.bits)
                            row[i] = (rp[bit >> 3] >> (8 - l.bits - (bit & 7))) & mask;
                    }
                    if (l.predictor == 2)  // horizontal differencing, per sample
                        for (std::size_t i = g.cs; i < n; ++i) row[i] = (row[i] + row[i - g.cs]) & mask;
                    for (uint32_t x = 0; x < vis_w; ++x)
                        for (uint32_t s = 0; s < g.cs; ++s)
                            samples[((y0 + y) * W + x0 + x) * spp + (l.planar == 2 ? p : s)] =
                                S(row[std::size_t(x) * g.cs + s]);
                }
            }
        }
    }
    return true;
}

template <class T>
bool decode_as(const uint8_t* data, std::size_t size, const TiffLayout& l, std::vector<T>& rgba, std::string* why) {
    rgba.clear();
    if (l.width == 0 || l.offsets.empty()) return fail(why, "no TIFF layout");
    if (l.bits <= 8) {
        std::vector<uint8_t> samples;
        if (!decode_samples(data, size, l, samples, why)) return false;
        rgba.resize(std::size_t(l.width) * l.height * 4);
        tiff_to_rgba(l, samples.data(), rgba.data());
    } else {
        std::vector<uint16_t> samples;
        if (!decode_samples(data, size, l, samples, why)) return false;
        rgba.resize(std::size_t(l.width) * l.height * 4);
        tiff_to_rgba(l, samples.data(), rgba.data());
    }
    return true;
}

} // namespace

int TiffLayout::channels() const {
    const int base = (photometric <= 1) ? 1 : 3;
    return base + (alpha >= 0 ? 1 : 0);
}

bool is_tiff(const uint8_t* data, std::size_t size) {
    return data && size >= 4 && (std::memcmp(data, "II*\0", 4) == 0 || std::memcmp(data, "MM\0*", 4) == 0);
}

bool tiff_layout(const uint8_t* data, std::size_t size, TiffLayout& out, std::string* why) {
    out = TiffLayout{};
    TiffReader r;
    if (!r.open(data, size)) return fail(why, "not a TIFF (or a BigTIFF, which is not read)");
    std::vector<TiffEntry> ifd;
    if (!r.read_ifd(r.first_ifd(), ifd)) return fail(why, "the TIFF's first directory is out of bounds");
    auto get = [&](uint16_t tag, uint32_t dflt) {
        uint32_t v = dflt;
        if (const TiffEntry* e = tiff_find(ifd, tag)) r.uint_at(*e, 0, v);
        return v;
    };
    TiffLayout& l = out;
    l.width = get(kWidth, 0);
    l.height = get(kHeight, 0);
    if (l.width == 0 || l.height == 0) return fail(why, "a TIFF without dimensions");
    if (l.width > kMaxSide || l.height > kMaxSide || uint64_t(l.width) * l.height > kMaxPixels)
        return fail(why, "TIFF dimensions exceed the decoder's limit");
    l.samples = get(kSamplesPerPixel, 1);
    if (l.samples == 0 || l.samples > 16) return fail(why, "unsupported TIFF samples per pixel");
    l.bits = get(kBitsPerSample, 1);
    if (const TiffEntry* e = tiff_find(ifd, kBitsPerSample))
        for (uint32_t i = 1; i < e->count && i < l.samples; ++i) {
            uint32_t b = 0;
            if (r.uint_at(*e, i, b) && b != l.bits) return fail(why, "TIFF samples of mixed depth are not read");
        }
    if (l.bits != 1 && l.bits != 2 && l.bits != 4 && l.bits != 8 && l.bits != 16)
        return fail(why, "unsupported TIFF bits per sample (1, 2, 4, 8 and 16 are read)");
    if (const TiffEntry* e = tiff_find(ifd, kSampleFormat))
        for (uint32_t i = 0; i < e->count && i < l.samples; ++i) {
            uint32_t f = 1;
            if (r.uint_at(*e, i, f) && f != 1)
                return fail(why, "TIFF signed or floating-point samples are not read");
        }
    l.compression = get(kCompression, 1);
    switch (l.compression) {
        case kTiffNone: case kTiffLzw: case kTiffJpeg: case kTiffAdobeDeflate: case kTiffDeflate:
        case kTiffPackBits: break;
        case kTiffOldJpeg: return fail(why, "old-style TIFF JPEG (compression 6) is not read");
        default: return fail(why, "unsupported TIFF compression");
    }
    l.photometric = get(kPhotometric, l.samples >= 3 ? 2 : 1);
    switch (l.photometric) {
        case 0: case 1: l.color = 1; break;
        case 2: l.color = 3; break;
        case 3: l.color = 1; break;
        case 5:
            if (get(kInkSet, 1) != 1) return fail(why, "a TIFF ink set other than CMYK is not read");
            l.color = 4;
            break;
        case 6:
            if (l.compression != kTiffJpeg) return fail(why, "TIFF YCbCr is read only JPEG-compressed");
            l.color = 3;
            break;
        default: return fail(why, "unsupported TIFF photometric interpretation");
    }
    if (l.samples < l.color) return fail(why, "too few TIFF samples for the photometric interpretation");
    if (l.samples > l.color) {
        const TiffEntry* e = tiff_find(ifd, kExtraSamples);
        uint32_t kind = 2;  // no ExtraSamples tag: the extra sample is read as alpha
        if (e) r.uint_at(*e, 0, kind);
        if (kind == 1 || kind == 2) {
            l.alpha = int(l.color);
            l.premultiplied = kind == 1;
        }
    }
    l.planar = get(kPlanarConfig, 1);
    if (l.planar != 1 && l.planar != 2) return fail(why, "unsupported TIFF planar configuration");
    if (l.samples == 1) l.planar = 1;
    l.predictor = get(kPredictor, 1);
    if (l.compression == kTiffNone || l.compression == kTiffPackBits || l.compression == kTiffJpeg) l.predictor = 1;
    if (l.predictor != 1 && l.predictor != 2) return fail(why, "the TIFF floating-point predictor is not read");
    l.fill_order = get(kFillOrder, 1) == 2 ? 2 : 1;
    if (l.compression == kTiffJpeg) {
        if (l.bits != 8 || l.planar != 1 || l.alpha >= 0 || l.samples != l.color || (l.color != 1 && l.color != 3) ||
            l.photometric == 0)
            return fail(why, "TIFF JPEG is read as 8-bit gray, RGB or YCbCr, interleaved, without alpha");
        if (const TiffEntry* e = tiff_find(ifd, kJpegTables)) {
            l.jpeg_tables = data + e->at;
            l.jpeg_tables_n = e->count;
        }
    }
    if (l.photometric == 3) {
        const TiffEntry* e = tiff_find(ifd, kColorMap);
        const std::size_t n = std::size_t(3) << l.bits;
        if (!e || e->count < n) return fail(why, "a palette TIFF without its colour map");
        l.colormap.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            uint32_t v = 0;
            r.uint_at(*e, uint32_t(i), v);
            l.colormap[i] = uint16_t(v);
        }
    }

    const TiffEntry* offs;
    const TiffEntry* counts;
    if (tiff_find(ifd, kTileOffsets) || tiff_find(ifd, kTileWidth)) {
        l.tiled = true;
        l.chunk_w = get(kTileWidth, 0);
        l.chunk_h = get(kTileLength, 0);
        if (l.chunk_w == 0 || l.chunk_h == 0 || uint64_t(l.chunk_w) * l.chunk_h > kMaxPixels)
            return fail(why, "bad TIFF tile size");
        offs = tiff_find(ifd, kTileOffsets);
        counts = tiff_find(ifd, kTileByteCounts);
    } else {
        l.chunk_w = l.width;
        l.chunk_h = std::min(get(kRowsPerStrip, l.height), l.height);
        if (l.chunk_h == 0) return fail(why, "bad TIFF rows per strip");
        offs = tiff_find(ifd, kStripOffsets);
        counts = tiff_find(ifd, kStripByteCounts);
    }
    const Grid g = grid_of(l);
    const uint64_t chunks = uint64_t(g.across) * g.down * g.planes;
    if (chunks > size) return fail(why, "more TIFF chunks than the file has bytes");
    if (!read_array(r, offs, std::size_t(chunks), l.offsets)) return fail(why, "TIFF chunk offsets are missing");
    uint64_t total = 0;  // decompressed bytes the chunks hold
    for (uint32_t cy = 0; cy < g.down; ++cy) total += g.row_bytes * stored_rows(l, cy) * g.across * g.planes;
    if (!read_array(r, counts, std::size_t(chunks), l.counts)) {
        // An uncompressed file may leave the byte counts out: each chunk is
        // exactly its rows.
        if (l.compression != kTiffNone) return fail(why, "TIFF chunk byte counts are missing");
        l.counts.resize(std::size_t(chunks));
        for (std::size_t i = 0; i < l.counts.size(); ++i)
            l.counts[i] = uint32_t(g.row_bytes * stored_rows(l, uint32_t(i / g.across % g.down)));
    }
    if (l.compression == kTiffNone && total > size) return fail(why, "the TIFF's pixels are truncated");
    if (l.compression != kTiffNone && total > uint64_t(size) * kMaxExpansion)
        return fail(why, "the TIFF claims far more pixels than its data could hold");
    return true;
}

bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& l, std::vector<uint8_t>& rgba,
                 std::string* why) {
    return decode_as(data, size, l, rgba, why);
}

bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& l, std::vector<uint16_t>& rgba,
                 std::string* why) {
    return decode_as(data, size, l, rgba, why);
}

} // namespace broimage::detail
