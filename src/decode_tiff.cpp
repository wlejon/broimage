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

uint16_t read_u16(const uint8_t* p, bool le) {
    return le ? uint16_t(p[0] | (uint16_t(p[1]) << 8))
              : uint16_t((uint16_t(p[0]) << 8) | p[1]);
}

uint32_t read_u32(const uint8_t* p, bool le) {
    return le ? (uint32_t(read_u16(p, true)) | (uint32_t(read_u16(p + 2, true)) << 16))
              : ((uint32_t(read_u16(p, false)) << 16) | uint32_t(read_u16(p + 2, false)));
}

uint64_t read_u64(const uint8_t* p, bool le) {
    return le ? (uint64_t(read_u32(p, true)) | (uint64_t(read_u32(p + 4, true)) << 32))
              : ((uint64_t(read_u32(p, false)) << 32) | uint64_t(read_u32(p + 4, false)));
}

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

bool read_array64(const TiffReader& r, const TiffEntry* e, std::size_t n, std::vector<uint64_t>& out) {
    if (!e || e->count < n) return false;
    out.resize(n);
    for (std::size_t i = 0; i < n; ++i)
        if (!r.uint64_at(*e, uint64_t(i), out[i])) return false;
    return true;
}

template <class S>
bool decode_samples(const uint8_t* data, std::size_t size, const TiffLayout& l, std::vector<S>& samples,
                    std::string* why) {
    const Grid g = grid_of(l);
    const std::size_t W = l.width, spp = l.samples;
    samples.assign(W * l.height * spp, S(0));
    std::vector<uint8_t> raw, scratch;
    std::vector<uint32_t> row(std::size_t(l.chunk_w) * g.cs);
    const uint32_t mask = l.bits >= 32 ? 0xFFFFFFFFu : (1u << l.bits) - 1;
    const bool le = data[0] == 'I';
    for (uint32_t p = 0; p < g.planes; ++p) {
        for (uint32_t cy = 0; cy < g.down; ++cy) {
            for (uint32_t cx = 0; cx < g.across; ++cx) {
                const std::size_t idx = (std::size_t(p) * g.down + cy) * g.across + cx;
                const uint32_t x0 = cx * l.chunk_w, y0 = cy * l.chunk_h;
                const uint32_t vis_w = std::min(l.chunk_w, l.width - x0);
                const uint32_t vis_h = std::min(l.chunk_h, l.height - y0);
                const uint64_t off = l.offsets[idx], cnt = l.counts[idx];
                if (off > size || cnt > size - off) return fail(why, "a TIFF chunk lies outside the file");
                const uint8_t* src = data + std::size_t(off);

                if (l.compression == kTiffJpeg) {
                    std::vector<uint8_t> px;
                    int jw = 0, jh = 0;
                    if (!tiff_jpeg(l.jpeg_tables, l.jpeg_tables_n, src, std::size_t(cnt), l.photometric == 2,
                                   int(g.cs), px, jw, jh))
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
                if (!tiff_decompress_chunk(l.compression, l.fill_order, src, std::size_t(cnt), raw.data(), expected))
                    return fail(why, "a TIFF chunk is truncated or corrupt");

                const std::size_t n = row.size();
                for (uint32_t y = 0; y < vis_h; ++y) {
                    uint8_t* rp = raw.data() + std::size_t(g.row_bytes) * y;
                    if (l.predictor == 3) {
                        tiff_undo_predictor3(rp, l.chunk_w, g.cs, l.bits / 8, le, scratch);
                    }
                    if (l.bits == 8) {
                        for (std::size_t i = 0; i < n; ++i) row[i] = rp[i];
                    } else if (l.bits == 16) {
                        for (std::size_t i = 0; i < n; ++i) row[i] = read_u16(rp + 2 * i, le);
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

bool decode_samples_f32(const uint8_t* data, std::size_t size, const TiffLayout& l, std::vector<float>& samples,
                        std::string* why) {
    const Grid g = grid_of(l);
    const std::size_t W = l.width, spp = l.samples;
    samples.assign(W * l.height * spp, 0.0f);
    std::vector<uint8_t> raw, scratch;
    const std::size_t n = std::size_t(l.chunk_w) * g.cs;
    std::vector<uint64_t> row64(n);
    const uint64_t mask = l.bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << l.bits) - 1);
    const bool le = data[0] == 'I';

    for (uint32_t p = 0; p < g.planes; ++p) {
        for (uint32_t cy = 0; cy < g.down; ++cy) {
            for (uint32_t cx = 0; cx < g.across; ++cx) {
                const std::size_t idx = (std::size_t(p) * g.down + cy) * g.across + cx;
                const uint32_t x0 = cx * l.chunk_w, y0 = cy * l.chunk_h;
                const uint32_t vis_w = std::min(l.chunk_w, l.width - x0);
                const uint32_t vis_h = std::min(l.chunk_h, l.height - y0);
                const uint64_t off = l.offsets[idx], cnt = l.counts[idx];
                if (off > size || cnt > size - off) return fail(why, "a TIFF chunk lies outside the file");
                const uint8_t* src = data + std::size_t(off);

                const uint32_t rows = stored_rows(l, cy);
                const std::size_t expected = std::size_t(g.row_bytes * rows);
                raw.resize(expected);
                if (!tiff_decompress_chunk(l.compression, l.fill_order, src, std::size_t(cnt), raw.data(), expected))
                    return fail(why, "a TIFF chunk is truncated or corrupt");

                for (uint32_t y = 0; y < vis_h; ++y) {
                    uint8_t* rp = raw.data() + std::size_t(g.row_bytes) * y;
                    if (l.predictor == 3) {
                        tiff_undo_predictor3(rp, l.chunk_w, g.cs, l.bits / 8, le, scratch);
                    }
                    if (l.bits == 8) {
                        for (std::size_t i = 0; i < n; ++i) row64[i] = rp[i];
                    } else if (l.bits == 16) {
                        for (std::size_t i = 0; i < n; ++i) row64[i] = read_u16(rp + 2 * i, le);
                    } else if (l.bits == 32) {
                        for (std::size_t i = 0; i < n; ++i) row64[i] = read_u32(rp + 4 * i, le);
                    } else if (l.bits == 64) {
                        for (std::size_t i = 0; i < n; ++i) row64[i] = read_u64(rp + 8 * i, le);
                    }
                    if (l.predictor == 2) {
                        for (std::size_t i = g.cs; i < n; ++i) row64[i] = (row64[i] + row64[i - g.cs]) & mask;
                    }
                    for (uint32_t x = 0; x < vis_w; ++x) {
                        for (uint32_t s = 0; s < g.cs; ++s) {
                            const uint64_t u = row64[std::size_t(x) * g.cs + s];
                            float val = 0.0f;
                            if (l.sample_format == 3) {
                                if (l.bits == 32) {
                                    const uint32_t u32 = uint32_t(u);
                                    std::memcpy(&val, &u32, 4);
                                } else {
                                    double d = 0.0;
                                    std::memcpy(&d, &u, 8);
                                    val = float(d);
                                }
                            } else if (l.sample_format == 2) {
                                if (l.bits == 8) val = float(int8_t(uint8_t(u)));
                                else if (l.bits == 16) val = float(int16_t(uint16_t(u)));
                                else val = float(int32_t(uint32_t(u)));
                            } else {
                                val = float(u);
                            }
                            samples[((y0 + y) * W + x0 + x) * spp + (l.planar == 2 ? p : s)] = val;
                        }
                    }
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
    if (l.sample_format == 1 && l.bits <= 8) {
        std::vector<uint8_t> samples;
        if (!decode_samples(data, size, l, samples, why)) return false;
        rgba.resize(std::size_t(l.width) * l.height * 4);
        tiff_to_rgba(l, samples.data(), rgba.data());
    } else if (l.sample_format == 1 && l.bits == 16) {
        std::vector<uint16_t> samples;
        if (!decode_samples(data, size, l, samples, why)) return false;
        rgba.resize(std::size_t(l.width) * l.height * 4);
        tiff_to_rgba(l, samples.data(), rgba.data());
    } else {
        std::vector<float> samples;
        if (!decode_samples_f32(data, size, l, samples, why)) return false;
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
    if (!data || size < 4) return false;
    if (std::memcmp(data, "II*\0", 4) == 0 || std::memcmp(data, "MM\0*", 4) == 0) return true;
    if (size >= 8 && (std::memcmp(data, "II+\0\x08\0\0\0", 8) == 0 ||
                      std::memcmp(data, "MM\0+\0\x08\0\0", 8) == 0))
        return true;
    return false;
}

bool tiff_decompress_chunk(uint32_t compression, uint32_t fill_order,
                           const uint8_t* src, std::size_t cnt,
                           uint8_t* dst, std::size_t expected) {
    bool ok = false;
    switch (compression) {
        case kTiffNone:
            ok = cnt >= expected;
            if (ok) std::memcpy(dst, src, expected);
            break;
        case kTiffLzw: ok = tiff_lzw(src, cnt, dst, expected); break;
        case kTiffPackBits: ok = tiff_packbits(src, cnt, dst, expected); break;
        case kTiffAdobeDeflate:
        case kTiffDeflate: ok = tiff_inflate(src, cnt, dst, expected); break;
        default: break;
    }
    if (!ok) return false;
    if (fill_order == 2) {
        for (std::size_t i = 0; i < expected; ++i) {
            uint8_t b = dst[i];
            b = uint8_t((b & 0xF0) >> 4 | (b & 0x0F) << 4);
            b = uint8_t((b & 0xCC) >> 2 | (b & 0x33) << 2);
            b = uint8_t((b & 0xAA) >> 1 | (b & 0x55) << 1);
            dst[i] = b;
        }
    }
    return true;
}

void tiff_undo_predictor3(uint8_t* rp, std::size_t row_pixels, uint32_t cs,
                          uint32_t bps, bool out_le, std::vector<uint8_t>& scratch) {
    const std::size_t stride = row_pixels * cs;
    const std::size_t row_len = stride * bps;
    for (std::size_t col = cs; col < row_len; ++col) {
        rp[col] = uint8_t(rp[col] + rp[col - cs]);
    }
    scratch.assign(rp, rp + row_len);
    for (std::size_t s = 0; s < stride; ++s) {
        for (uint32_t b = 0; b < bps; ++b) {
            const uint32_t dst_b = out_le ? (bps - 1 - b) : b;
            rp[s * bps + dst_b] = scratch[std::size_t(b) * stride + s];
        }
    }
}

bool tiff_layout(const uint8_t* data, std::size_t size, TiffLayout& out, std::string* why) {
    out = TiffLayout{};
    TiffReader r;
    if (!r.open(data, size)) return fail(why, "not a TIFF");
    std::vector<TiffEntry> ifd;
    if (!r.read_ifd(r.first_ifd(), ifd)) return fail(why, "the TIFF's first directory is out of bounds");
    auto get = [&](uint16_t tag, uint32_t dflt) {
        uint32_t v = dflt;
        if (const TiffEntry* e = tiff_find(ifd, tag)) r.uint_at(*e, 0, v);
        return v;
    };
    TiffLayout& l = out;
    l.bigtiff = r.is_bigtiff();
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
    l.sample_format = get(kSampleFormat, 1);
    if (const TiffEntry* e = tiff_find(ifd, kSampleFormat))
        for (uint32_t i = 0; i < e->count && i < l.samples; ++i) {
            uint32_t f = 1;
            if (r.uint_at(*e, i, f) && f != l.sample_format)
                return fail(why, "TIFF samples of mixed format are not read");
        }
    if (l.sample_format == 1) {
        if (l.bits != 1 && l.bits != 2 && l.bits != 4 && l.bits != 8 && l.bits != 16 && l.bits != 32)
            return fail(why, "unsupported TIFF unsigned bits per sample");
    } else if (l.sample_format == 2) {
        if (l.bits != 8 && l.bits != 16 && l.bits != 32)
            return fail(why, "unsupported TIFF signed bits per sample (8, 16 and 32 are read)");
    } else if (l.sample_format == 3) {
        if (l.bits != 32 && l.bits != 64)
            return fail(why, "unsupported TIFF floating-point bits per sample (32 and 64 are read)");
    } else {
        return fail(why, "unsupported TIFF sample format");
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
        case 3:
            if (l.sample_format != 1 || l.bits > 16)
                return fail(why, "palette TIFF requires 1..16-bit unsigned samples");
            l.color = 1;
            break;
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
    if (l.predictor != 1 && l.predictor != 2 && l.predictor != 3)
        return fail(why, "unsupported TIFF predictor");
    if (l.predictor == 3 && l.bits != 16 && l.bits != 32 && l.bits != 64)
        return fail(why, "TIFF floating-point predictor requires 16, 32 or 64-bit samples");
    l.fill_order = get(kFillOrder, 1) == 2 ? 2 : 1;
    if (l.compression == kTiffJpeg) {
        if (l.bits != 8 || l.sample_format != 1 || l.planar != 1 || l.alpha >= 0 || l.samples != l.color ||
            (l.color != 1 && l.color != 3) || l.photometric == 0)
            return fail(why, "TIFF JPEG is read as 8-bit gray, RGB or YCbCr, interleaved, without alpha");
        if (const TiffEntry* e = tiff_find(ifd, kJpegTables)) {
            l.jpeg_tables = data + std::size_t(e->at);
            l.jpeg_tables_n = std::size_t(e->count);
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
    if (!read_array64(r, offs, std::size_t(chunks), l.offsets)) return fail(why, "TIFF chunk offsets are missing");
    uint64_t total = 0;  // decompressed bytes the chunks hold
    for (uint32_t cy = 0; cy < g.down; ++cy) total += g.row_bytes * stored_rows(l, cy) * g.across * g.planes;
    if (!read_array64(r, counts, std::size_t(chunks), l.counts)) {
        // An uncompressed file may leave the byte counts out: each chunk is
        // exactly its rows.
        if (l.compression != kTiffNone) return fail(why, "TIFF chunk byte counts are missing");
        l.counts.resize(std::size_t(chunks));
        for (std::size_t i = 0; i < l.counts.size(); ++i)
            l.counts[i] = g.row_bytes * stored_rows(l, uint32_t(i / g.across % g.down));
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

bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& l, std::vector<float>& rgba,
                 std::string* why) {
    return decode_as(data, size, l, rgba, why);
}

} // namespace broimage::detail
