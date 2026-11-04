// GeoTIFF metadata and native-channel float32 raster decoder (geotiff.h),
// with windowed crop decoding from memory or file streams (classic TIFF and
// BigTIFF).

#include "broimage/geotiff.h"

#include "decode_tiff.h"
#include "tiff_codecs.h"
#include "tiff_ifd.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string_view>

namespace broimage {

namespace {

using detail::TiffEntry;
using detail::TiffReader;
using detail::tiff_find;

constexpr uint64_t kMaxPixels = uint64_t(1) << 28;
constexpr uint64_t kMaxSide = uint64_t(1) << 26;
constexpr uint64_t kMaxExpansion = 4096;

enum Tag : uint16_t {
    kWidth = 256,
    kHeight = 257,
    kBitsPerSample = 258,
    kCompression = 259,
    kFillOrder = 266,
    kStripOffsets = 273,
    kSamplesPerPixel = 277,
    kRowsPerStrip = 278,
    kStripByteCounts = 279,
    kPlanarConfig = 284,
    kPredictor = 317,
    kTileWidth = 322,
    kTileLength = 323,
    kTileOffsets = 324,
    kTileByteCounts = 325,
    kSampleFormat = 339,

    // GeoTIFF & GDAL tags
    kModelPixelScale = 33550,
    kModelTiepoint = 33922,
    kModelTransformation = 34264,
    kGeoKeyDirectory = 34735,
    kGdalNoData = 42113,
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

struct FileGuard {
    std::FILE* f = nullptr;
    ~FileGuard() {
        if (f) std::fclose(f);
    }
};

bool open_tiff_file_stream(const std::string& path, FileGuard& fg, TiffReader& r, std::string* error) {
    const std::filesystem::path fsPath(
        std::u8string_view(reinterpret_cast<const char8_t*>(path.data()), path.size()));
    std::error_code ec;
    if (!std::filesystem::is_regular_file(fsPath, ec)) return fail(error, "can't fopen");
    const uint64_t file_size = std::filesystem::file_size(fsPath, ec);
    if (ec || file_size < 8) return fail(error, "not a TIFF");

#ifdef _MSC_VER
    if (_wfopen_s(&fg.f, fsPath.c_str(), L"rb") != 0) fg.f = nullptr;
#else
    fg.f = std::fopen(path.c_str(), "rb");
#endif
    if (!fg.f) return fail(error, "can't fopen");

    std::FILE* fp = fg.f;
    auto read_fn = [fp](uint64_t offset, void* dst, std::size_t len) -> bool {
        if (len == 0) return true;
#ifdef _MSC_VER
        if (_fseeki64(fp, static_cast<int64_t>(offset), SEEK_SET) != 0) return false;
#else
        if (fseeko(fp, static_cast<off_t>(offset), SEEK_SET) != 0) return false;
#endif
        return std::fread(dst, 1, len, fp) == len;
    };

    if (!r.open_stream(file_size, std::move(read_fn))) return fail(error, "not a TIFF");
    return true;
}

struct RasterLayout {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t samples = 1;
    uint32_t bits = 8;
    uint32_t sample_format = 1;
    uint32_t compression = 1;
    uint32_t predictor = 1;
    uint32_t planar = 1;
    uint32_t fill_order = 1;
    bool tiled = false;
    uint32_t chunk_w = 0;
    uint32_t chunk_h = 0;
    TiffEntry offs_entry{};
    TiffEntry counts_entry{};
    bool has_counts = false;
};

void parse_geotiff_tags(const TiffReader& r, const std::vector<TiffEntry>& ifd, GeoTiffInfo& info) {
    if (const TiffEntry* e = tiff_find(ifd, kModelPixelScale); e && e->count >= 2) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        if (r.double_at(*e, 0, sx) && r.double_at(*e, 1, sy)) {
            if (e->count >= 3) r.double_at(*e, 2, sz);
            info.has_pixel_scale = true;
            info.pixel_scale[0] = sx;
            info.pixel_scale[1] = sy;
            info.pixel_scale[2] = sz;
        }
    }

    if (const TiffEntry* e = tiff_find(ifd, kModelTiepoint); e && e->count >= 6 && e->count <= 65536) {
        info.tiepoints.resize(std::size_t(e->count));
        for (uint64_t i = 0; i < e->count; ++i) {
            if (!r.double_at(*e, i, info.tiepoints[std::size_t(i)])) {
                info.tiepoints.clear();
                break;
            }
        }
    }

    if (const TiffEntry* e = tiff_find(ifd, kModelTransformation); e && e->count >= 16) {
        info.model_transform.resize(16);
        for (uint64_t i = 0; i < 16; ++i) {
            if (!r.double_at(*e, i, info.model_transform[std::size_t(i)])) {
                info.model_transform.clear();
                break;
            }
        }
    }

    if (const TiffEntry* e = tiff_find(ifd, kGeoKeyDirectory); e && e->count >= 4) {
        uint32_t num_keys = 0;
        if (r.uint_at(*e, 3, num_keys)) {
            int projected_epsg = 0;
            int geographic_epsg = 0;
            for (uint32_t k = 0; k < num_keys; ++k) {
                const uint64_t base = 4 + uint64_t(k) * 4;
                if (base + 3 >= e->count) break;
                uint32_t key_id = 0, tag_loc = 0, count = 0, val_off = 0;
                if (!r.uint_at(*e, base, key_id) ||
                    !r.uint_at(*e, base + 1, tag_loc) ||
                    !r.uint_at(*e, base + 2, count) ||
                    !r.uint_at(*e, base + 3, val_off)) {
                    break;
                }
                if (tag_loc == 0 && val_off > 0 && val_off != 32767) {
                    if (key_id == 3072) projected_epsg = int(val_off);
                    else if (key_id == 2048) geographic_epsg = int(val_off);
                }
            }
            if (projected_epsg > 0) info.epsg = projected_epsg;
            else if (geographic_epsg > 0) info.epsg = geographic_epsg;
        }
    }

    if (const TiffEntry* e = tiff_find(ifd, kGdalNoData)) {
        std::string s;
        if (r.ascii(*e, s) && !s.empty()) {
            char* end = nullptr;
            const double v = std::strtod(s.c_str(), &end);
            if (end != s.c_str()) {
                info.has_nodata = true;
                info.nodata = v;
            }
        }
    }
}

bool parse_raster_ifd(const TiffReader& r, GeoTiffInfo& info, RasterLayout& l, std::string* why) {
    info = GeoTiffInfo{};
    l = RasterLayout{};
    std::vector<TiffEntry> ifd;
    if (!r.read_ifd(r.first_ifd(), ifd)) return fail(why, "the TIFF's first directory is out of bounds");

    auto get = [&](uint16_t tag, uint32_t dflt) {
        uint32_t v = dflt;
        if (const TiffEntry* e = tiff_find(ifd, tag)) r.uint_at(*e, 0, v);
        return v;
    };

    l.width = get(kWidth, 0);
    l.height = get(kHeight, 0);
    if (l.width == 0 || l.height == 0) return fail(why, "a TIFF without dimensions");
    if (l.width > kMaxSide || l.height > kMaxSide)
        return fail(why, "TIFF dimensions exceed the decoder's limit");

    l.samples = get(kSamplesPerPixel, 1);
    if (l.samples == 0 || l.samples > 16) return fail(why, "unsupported TIFF samples per pixel");

    l.bits = get(kBitsPerSample, 1);
    if (const TiffEntry* e = tiff_find(ifd, kBitsPerSample)) {
        for (uint32_t i = 1; i < e->count && i < l.samples; ++i) {
            uint32_t b = 0;
            if (r.uint_at(*e, i, b) && b != l.bits)
                return fail(why, "TIFF samples of mixed depth are not read");
        }
    }

    l.sample_format = get(kSampleFormat, 1);
    if (const TiffEntry* e = tiff_find(ifd, kSampleFormat)) {
        for (uint32_t i = 0; i < e->count && i < l.samples; ++i) {
            uint32_t f = 1;
            if (r.uint_at(*e, i, f) && f != l.sample_format)
                return fail(why, "TIFF samples of mixed format are not read");
        }
    }

    if (l.sample_format == 1) {
        if (l.bits != 1 && l.bits != 2 && l.bits != 4 && l.bits != 8 && l.bits != 16 && l.bits != 32)
            return fail(why, "unsupported TIFF unsigned bits per sample");
    } else if (l.sample_format == 2) {
        if (l.bits != 8 && l.bits != 16 && l.bits != 32)
            return fail(why, "unsupported TIFF signed bits per sample");
    } else if (l.sample_format == 3) {
        if (l.bits != 32 && l.bits != 64)
            return fail(why, "unsupported TIFF floating-point bits per sample");
    } else {
        return fail(why, "unsupported TIFF sample format");
    }

    l.compression = get(kCompression, 1);
    switch (l.compression) {
        case detail::kTiffNone:
        case detail::kTiffLzw:
        case detail::kTiffAdobeDeflate:
        case detail::kTiffDeflate:
        case detail::kTiffPackBits:
            break;
        default:
            return fail(why, "unsupported TIFF compression for raster decode");
    }

    l.planar = get(kPlanarConfig, 1);
    if (l.planar != 1 && l.planar != 2) return fail(why, "unsupported TIFF planar configuration");
    if (l.samples == 1) l.planar = 1;

    l.predictor = get(kPredictor, 1);
    if (l.compression == detail::kTiffNone || l.compression == detail::kTiffPackBits) l.predictor = 1;
    if (l.predictor != 1 && l.predictor != 2 && l.predictor != 3)
        return fail(why, "unsupported TIFF predictor");
    if (l.predictor == 3 && l.bits != 16 && l.bits != 32 && l.bits != 64)
        return fail(why, "TIFF floating-point predictor requires 16, 32 or 64-bit samples");

    l.fill_order = get(kFillOrder, 1) == 2 ? 2 : 1;

    const TiffEntry* offs = nullptr;
    const TiffEntry* counts = nullptr;
    if (tiff_find(ifd, kTileOffsets) || tiff_find(ifd, kTileWidth)) {
        l.tiled = true;
        l.chunk_w = get(kTileWidth, 0);
        l.chunk_h = get(kTileLength, 0);
        if (l.chunk_w == 0 || l.chunk_h == 0 || uint64_t(l.chunk_w) * l.chunk_h > kMaxPixels)
            return fail(why, "bad TIFF tile size");
        offs = tiff_find(ifd, kTileOffsets);
        counts = tiff_find(ifd, kTileByteCounts);
    } else {
        l.tiled = false;
        l.chunk_w = l.width;
        l.chunk_h = std::min(get(kRowsPerStrip, l.height), l.height);
        if (l.chunk_h == 0) return fail(why, "bad TIFF rows per strip");
        offs = tiff_find(ifd, kStripOffsets);
        counts = tiff_find(ifd, kStripByteCounts);
    }

    const uint64_t across = ceil_div(l.width, l.chunk_w);
    const uint64_t down = ceil_div(l.height, l.chunk_h);
    const uint64_t planes = l.planar == 2 ? l.samples : 1;
    const uint64_t total_chunks = across * down * planes;
    if (!offs || offs->count < total_chunks) return fail(why, "TIFF chunk offsets are missing");
    l.offs_entry = *offs;
    if (counts && counts->count >= total_chunks) {
        l.counts_entry = *counts;
        l.has_counts = true;
    } else if (l.compression != detail::kTiffNone) {
        return fail(why, "TIFF chunk byte counts are missing");
    }

    info.width = int(l.width);
    info.height = int(l.height);
    info.channels = int(l.samples);
    info.bits_per_sample = int(l.bits);
    info.sample_format = int(l.sample_format);
    info.is_tiled = l.tiled;
    info.tile_width = l.tiled ? int(l.chunk_w) : 0;
    info.tile_height = int(l.chunk_h);
    info.is_bigtiff = r.is_bigtiff();

    parse_geotiff_tags(r, ifd, info);
    return true;
}

bool decode_raster_window(const TiffReader& r, RasterF32& out, const RasterWindow& window, std::string* why) {
    out = RasterF32{};
    RasterLayout l;
    if (!parse_raster_ifd(r, out.info, l, why)) return false;

    if (window.x < 0 || window.y < 0 || window.width < 0 || window.height < 0)
        return fail(why, "invalid raster window");
    const uint32_t wx = uint32_t(window.x);
    const uint32_t wy = uint32_t(window.y);
    if (wx >= l.width || wy >= l.height)
        return fail(why, "raster window lies outside the image");
    const uint32_t ww = window.width == 0 ? (l.width - wx) : uint32_t(window.width);
    const uint32_t wh = window.height == 0 ? (l.height - wy) : uint32_t(window.height);
    if (ww == 0 || wh == 0 || uint64_t(wx) + ww > l.width || uint64_t(wy) + wh > l.height)
        return fail(why, "raster window exceeds image bounds");
    if (uint64_t(ww) * wh > kMaxPixels)
        return fail(why, "raster window dimensions exceed the decoder's limit");

    const uint32_t across = uint32_t(ceil_div(l.width, l.chunk_w));
    const uint32_t down = uint32_t(ceil_div(l.height, l.chunk_h));
    const uint32_t planes = l.planar == 2 ? l.samples : 1;
    const uint32_t cs = l.planar == 2 ? 1 : l.samples;
    const uint64_t row_bytes = ceil_div(uint64_t(l.chunk_w) * cs * l.bits, 8);

    const uint32_t cx0 = wx / l.chunk_w;
    const uint32_t cx1 = (wx + ww - 1) / l.chunk_w;
    const uint32_t cy0 = wy / l.chunk_h;
    const uint32_t cy1 = (wy + wh - 1) / l.chunk_h;

    out.width = int(ww);
    out.height = int(wh);
    out.channels = int(l.samples);
    out.samples.assign(std::size_t(ww) * wh * l.samples, 0.0f);

    std::vector<uint8_t> comp_buf, raw, scratch;
    const std::size_t row_samples = std::size_t(l.chunk_w) * cs;
    std::vector<uint64_t> row64(row_samples);
    const uint64_t mask = l.bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << l.bits) - 1);
    const bool le = r.little_endian();
    const std::size_t spp = l.samples;

    for (uint32_t p = 0; p < planes; ++p) {
        for (uint32_t cy = cy0; cy <= cy1; ++cy) {
            const uint32_t rows = l.tiled ? l.chunk_h : std::min<uint32_t>(l.chunk_h, l.height - cy * l.chunk_h);
            const std::size_t expected = std::size_t(row_bytes * rows);
            for (uint32_t cx = cx0; cx <= cx1; ++cx) {
                const uint64_t idx = (uint64_t(p) * down + cy) * across + cx;
                uint64_t off = 0, cnt = expected;
                if (!r.uint64_at(l.offs_entry, idx, off))
                    return fail(why, "TIFF chunk offset could not be read");
                if (l.has_counts && !r.uint64_at(l.counts_entry, idx, cnt))
                    return fail(why, "TIFF chunk byte count could not be read");
                if (off > r.size() || cnt > r.size() - off)
                    return fail(why, "a TIFF chunk lies outside the file");
                if (l.compression != detail::kTiffNone && expected > cnt * kMaxExpansion)
                    return fail(why, "the TIFF chunk claims far more pixels than its data could hold");

                const uint8_t* src = nullptr;
                if (r.data()) {
                    src = r.data() + std::size_t(off);
                } else {
                    comp_buf.resize(std::size_t(cnt));
                    if (!r.read_bytes(off, comp_buf.data(), std::size_t(cnt)))
                        return fail(why, "failed to read TIFF chunk from file");
                    src = comp_buf.data();
                }

                raw.resize(expected);
                if (!detail::tiff_decompress_chunk(l.compression, l.fill_order, src, std::size_t(cnt),
                                                   raw.data(), expected))
                    return fail(why, "a TIFF chunk is truncated or corrupt");

                const uint32_t x0 = cx * l.chunk_w;
                const uint32_t y0 = cy * l.chunk_h;
                const uint32_t vis_w = std::min(l.chunk_w, l.width - x0);
                const uint32_t vis_h = std::min(l.chunk_h, l.height - y0);

                const uint32_t ix0 = std::max(x0, wx);
                const uint32_t ix1 = std::min(x0 + vis_w, wx + ww);
                const uint32_t iy0 = std::max(y0, wy);
                const uint32_t iy1 = std::min(y0 + vis_h, wy + wh);

                for (uint32_t iy = iy0; iy < iy1; ++iy) {
                    const uint32_t y = iy - y0;
                    uint8_t* rp = raw.data() + std::size_t(row_bytes) * y;
                    if (l.predictor == 3) {
                        detail::tiff_undo_predictor3(rp, l.chunk_w, cs, l.bits / 8, le, scratch);
                    }
                    if (l.bits == 8) {
                        for (std::size_t i = 0; i < row_samples; ++i) row64[i] = rp[i];
                    } else if (l.bits == 16) {
                        for (std::size_t i = 0; i < row_samples; ++i) row64[i] = read_u16(rp + 2 * i, le);
                    } else if (l.bits == 32) {
                        for (std::size_t i = 0; i < row_samples; ++i) row64[i] = read_u32(rp + 4 * i, le);
                    } else if (l.bits == 64) {
                        for (std::size_t i = 0; i < row_samples; ++i) row64[i] = read_u64(rp + 8 * i, le);
                    } else {
                        std::size_t bit = 0;
                        for (std::size_t i = 0; i < row_samples; ++i, bit += l.bits)
                            row64[i] = (rp[bit >> 3] >> (8 - l.bits - (bit & 7))) & mask;
                    }
                    if (l.predictor == 2) {
                        for (std::size_t i = cs; i < row_samples; ++i)
                            row64[i] = (row64[i] + row64[i - cs]) & mask;
                    }

                    const std::size_t dst_row = std::size_t(iy - wy) * ww;
                    for (uint32_t ix = ix0; ix < ix1; ++ix) {
                        const uint32_t x = ix - x0;
                        for (uint32_t s = 0; s < cs; ++s) {
                            const uint64_t u = row64[std::size_t(x) * cs + s];
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
                            const std::size_t ch = l.planar == 2 ? p : s;
                            out.samples[(dst_row + (ix - wx)) * spp + ch] = val;
                        }
                    }
                }
            }
        }
    }
    return true;
}

} // namespace

bool read_geotiff_info_memory(const uint8_t* data, std::size_t size, GeoTiffInfo& out, std::string* error) {
    out = GeoTiffInfo{};
    TiffReader r;
    if (!r.open(data, size)) return fail(error, "not a TIFF");
    RasterLayout l;
    return parse_raster_ifd(r, out, l, error);
}

bool read_geotiff_info_file(const std::string& path, GeoTiffInfo& out, std::string* error) {
    out = GeoTiffInfo{};
    FileGuard fg;
    TiffReader r;
    if (!open_tiff_file_stream(path, fg, r, error)) return false;
    RasterLayout l;
    return parse_raster_ifd(r, out, l, error);
}

bool decode_geotiff_memory_f32(const uint8_t* data, std::size_t size, RasterF32& out,
                               const RasterWindow& window, std::string* error) {
    out = RasterF32{};
    TiffReader r;
    if (!r.open(data, size)) return fail(error, "not a TIFF");
    return decode_raster_window(r, out, window, error);
}

bool decode_geotiff_file_f32(const std::string& path, RasterF32& out,
                             const RasterWindow& window, std::string* error) {
    out = RasterF32{};
    FileGuard fg;
    TiffReader r;
    if (!open_tiff_file_stream(path, fg, r, error)) return false;
    return decode_raster_window(r, out, window, error);
}

} // namespace broimage
