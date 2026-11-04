// GeoTIFF, Float32/Float64, Signed Int8/Int16/Int32, Predictor 2 (32-bit),
// Predictor 3 (floating-point byte-lane horizontal differencing), BigTIFF
// (magic 43), and windowed file decode tests.

#include "broimage/decode.h"
#include "broimage/geotiff.h"

#include "tiff_writer.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

namespace {

// Helper to build a DOUBLE (type 12) IFD entry for tiffw.
tiffw::Entry doubles_entry(uint16_t tag, const std::vector<double>& vals, bool le = true) {
    tiffw::Entry e{tag, 12, {}, {}};
    // We store raw bytes so write_ifd writes them directly, and adjust count via type 12 in custom writer
    // Wait: tiffw::write_ifd uses type_size(e.type). Let's check tiffw::type_size!
    // In tiff_writer.h, type_size only knows 3, 4, 5, 10! So let's write a dedicated GeoTIFF/BigTIFF builder here.
    (void)vals;
    (void)le;
    return e;
}

struct GeoBuf {
    std::vector<uint8_t> b;
    bool le = true;

    void u8(uint8_t v) { b.push_back(v); }
    void u16(uint16_t v) {
        if (le) { u8(uint8_t(v)); u8(uint8_t(v >> 8)); }
        else { u8(uint8_t(v >> 8)); u8(uint8_t(v)); }
    }
    void u32(uint32_t v) {
        if (le) { u16(uint16_t(v)); u16(uint16_t(v >> 16)); }
        else { u16(uint16_t(v >> 16)); u16(uint16_t(v)); }
    }
    void u64(uint64_t v) {
        if (le) { u32(uint32_t(v)); u32(uint32_t(v >> 32)); }
        else { u32(uint32_t(v >> 32)); u32(uint32_t(v)); }
    }
    void f32(float v) {
        uint32_t u = 0;
        std::memcpy(&u, &v, 4);
        u32(u);
    }
    void f64(double v) {
        uint64_t u = 0;
        std::memcpy(&u, &v, 8);
        u64(u);
    }
    void put32_at(std::size_t at, uint32_t v) {
        GeoBuf t; t.le = le; t.u32(v);
        std::memcpy(b.data() + at, t.b.data(), 4);
    }
    void put64_at(std::size_t at, uint64_t v) {
        GeoBuf t; t.le = le; t.u64(v);
        std::memcpy(b.data() + at, t.b.data(), 8);
    }
    void bytes(const void* p, std::size_t n) {
        const auto* s = static_cast<const uint8_t*>(p);
        b.insert(b.end(), s, s + n);
    }
    void align() { if (b.size() & 1) u8(0); }
};

struct RawTag {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint64_t count = 0;
    std::vector<uint8_t> payload;
};

RawTag tag_u16(uint16_t tag, const std::vector<uint16_t>& vals, bool le) {
    GeoBuf b; b.le = le;
    for (uint16_t v : vals) b.u16(v);
    return {tag, 3, vals.size(), std::move(b.b)};
}

RawTag tag_u32(uint16_t tag, const std::vector<uint32_t>& vals, bool le) {
    GeoBuf b; b.le = le;
    for (uint32_t v : vals) b.u32(v);
    return {tag, 4, vals.size(), std::move(b.b)};
}

RawTag tag_u64(uint16_t tag, const std::vector<uint64_t>& vals, bool le) {
    GeoBuf b; b.le = le;
    for (uint64_t v : vals) b.u64(v);
    return {tag, 16, vals.size(), std::move(b.b)};
}

RawTag tag_f64(uint16_t tag, const std::vector<double>& vals, bool le) {
    GeoBuf b; b.le = le;
    for (double v : vals) b.f64(v);
    return {tag, 12, vals.size(), std::move(b.b)};
}

RawTag tag_ascii(uint16_t tag, const std::string& s) {
    std::vector<uint8_t> p(s.begin(), s.end());
    p.push_back(0);
    return {tag, 2, p.size(), std::move(p)};
}

uint64_t write_geo_ifd(GeoBuf& buf, std::vector<RawTag> tags, bool bigtiff) {
    std::sort(tags.begin(), tags.end(), [](const RawTag& a, const RawTag& b) { return a.tag < b.tag; });
    const std::size_t inline_max = bigtiff ? 8 : 4;
    std::vector<uint64_t> offsets(tags.size(), 0);
    for (std::size_t i = 0; i < tags.size(); ++i) {
        if (tags[i].payload.size() > inline_max) {
            buf.align();
            offsets[i] = buf.b.size();
            buf.bytes(tags[i].payload.data(), tags[i].payload.size());
        }
    }
    buf.align();
    const uint64_t ifd_off = buf.b.size();
    if (bigtiff) buf.u64(tags.size());
    else buf.u16(uint16_t(tags.size()));

    for (std::size_t i = 0; i < tags.size(); ++i) {
        const RawTag& t = tags[i];
        buf.u16(t.tag);
        buf.u16(t.type);
        if (bigtiff) buf.u64(t.count);
        else buf.u32(uint32_t(t.count));
        if (t.payload.size() <= inline_max) {
            std::vector<uint8_t> pad = t.payload;
            pad.resize(inline_max, 0);
            buf.bytes(pad.data(), inline_max);
        } else {
            if (bigtiff) buf.u64(offsets[i]);
            else buf.u32(uint32_t(offsets[i]));
        }
    }
    if (bigtiff) buf.u64(0);
    else buf.u32(0);
    return ifd_off;
}

// Apply forward Predictor = 3 (Adobe Tech Note 3) to a row of big-endian MSB-first byte lanes!
void encode_predictor3_row(uint8_t* rp, std::size_t row_pixels, uint32_t cs, uint32_t bps, bool file_le) {
    const std::size_t stride = row_pixels * cs;
    const std::size_t row_len = stride * bps;
    std::vector<uint8_t> shuffled(row_len);
    for (std::size_t s = 0; s < stride; ++s) {
        for (uint32_t b = 0; b < bps; ++b) {
            // Lane 0 is always MSB, Lane bps-1 is always LSB.
            const uint32_t src_b = file_le ? (bps - 1 - b) : b;
            shuffled[std::size_t(b) * stride + s] = rp[s * bps + src_b];
        }
    }
    for (std::size_t col = row_len; col-- > cs;) {
        shuffled[col] = uint8_t(shuffled[col] - shuffled[col - cs]);
    }
    std::memcpy(rp, shuffled.data(), row_len);
}

struct FloatGeoSpec {
    bool le = true;
    bool bigtiff = false;
    uint32_t w = 16;
    uint32_t h = 16;
    uint32_t spp = 1;
    uint32_t bits = 32;          // 32 or 64 for float
    uint32_t sample_format = 3;  // 3 = float, 2 = sint, 1 = uint
    uint32_t compression = 8;    // 1 = none, 5 = LZW, 8 = AdobeDeflate, 32946 = Deflate
    uint32_t predictor = 3;      // 1, 2, or 3
    uint32_t rows_per_strip = 0;
    uint32_t tile_w = 0;
    uint32_t tile_h = 0;
    bool with_geotiff_tags = false;
    bool with_model_transform = false;
    int epsg = 2056;
    std::string nodata_str = "-9999";
};

std::vector<uint8_t> make_geotiff(const FloatGeoSpec& s, const std::vector<double>& samples) {
    GeoBuf buf;
    buf.le = s.le;
    buf.u8(s.le ? 'I' : 'M');
    buf.u8(s.le ? 'I' : 'M');
    if (s.bigtiff) {
        buf.u16(43);
        buf.u16(8);
        buf.u16(0);
        buf.u64(0);
    } else {
        buf.u16(42);
        buf.u32(0);
    }

    const bool tiled = s.tile_w > 0;
    const uint32_t cw = tiled ? s.tile_w : s.w;
    const uint32_t ch = tiled ? s.tile_h : (s.rows_per_strip ? std::min(s.rows_per_strip, s.h) : s.h);
    const uint32_t across = (s.w + cw - 1) / cw;
    const uint32_t down = (s.h + ch - 1) / ch;
    const uint32_t cs = s.spp;
    const uint32_t bps = s.bits / 8;

    std::vector<uint64_t> offs64, cnts64;
    std::vector<uint32_t> offs32, cnts32;

    for (uint32_t cy = 0; cy < down; ++cy) {
        for (uint32_t cx = 0; cx < across; ++cx) {
            const uint32_t rows = tiled ? ch : std::min(ch, s.h - cy * ch);
            GeoBuf raw;
            raw.le = s.le;
            for (uint32_t y = 0; y < rows; ++y) {
                std::vector<double> row_vals(std::size_t(cw) * cs, 0.0);
                for (uint32_t x = 0; x < cw; ++x) {
                    for (uint32_t c = 0; c < cs; ++c) {
                        const uint32_t ix = cx * cw + x;
                        const uint32_t iy = cy * ch + y;
                        if (ix < s.w && iy < s.h) {
                            row_vals[std::size_t(x) * cs + c] =
                                samples[(std::size_t(iy) * s.w + ix) * s.spp + c];
                        }
                    }
                }
                const std::size_t row_start = raw.b.size();
                if (s.sample_format == 3) {
                    for (double v : row_vals) {
                        if (s.bits == 32) raw.f32(float(v));
                        else raw.f64(v);
                    }
                } else {
                    std::vector<uint64_t> int_row(row_vals.size());
                    const uint64_t mask = s.bits >= 64 ? ~uint64_t(0) : ((uint64_t(1) << s.bits) - 1);
                    for (std::size_t i = 0; i < row_vals.size(); ++i) {
                        const int64_t iv = static_cast<int64_t>(std::llround(row_vals[i]));
                        int_row[i] = uint64_t(iv) & mask;
                    }
                    if (s.predictor == 2) {
                        for (std::size_t i = int_row.size(); i-- > cs;) {
                            int_row[i] = (int_row[i] - int_row[i - cs]) & mask;
                        }
                    }
                    for (uint64_t u : int_row) {
                        if (s.bits == 8) raw.u8(uint8_t(u));
                        else if (s.bits == 16) raw.u16(uint16_t(u));
                        else if (s.bits == 32) raw.u32(uint32_t(u));
                    }
                }
                if (s.predictor == 3) {
                    encode_predictor3_row(raw.b.data() + row_start, cw, cs, bps, s.le);
                }
            }

            tiffw::Spec ts;
            ts.compression = s.compression;
            const std::vector<uint8_t> z = tiffw::compress(ts, raw.b);
            buf.align();
            offs64.push_back(buf.b.size());
            cnts64.push_back(z.size());
            offs32.push_back(uint32_t(buf.b.size()));
            cnts32.push_back(uint32_t(z.size()));
            buf.bytes(z.data(), z.size());
        }
    }

    std::vector<RawTag> tags = {
        tag_u32(256, {s.w}, s.le),
        tag_u32(257, {s.h}, s.le),
        tag_u16(258, std::vector<uint16_t>(s.spp, uint16_t(s.bits)), s.le),
        tag_u16(259, {uint16_t(s.compression)}, s.le),
        tag_u16(262, {uint16_t(s.spp >= 3 ? 2 : 1)}, s.le),
        tag_u16(277, {uint16_t(s.spp)}, s.le),
        tag_u16(284, {1}, s.le),
        tag_u16(339, std::vector<uint16_t>(s.spp, uint16_t(s.sample_format)), s.le),
    };
    if (s.predictor != 1) {
        tags.push_back(tag_u16(317, {uint16_t(s.predictor)}, s.le));
    }
    if (tiled) {
        tags.push_back(tag_u32(322, {s.tile_w}, s.le));
        tags.push_back(tag_u32(323, {s.tile_h}, s.le));
        tags.push_back(s.bigtiff ? tag_u64(324, offs64, s.le) : tag_u32(324, offs32, s.le));
        tags.push_back(s.bigtiff ? tag_u64(325, cnts64, s.le) : tag_u32(325, cnts32, s.le));
    } else {
        tags.push_back(s.bigtiff ? tag_u64(273, offs64, s.le) : tag_u32(273, offs32, s.le));
        tags.push_back(tag_u32(278, {ch}, s.le));
        tags.push_back(s.bigtiff ? tag_u64(279, cnts64, s.le) : tag_u32(279, cnts32, s.le));
    }

    if (s.with_geotiff_tags) {
        tags.push_back(tag_f64(33550, {0.5, 0.5, 0.0}, s.le));
        tags.push_back(tag_f64(33922, {0.0, 0.0, 0.0, 2600000.0, 1200000.0, 0.0}, s.le));
        if (s.with_model_transform) {
            tags.push_back(tag_f64(34264, {
                0.5, 0.0, 0.0, 2600000.0,
                0.0, -0.5, 0.0, 1200000.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            }, s.le));
        }
        const uint16_t gk_key = (s.epsg == 4326) ? 2048 : 3072;
        tags.push_back(tag_u16(34735, {
            1, 1, 0, 2,
            1024, 0, 1, 1,
            gk_key, 0, 1, uint16_t(s.epsg)
        }, s.le));
        if (!s.nodata_str.empty()) {
            tags.push_back(tag_ascii(42113, s.nodata_str));
        }
    }

    const uint64_t ifd_off = write_geo_ifd(buf, std::move(tags), s.bigtiff);
    if (s.bigtiff) buf.put64_at(8, ifd_off);
    else buf.put32_at(4, uint32_t(ifd_off));
    return buf.b;
}

void test_float32_predictor3_stripped_and_tiled() {
    const uint32_t W = 29, H = 21;
    std::vector<double> dem(W * H);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            dem[y * W + x] = 1450.25 + double(x) * 3.125 - double(y) * 1.75 + ((x == 3 && y == 4) ? -9999.0 : 0.0);
        }
    }

    for (bool le : {true, false}) {
        for (uint32_t comp : {5u, 8u, 32946u}) {
            for (bool tiled : {false, true}) {
                FloatGeoSpec s;
                s.le = le;
                s.w = W;
                s.h = H;
                s.bits = 32;
                s.sample_format = 3;
                s.compression = comp;
                s.predictor = 3;
                if (tiled) {
                    s.tile_w = 16;
                    s.tile_h = 16;
                } else {
                    s.rows_per_strip = 8;
                }
                const auto tiff = make_geotiff(s, dem);

                broimage::RasterF32 raster;
                std::string err;
                CHECK(broimage::decode_geotiff_memory_f32(tiff.data(), tiff.size(), raster, {}, &err));
                CHECK(raster.width == int(W) && raster.height == int(H) && raster.channels == 1);
                CHECK(raster.samples.size() == dem.size());
                for (std::size_t i = 0; i < dem.size(); ++i) {
                    CHECK(raster.samples[i] == float(dem[i]));
                }

                // Also verify decode_memory_f32 (RGBA float32)
                broimage::ImageF32 imgf;
                CHECK(broimage::decode_memory_f32(tiff.data(), tiff.size(), imgf, &err));
                CHECK(imgf.width == int(W) && imgf.height == int(H) && imgf.channels == 4);
                for (std::size_t i = 0; i < dem.size(); ++i) {
                    CHECK(imgf.pixels[i * 4 + 0] == float(dem[i]));
                    CHECK(imgf.pixels[i * 4 + 1] == float(dem[i]));
                    CHECK(imgf.pixels[i * 4 + 2] == float(dem[i]));
                    CHECK(imgf.pixels[i * 4 + 3] == 1.0f);
                }
            }
        }
    }
}

void test_float64_and_signed_ints() {
    const uint32_t W = 18, H = 14;
    std::vector<double> vals(W * H);
    for (std::size_t i = 0; i < vals.size(); ++i) {
        vals[i] = double(int(i) * 17 - 1500);
    }

    // Float64 with Predictor 3 + Deflate
    for (bool le : {true, false}) {
        FloatGeoSpec s;
        s.le = le;
        s.w = W; s.h = H; s.bits = 64; s.sample_format = 3; s.compression = 8; s.predictor = 3;
        s.rows_per_strip = 6;
        const auto tiff = make_geotiff(s, vals);
        broimage::RasterF32 r;
        CHECK(broimage::decode_geotiff_memory_f32(tiff.data(), tiff.size(), r));
        CHECK(r.info.bits_per_sample == 64 && r.info.sample_format == 3);
        for (std::size_t i = 0; i < vals.size(); ++i) {
            CHECK(r.samples[i] == float(vals[i]));
        }
    }

    // Signed int16 and int32 with Predictor 2 + LZW / Deflate
    for (uint32_t bits : {16u, 32u}) {
        for (bool le : {true, false}) {
            FloatGeoSpec s;
            s.le = le;
            s.w = W; s.h = H; s.bits = bits; s.sample_format = 2; s.compression = 5; s.predictor = 2;
            s.tile_w = 8; s.tile_h = 8;
            const auto tiff = make_geotiff(s, vals);
            broimage::RasterF32 r;
            CHECK(broimage::decode_geotiff_memory_f32(tiff.data(), tiff.size(), r));
            CHECK(r.info.bits_per_sample == int(bits) && r.info.sample_format == 2);
            for (std::size_t i = 0; i < vals.size(); ++i) {
                CHECK(r.samples[i] == float(vals[i]));
            }

            // Also test via decode_memory_f32
            broimage::ImageF32 imgf;
            CHECK(broimage::decode_memory_f32(tiff.data(), tiff.size(), imgf));
            for (std::size_t i = 0; i < vals.size(); ++i) {
                CHECK(imgf.pixels[i * 4] == float(vals[i]));
            }
        }
    }
}

void test_bigtiff_and_geotiff_metadata_and_window() {
    const uint32_t W = 48, H = 36;
    std::vector<double> dem(W * H);
    for (uint32_t y = 0; y < H; ++y) {
        for (uint32_t x = 0; x < W; ++x) {
            dem[y * W + x] = 400.0 + double(y * 100 + x) * 0.25;
        }
    }

    for (bool bigtiff : {false, true}) {
        for (bool le : {true, false}) {
            FloatGeoSpec s;
            s.le = le;
            s.bigtiff = bigtiff;
            s.w = W;
            s.h = H;
            s.bits = 32;
            s.sample_format = 3;
            s.compression = 8;
            s.predictor = 3;
            s.tile_w = 16;
            s.tile_h = 16;
            s.with_geotiff_tags = true;
            s.with_model_transform = true;
            s.epsg = bigtiff ? 4326 : 2056;
            s.nodata_str = "-9999.5";

            const auto tiff = make_geotiff(s, dem);
            const std::string path = "broimage_test_geotiff_tmp.tif";
            if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(tiff.data(), 1, tiff.size(), f);
                std::fclose(f);
            }

            broimage::GeoTiffInfo info_mem, info_file;
            CHECK(broimage::read_geotiff_info_memory(tiff.data(), tiff.size(), info_mem));
            CHECK(broimage::read_geotiff_info_file(path, info_file));
            for (const auto* info : {&info_mem, &info_file}) {
                CHECK(info->width == int(W) && info->height == int(H));
                CHECK(info->channels == 1);
                CHECK(info->bits_per_sample == 32);
                CHECK(info->sample_format == 3);
                CHECK(info->is_tiled && info->tile_width == 16 && info->tile_height == 16);
                CHECK(info->is_bigtiff == bigtiff);
                CHECK(info->has_pixel_scale);
                CHECK(info->pixel_scale[0] == 0.5 && info->pixel_scale[1] == 0.5 && info->pixel_scale[2] == 0.0);
                CHECK(info->tiepoints.size() == 6);
                CHECK(info->tiepoints[3] == 2600000.0 && info->tiepoints[4] == 1200000.0);
                CHECK(info->model_transform.size() == 16);
                CHECK(info->model_transform[0] == 0.5 && info->model_transform[5] == -0.5);
                CHECK(info->has_nodata && info->nodata == -9999.5);
                CHECK(info->epsg == s.epsg);
            }

            // Windowed crop across multiple tiles: x=10..35 (width=25), y=7..29 (height=22)
            broimage::RasterWindow win{10, 7, 25, 22};
            broimage::RasterF32 crop_mem, crop_file;
            CHECK(broimage::decode_geotiff_memory_f32(tiff.data(), tiff.size(), crop_mem, win));
            CHECK(broimage::decode_geotiff_file_f32(path, crop_file, win));
            CHECK(crop_file.width == 25 && crop_file.height == 22 && crop_file.channels == 1);
            CHECK(crop_file.info.width == int(W) && crop_file.info.height == int(H));
            CHECK(crop_mem.samples == crop_file.samples);
            for (int y = 0; y < win.height; ++y) {
                for (int x = 0; x < win.width; ++x) {
                    const float got = crop_file.samples[std::size_t(y) * win.width + x];
                    const float want = float(dem[std::size_t(win.y + y) * W + (win.x + x)]);
                    CHECK(got == want);
                }
            }

            // Out-of-bounds window fails cleanly
            broimage::RasterF32 bad;
            CHECK(!broimage::decode_geotiff_file_f32(path, bad, {40, 0, 16, 10}));
            CHECK(!broimage::decode_geotiff_file_f32(path, bad, {-1, 0, 4, 4}));

            std::remove(path.c_str());
        }
    }
}

} // namespace

int main() {
    test_float32_predictor3_stripped_and_tiled();
    test_float64_and_signed_ints();
    test_bigtiff_and_geotiff_metadata_and_window();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test_geotiff: all passed\n");
    return 0;
}
