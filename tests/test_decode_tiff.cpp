// TIFF decode: every layout and compression the decoder reads, each built
// in-test from known sample values (tests/tiff_writer.h) and checked pixel
// for pixel through the public API (decode_memory, decode_memory_u16,
// decode_memory_oriented, probe_dimensions_memory, decode_memory_bounded,
// decode_file); then hostile input, which must fail without reading out of
// bounds or allocating what the file claims.
#include "broimage/decode.h"
#include "broimage/encode.h"

#include "tiff_writer.h"

#include <cstdint>
#include <cstdio>
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

using tiffw::Spec;

uint32_t g_seed = 12345;
uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }

std::vector<uint32_t> noise(std::size_t n, uint32_t bits) {
    std::vector<uint32_t> v(n);
    for (auto& x : v) x = rnd() & ((1u << bits) - 1);
    return v;
}

// A smooth gradient (what predictor 2 exists for) with some noise.
std::vector<uint32_t> gradient(uint32_t w, uint32_t h, uint32_t spp, uint32_t bits) {
    std::vector<uint32_t> v(std::size_t(w) * h * spp);
    const uint32_t mask = (1u << bits) - 1;
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
            for (uint32_t c = 0; c < spp; ++c)
                v[(std::size_t(y) * w + x) * spp + c] = ((x * 7 + y * 3 + c * 50) * (mask / 255 ? mask / 255 : 1) + (rnd() & 3)) & mask;
    return v;
}

uint8_t to8(uint32_t v, uint32_t bits) {
    const uint64_t m = (uint64_t(1) << bits) - 1;
    return uint8_t((v * 255ull + m / 2) / m);
}

// The RGBA8 a gray / RGB (+ unassociated alpha) image decodes to.
std::vector<uint8_t> expect8(const Spec& s, const std::vector<uint32_t>& v, bool white_is_zero = false) {
    std::vector<uint8_t> out(std::size_t(s.w) * s.h * 4);
    const uint32_t color = s.photometric == 2 ? 3 : 1;
    const uint32_t m = (1u << s.bits) - 1;
    for (std::size_t i = 0; i < std::size_t(s.w) * s.h; ++i) {
        const uint32_t* p = &v[i * s.spp];
        for (int c = 0; c < 3; ++c) {
            uint32_t x = p[color == 3 ? c : 0];
            if (white_is_zero) x = m - x;
            out[i * 4 + c] = to8(x, s.bits);
        }
        out[i * 4 + 3] = s.spp > color ? to8(p[color], s.bits) : 255;
    }
    return out;
}

bool decodes_to(const std::vector<uint8_t>& tiff, const std::vector<uint8_t>& want, int w, int h) {
    broimage::Image img;
    std::string err;
    if (!broimage::decode_memory(tiff.data(), tiff.size(), img, &err)) {
        std::fprintf(stderr, "  decode failed: %s\n", err.c_str());
        return false;
    }
    if (img.width != w || img.height != h || img.pixels != want) {
        std::fprintf(stderr, "  decoded %dx%d, want %dx%d\n", img.width, img.height, w, h);
        for (std::size_t i = 0; i < want.size() && i < img.pixels.size(); ++i)
            if (want[i] != img.pixels[i]) {
                std::fprintf(stderr, "  first difference at byte %zu: %d, want %d\n", i, img.pixels[i], want[i]);
                break;
            }
        return false;
    }
    return true;
}

void test_uncompressed_both_orders() {
    for (bool le : {true, false}) {
        Spec s;
        s.le = le; s.w = 7; s.h = 5; s.spp = 4; s.photometric = 2; s.rows_per_strip = 2; s.extra_samples = {2};
        const auto v = noise(7 * 5 * 4, 8);
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 7, 5));
        // Missing ExtraSamples: the fourth sample is still alpha.
        s.extra_samples.clear();
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 7, 5));
    }
}

void test_lzw() {
    // Noise big enough to grow the code width to 12 bits and clear the table.
    for (uint32_t predictor : {1u, 2u}) {
        Spec s;
        s.w = 96; s.h = 80; s.spp = 3; s.photometric = 2; s.compression = 5; s.predictor = predictor;
        s.rows_per_strip = 32;
        const auto v = predictor == 1 ? noise(96 * 80 * 3, 8) : gradient(96, 80, 3, 8);
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 96, 80));
        s.le = false;
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 96, 80));
    }
    // Long runs: strings far longer than one byte.
    Spec s;
    s.w = 300; s.h = 40; s.compression = 5;
    std::vector<uint32_t> v(300 * 40, 77);
    for (std::size_t i = 0; i < v.size(); i += 997) v[i] = 3;
    CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 300, 40));
}

void test_sixteen_bit() {
    for (bool le : {true, false}) {
        Spec s;
        s.le = le; s.w = 9; s.h = 6; s.bits = 16; s.spp = 4; s.photometric = 2; s.compression = 5;
        s.predictor = 2; s.extra_samples = {2}; s.rows_per_strip = 4;
        const auto v = noise(9 * 6 * 4, 16);
        const auto tiff = tiffw::make(s, v);
        broimage::ImageU16 u;
        CHECK(broimage::decode_memory_u16(tiff.data(), tiff.size(), u));
        bool exact = u.width == 9 && u.height == 6 && u.pixels.size() == v.size();
        for (std::size_t i = 0; exact && i < v.size(); ++i) exact = u.pixels[i] == v[i];
        CHECK(exact);
        CHECK(decodes_to(tiff, expect8(s, v), 9, 6));
    }
    // 16-bit gray, Deflate.
    Spec s;
    s.w = 5; s.h = 3; s.bits = 16; s.compression = 32946;
    const auto v = noise(15, 16);
    CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 5, 3));
}

void test_deflate_and_packbits() {
    for (uint32_t comp : {8u, 32946u}) {
        Spec s;
        s.w = 33; s.h = 17; s.compression = comp; s.predictor = 2; s.rows_per_strip = 5;
        const auto v = gradient(33, 17, 1, 8);
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 33, 17));
    }
    Spec s;
    s.w = 40; s.h = 9; s.spp = 2; s.compression = 32773; s.extra_samples = {2}; s.rows_per_strip = 4;
    std::vector<uint32_t> v(40 * 9 * 2);
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = (i / 24) % 3 == 0 ? rnd() & 255 : 200;  // runs and literals
    CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 40, 9));
}

void test_low_bit_depths() {
    // 1-bit bilevel, WhiteIsZero, a width that is not a whole number of bytes.
    {
        Spec s;
        s.w = 13; s.h = 4; s.bits = 1; s.photometric = 0;
        const auto v = noise(13 * 4, 1);
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v, true), 13, 4));
        s.fill_order2 = true;  // bits stored LSB first
        s.compression = 32773;
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v, true), 13, 4));
    }
    // 2- and 4-bit gray.
    for (uint32_t bits : {2u, 4u}) {
        Spec s;
        s.w = 11; s.h = 3; s.bits = bits; s.compression = 5;
        const auto v = noise(33, bits);
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 11, 3));
    }
}

void test_palette() {
    for (uint32_t bits : {4u, 8u}) {
        Spec s;
        s.w = 6; s.h = 4; s.bits = bits; s.photometric = 3; s.compression = 5;
        const uint32_t n = 1u << bits;
        s.colormap.resize(3 * n);
        for (uint32_t i = 0; i < 3 * n; ++i) s.colormap[i] = (rnd() & 0xFFFF);
        const auto v = noise(24, bits);
        std::vector<uint8_t> want(24 * 4);
        for (std::size_t i = 0; i < 24; ++i) {
            for (int c = 0; c < 3; ++c) want[i * 4 + c] = uint8_t((s.colormap[c * n + v[i]] * 255u + 32767) / 65535);
            want[i * 4 + 3] = 255;
        }
        CHECK(decodes_to(tiffw::make(s, v), want, 6, 4));
    }
}

void test_tiles_and_planar() {
    // 40x24 in 16x16 tiles: the right and bottom tiles hang over the edge.
    for (uint32_t comp : {1u, 5u, 32946u}) {
        Spec s;
        s.w = 40; s.h = 24; s.spp = 3; s.photometric = 2; s.compression = comp; s.tile_w = 16; s.tile_h = 16;
        s.predictor = comp == 1 ? 1 : 2;
        const auto v = gradient(40, 24, 3, 8);
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 40, 24));
        s.planar = 2;  // tiles per plane
        CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 40, 24));
    }
    // Planar strips, RGBA.
    Spec s;
    s.w = 10; s.h = 7; s.spp = 4; s.photometric = 2; s.planar = 2; s.rows_per_strip = 3; s.compression = 5;
    s.extra_samples = {2};
    const auto v = noise(10 * 7 * 4, 8);
    CHECK(decodes_to(tiffw::make(s, v), expect8(s, v), 10, 7));
}

void test_alpha_and_cmyk() {
    // Associated alpha is straightened, 8 and 16 bits.
    Spec s;
    s.w = 2; s.h = 1; s.spp = 4; s.photometric = 2; s.extra_samples = {1};
    CHECK(decodes_to(tiffw::make(s, {100, 50, 0, 128, 10, 20, 30, 0}), {199, 100, 0, 128, 10, 20, 30, 0}, 2, 1));
    s.bits = 16;
    const auto t16 = tiffw::make(s, {25700, 12850, 0, 32896, 0, 0, 0, 65535});
    broimage::ImageU16 u;
    CHECK(broimage::decode_memory_u16(t16.data(), t16.size(), u));
    CHECK(u.pixels.size() == 8 && u.pixels[0] == 51199 && u.pixels[1] == 25600 && u.pixels[3] == 32896 &&
          u.pixels[7] == 65535);
    // Gray + associated alpha.
    Spec g;
    g.w = 1; g.h = 1; g.spp = 2; g.extra_samples = {1};
    CHECK(decodes_to(tiffw::make(g, {64, 128}), {128, 128, 128, 128}, 1, 1));
    // An extra sample declared unspecified is not alpha.
    g.extra_samples = {0};
    CHECK(decodes_to(tiffw::make(g, {64, 128}), {64, 64, 64, 255}, 1, 1));
    // CMYK.
    Spec c;
    c.w = 2; c.h = 1; c.spp = 4; c.photometric = 5;
    CHECK(decodes_to(tiffw::make(c, {0, 0, 0, 0, 255, 0, 128, 51}), {255, 255, 255, 255, 0, 204, 102, 255}, 2, 1));
}

// JPEG segments: DQT / DHT go to JPEGTables, everything else stays.
void split_jpeg(const std::vector<uint8_t>& jpg, std::vector<uint8_t>& tables, std::vector<uint8_t>& rest) {
    tables = {0xFF, 0xD8};
    rest = {0xFF, 0xD8};
    std::size_t i = 2;
    while (i + 4 <= jpg.size()) {
        const uint8_t m = jpg[i + 1];
        if (m == 0xDA) break;
        const std::size_t len = std::size_t(jpg[i + 2]) << 8 | jpg[i + 3];
        auto& to = (m == 0xDB || m == 0xC4) ? tables : rest;
        to.insert(to.end(), jpg.begin() + long(i), jpg.begin() + long(i + 2 + len));
        i += 2 + len;
    }
    rest.insert(rest.end(), jpg.begin() + long(i), jpg.end());
    tables.insert(tables.end(), {0xFF, 0xD9});
}

void test_jpeg() {
    const int W = 24, H = 20, RPS = 12;
    std::vector<uint8_t> rgb(std::size_t(W) * H * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t((i * 37) ^ (i / 72 * 11));
    // Each strip its own JPEG; the expectation is stb's decode of the same
    // JPEG, so the comparison is exact.
    std::vector<std::vector<uint8_t>> whole, abbreviated;
    std::vector<uint8_t> tables, want(std::size_t(W) * H * 4);
    for (int y0 = 0; y0 < H; y0 += RPS) {
        const int rows = std::min(RPS, H - y0);
        std::vector<uint8_t> jpg;
        CHECK(broimage::encode_jpeg_memory(jpg, rgb.data() + std::size_t(y0) * W * 3, W, rows, 3, 85));
        broimage::Image ref;
        CHECK(broimage::decode_memory(jpg.data(), jpg.size(), ref));
        std::copy(ref.pixels.begin(), ref.pixels.end(), want.begin() + long(std::size_t(y0) * W * 4));
        whole.push_back(jpg);
        std::vector<uint8_t> rest;
        split_jpeg(jpg, tables, rest);  // stb writes the same tables for every strip
        abbreviated.push_back(rest);
    }
    Spec s;
    s.w = W; s.h = H; s.spp = 3; s.photometric = 6; s.compression = 7; s.rows_per_strip = RPS;
    CHECK(decodes_to(tiffw::make_raw_chunks(s, whole), want, W, H));
    s.more = {tiffw::undefined(347, tables)};
    s.le = false;
    CHECK(decodes_to(tiffw::make_raw_chunks(s, abbreviated), want, W, H));

    // Photometric RGB: the same YCbCr-coded gray image read as RGB comes out
    // as (Y, Cb, Cr) = (gray, ~128, ~128).
    std::vector<uint8_t> gray3(16 * 8 * 3);
    for (std::size_t i = 0; i < gray3.size(); ++i) gray3[i] = uint8_t(((i / 3) * 5) & 255);
    std::vector<uint8_t> jpg;
    CHECK(broimage::encode_jpeg_memory(jpg, gray3.data(), 16, 8, 3, 90));
    broimage::Image ycc;
    CHECK(broimage::decode_memory(jpg.data(), jpg.size(), ycc));
    Spec r;
    r.w = 16; r.h = 8; r.spp = 3; r.photometric = 2; r.compression = 7;
    const auto rgbTiff = tiffw::make_raw_chunks(r, {jpg});
    broimage::Image img;
    CHECK(broimage::decode_memory(rgbTiff.data(), rgbTiff.size(), img));
    bool ok = img.pixels.size() == ycc.pixels.size();
    for (std::size_t i = 0; ok && i < 16 * 8; ++i)
        ok = int(img.pixels[i * 4]) - int(ycc.pixels[i * 4]) >= -2 &&
             int(img.pixels[i * 4]) - int(ycc.pixels[i * 4]) <= 2 && img.pixels[i * 4 + 1] >= 126 && img.pixels[i * 4 + 1] <= 130 &&
             img.pixels[i * 4 + 2] >= 126 && img.pixels[i * 4 + 2] <= 130;
    CHECK(ok);
    // Gray JPEG (one sample per pixel).
    Spec g;
    g.w = 16; g.h = 8; g.spp = 1; g.photometric = 1; g.compression = 7;
    const auto grayTiff = tiffw::make_raw_chunks(g, {jpg});
    CHECK(broimage::decode_memory(grayTiff.data(), grayTiff.size(), img));
    ok = img.width == 16 && img.pixels.size() == 16 * 8 * 4;
    for (std::size_t i = 0; ok && i < 16 * 8; ++i) {
        const int d = int(img.pixels[i * 4]) - int(ycc.pixels[i * 4]);
        ok = d >= -1 && d <= 1 && img.pixels[i * 4 + 1] == img.pixels[i * 4];
    }
    CHECK(ok);
}

void test_orientation_probe_pages() {
    // 3x2 RGB with Orientation 6 in IFD0: upright it is 2x3.
    Spec s;
    s.w = 3; s.h = 2; s.spp = 3; s.photometric = 2; s.compression = 5;
    s.more = {tiffw::shorts(274, {6})};
    s.second_page = true;
    std::vector<uint32_t> v(18);
    for (std::size_t i = 0; i < 18; ++i) v[i] = uint32_t(i * 10);
    const auto tiff = tiffw::make(s, v);
    CHECK(broimage::read_exif_orientation(tiff.data(), tiff.size()) == broimage::ExifOrientation::Rotate90CW);
    broimage::Image img;
    CHECK(broimage::decode_memory(tiff.data(), tiff.size(), img) && img.width == 3 && img.height == 2);
    CHECK(broimage::decode_memory_oriented(tiff.data(), tiff.size(), img));
    // Rotate 90 CW: the stored bottom-left pixel (0,1) lands at the top-left.
    CHECK(img.width == 2 && img.height == 3 && img.pixels[0] == 90 && img.pixels[4] == 0);
    int w = 0, h = 0, c = 0;
    CHECK(broimage::probe_dimensions_memory(tiff.data(), tiff.size(), &w, &h, &c) && w == 3 && h == 2 && c == 3);
    Spec a;
    a.w = 4; a.h = 1; a.spp = 2; a.extra_samples = {2};
    const auto ga = tiffw::make(a, noise(8, 8));
    CHECK(broimage::probe_dimensions_memory(ga.data(), ga.size(), &w, &h, &c) && w == 4 && c == 2);

    broimage::Animation anim;
    std::string err;
    CHECK(broimage::decode_memory_bounded(tiff.data(), tiff.size(), {}, anim, &err) && anim.width == 3 &&
          anim.frames.size() == 1);
    broimage::DecodeLimits lim;
    lim.max_width = 2;
    CHECK(!broimage::decode_memory_bounded(tiff.data(), tiff.size(), lim, anim, &err));

    // decode_file and decode_file_oriented read it from disk too.
    const std::string path = "broimage_test_tiff.tif";
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(tiff.data(), 1, tiff.size(), f);
        std::fclose(f);
        CHECK(broimage::decode_file(path, img) && img.width == 3);
        CHECK(broimage::decode_file_oriented(path, img) && img.width == 2 && img.height == 3);
        CHECK(broimage::read_exif_orientation_file(path) == broimage::ExifOrientation::Rotate90CW);
        std::remove(path.c_str());
    } else {
        std::fprintf(stderr, "FAIL: could not write %s\n", path.c_str());
        g_failed++;
    }
}

void test_hostile() {
    broimage::Image img;
    std::string err;
    Spec s;
    s.w = 8; s.h = 8; s.compression = 5;
    const auto v = noise(64, 8);
    const auto good = tiffw::make(s, v);
    // Every truncation fails cleanly (or, if the IFD survives, still does not
    // read out of bounds).
    for (std::size_t n = 0; n < good.size(); ++n) {
        std::vector<uint8_t> cut(good.begin(), good.begin() + long(n));
        broimage::decode_memory(cut.data(), cut.size(), img, &err);
        int w, h, c;
        broimage::probe_dimensions_memory(cut.data(), cut.size(), &w, &h, &c);
    }
    // Corrupt bytes anywhere: never a crash.
    for (std::size_t i = 4; i < good.size(); ++i) {
        std::vector<uint8_t> bad = good;
        bad[i] ^= 0xA5;
        broimage::decode_memory(bad.data(), bad.size(), img, &err);
    }
    // A tiny file that claims 16384x16384 of LZW: refused before allocating.
    Spec huge;
    huge.w = 16384; huge.h = 16384; huge.compression = 5; huge.rows_per_strip = 16384;
    tiffw::Buf b;
    b.b = {'I', 'I'}; b.u16(42); b.u32(0);
    const uint32_t ifd = tiffw::write_ifd(b, {tiffw::longs(256, {16384}), tiffw::longs(257, {16384}),
                                              tiffw::shorts(258, {8}), tiffw::shorts(259, {5}),
                                              tiffw::shorts(262, {1}), tiffw::longs(273, {8}),
                                              tiffw::longs(279, {4})});
    b.put32_at(4, ifd);
    CHECK(!broimage::decode_memory(b.b.data(), b.b.size(), img, &err) && img.pixels.empty());
    // Too many pixels at all.
    tiffw::Buf b2;
    b2.b = {'I', 'I'}; b2.u16(42); b2.u32(0);
    const uint32_t ifd2 = tiffw::write_ifd(b2, {tiffw::longs(256, {1u << 20}), tiffw::longs(257, {1u << 20}),
                                                tiffw::shorts(258, {8}), tiffw::longs(273, {8}),
                                                tiffw::longs(279, {4})});
    b2.put32_at(4, ifd2);
    CHECK(!broimage::decode_memory(b2.b.data(), b2.b.size(), img, &err));
    // Palette without a colour map; float samples; BigTIFF; old JPEG.
    Spec p;
    p.w = 2; p.h = 2; p.photometric = 3;
    auto pal = tiffw::make(p, {0, 1, 2, 3});
    CHECK(!broimage::decode_memory(pal.data(), pal.size(), img, &err));
    Spec f;
    f.w = 1; f.h = 1; f.bits = 16; f.more = {tiffw::shorts(339, {3})};
    auto flt = tiffw::make(f, {0});
    CHECK(!broimage::decode_memory(flt.data(), flt.size(), img, &err));
    const uint8_t big[] = {'I', 'I', 43, 0, 8, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0};
    CHECK(!broimage::decode_memory(big, sizeof big, img, &err));
    Spec o;
    o.w = 1; o.h = 1; o.compression = 6;
    auto oj = tiffw::make(o, {0});
    CHECK(!broimage::decode_memory(oj.data(), oj.size(), img, &err));
    // A strip offset past the end of the file.
    tiffw::Buf b3;
    b3.b = {'I', 'I'}; b3.u16(42); b3.u32(0);
    b3.u32(0x01020304);
    const uint32_t ifd3 = tiffw::write_ifd(b3, {tiffw::longs(256, {2}), tiffw::longs(257, {2}),
                                                tiffw::shorts(258, {8}), tiffw::shorts(259, {5}),
                                                tiffw::longs(273, {100000}), tiffw::longs(279, {4})});
    b3.put32_at(4, ifd3);
    CHECK(!broimage::decode_memory(b3.b.data(), b3.b.size(), img, &err));
    // Truncated LZW data inside an intact file: fails.
    Spec t;
    t.w = 64; t.h = 64; t.compression = 5;
    auto lz = tiffw::make(t, noise(64 * 64, 8));
    // Zero the last 200 bytes of the strip: the code stream turns to garbage
    // or ends early; the decode fails or yields an image, never a crash.
    for (std::size_t i = 10; i < 210 && i < lz.size(); ++i) lz[i] = 0;
    broimage::decode_memory(lz.data(), lz.size(), img, &err);
}

} // namespace

int main() {
    test_uncompressed_both_orders();
    test_lzw();
    test_sixteen_bit();
    test_deflate_and_packbits();
    test_low_bit_depths();
    test_palette();
    test_tiles_and_planar();
    test_alpha_and_cmyk();
    test_jpeg();
    test_orientation_probe_pages();
    test_hostile();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test_decode_tiff: all passed\n");
    return 0;
}
