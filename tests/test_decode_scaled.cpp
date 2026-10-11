// Scaled decode (decode_memory_scaled), the EXIF thumbnail, and the 8-bit
// area resize it finishes with.

#include <broimage/decode.h>
#include <broimage/encode.h>
#include <broimage/exif.h>
#include <broimage/geometric.h>
#include <broimage/heif.h>

#include <stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

// Which path a JPEG takes depends on libjpeg-turbo; the sizes do not.
#if BROIMAGE_HAS_JPEG_TURBO
#define CHECK_DCT(cond) CHECK(cond)
#else
#define CHECK_DCT(cond) do { } while (0)
#endif

namespace {

using broimage::Image;
using broimage::ScaledDecodeInfo;
using broimage::ScaledDecodeOptions;
using broimage::ScaledDecodePath;

constexpr int W = 96, H = 64;  // 3:2, as a camera's photo
constexpr int TW = 40, TH = 30; // 4:3, as its EXIF thumbnail

void put16(std::vector<uint8_t>& v, uint32_t x) { v.push_back(uint8_t(x)); v.push_back(uint8_t(x >> 8)); }
void put32(std::vector<uint8_t>& v, uint32_t x) { put16(v, x & 0xFFFF); put16(v, x >> 16); }
void entry(std::vector<uint8_t>& v, uint16_t tag, uint16_t type, uint32_t value) {
    put16(v, tag);
    put16(v, type);
    put32(v, 1);
    if (type == 3) { put16(v, value); put16(v, 0); }
    else put32(v, value);
}

// A smooth RGB gradient, so a DCT-scaled decode and a resized full one agree.
std::vector<uint8_t> gradient(int w, int h) {
    std::vector<uint8_t> px(std::size_t(w) * h * 3);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t* p = px.data() + (std::size_t(y) * w + x) * 3;
            p[0] = uint8_t(x * 255 / (w - 1));
            p[1] = uint8_t(y * 255 / (h - 1));
            p[2] = 96;
        }
    return px;
}

// The thumbnail: red, padded to 4:3 with black bars above and below.
std::vector<uint8_t> thumbnail_jpeg() {
    std::vector<uint8_t> px(std::size_t(TW) * TH * 3, 0);
    const int content = int(std::lround(TW * double(H) / W));  // 27 rows
    const int top = (TH - content) / 2;
    for (int y = top; y < top + content; ++y)
        for (int x = 0; x < TW; ++x) px[(std::size_t(y) * TW + x) * 3] = 230;
    std::vector<uint8_t> jpg;
    broimage::encode_jpeg_memory(jpg, px.data(), TW, TH, 3, 95);
    return jpg;
}

// A 96 x 64 gradient JPEG whose APP1 carries `orientation` and, when asked,
// an IFD1 thumbnail.
std::vector<uint8_t> photo(int orientation, bool with_thumbnail) {
    std::vector<uint8_t> main;
    const std::vector<uint8_t> px = gradient(W, H);
    broimage::encode_jpeg_memory(main, px.data(), W, H, 3, 95);

    const std::vector<uint8_t> thumb = with_thumbnail ? thumbnail_jpeg() : std::vector<uint8_t>{};
    std::vector<uint8_t> tiff = {'I', 'I', 42, 0};
    put32(tiff, 8);
    put16(tiff, 1);  // IFD0: Orientation
    entry(tiff, 0x0112, 3, uint32_t(orientation));
    put32(tiff, with_thumbnail ? 26 : 0);
    if (with_thumbnail) {
        put16(tiff, 2);  // IFD1 at 26: the JPEG thumbnail at 56
        entry(tiff, 0x0201, 4, 56);
        entry(tiff, 0x0202, 4, uint32_t(thumb.size()));
        put32(tiff, 0);
        tiff.insert(tiff.end(), thumb.begin(), thumb.end());
    }
    std::vector<uint8_t> app1 = {0xFF, 0xE1};
    const std::size_t len = 2 + 6 + tiff.size();
    app1.push_back(uint8_t(len >> 8));
    app1.push_back(uint8_t(len));
    app1.insert(app1.end(), {'E', 'x', 'i', 'f', 0, 0});
    app1.insert(app1.end(), tiff.begin(), tiff.end());
    main.insert(main.begin() + 2, app1.begin(), app1.end());
    return main;
}

bool scaled(const std::vector<uint8_t>& f, int bw, int bh, broimage::ScaleFit fit, bool embedded, Image& out,
            ScaledDecodeInfo& info) {
    ScaledDecodeOptions o;
    o.width = bw;
    o.height = bh;
    o.fit = fit;
    o.embedded_thumbnail = embedded;
    std::string why;
    const bool ok = broimage::decode_memory_scaled(f.data(), f.size(), o, out, &why, &info);
    if (!ok) std::fprintf(stderr, "  decode_memory_scaled: %s\n", why.c_str());
    return ok && out.channels == 4 && out.pixels.size() == std::size_t(out.width) * out.height * 4;
}

// The largest channel difference between `a` and the full oriented decode of
// `f` area-resized to a's size.
int diff_from_full(const std::vector<uint8_t>& f, const Image& a) {
    Image full;
    if (!broimage::decode_memory_oriented(f.data(), f.size(), full)) return 999;
    std::vector<uint8_t> ref(std::size_t(a.width) * a.height * 4);
    broimage::resize_hwc_u8(full.pixels.data(), full.width, full.height, 4, ref.data(), a.width, a.height,
                            broimage::Filter::Area);
    int worst = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::abs(int(ref[i]) - int(a.pixels[i])));
    return worst;
}

void test_paths() {
    const std::vector<uint8_t> f = photo(1, true);
    Image out;
    ScaledDecodeInfo info;

    // No box: the image itself.
    CHECK(scaled(f, 0, 0, broimage::ScaleFit::Contain, true, out, info));
    CHECK(out.width == W && out.height == H);
    CHECK(info.path == ScaledDecodePath::Full && info.source_width == W && info.source_height == H);

    // 48 x 48: bigger than the thumbnail, so the IDCT at 4/8 gives 48 x 32.
    CHECK(scaled(f, 48, 48, broimage::ScaleFit::Contain, true, out, info));
    CHECK(out.width == 48 && out.height == 32);
    CHECK_DCT(info.path == ScaledDecodePath::Dct && info.dct_scale == 4);
    CHECK_DCT(info.decoded_width == 48 && info.decoded_height == 32);
    CHECK(diff_from_full(f, out) <= 24);

    // Cover: at least 48 x 48, so 72 x 48 at 6/8.
    CHECK(scaled(f, 48, 48, broimage::ScaleFit::Cover, true, out, info));
    CHECK(out.width == 72 && out.height == 48);
    CHECK_DCT(info.path == ScaledDecodePath::Dct && info.dct_scale == 6);

    // 20 x 20 fits in the 40 x 27 of thumbnail left once its bars are cut: red.
    CHECK(scaled(f, 20, 20, broimage::ScaleFit::Contain, true, out, info));
    CHECK(out.width == 20 && out.height == 13);
    CHECK(info.path == ScaledDecodePath::ExifThumbnail);
    CHECK(info.decoded_width == TW && info.decoded_height == 27);
    CHECK(info.source_width == W && info.source_height == H);
    {
        int dark = 0;  // a bar left behind would make a row near black
        for (int y = 0; y < out.height; ++y) {
            const uint8_t* p = out.pixels.data() + (std::size_t(y) * out.width + out.width / 2) * 4;
            if (p[0] < 150) ++dark;
        }
        CHECK(dark == 0);
    }

    // The same box with the thumbnail declined: the IDCT at 2/8, then 20 x 13.
    CHECK(scaled(f, 20, 20, broimage::ScaleFit::Contain, false, out, info));
    CHECK(out.width == 20 && out.height == 13);
    CHECK_DCT(info.path == ScaledDecodePath::Dct && info.dct_scale == 2);
    CHECK(diff_from_full(f, out) <= 24);

    // Width only.
    CHECK(scaled(f, 12, 0, broimage::ScaleFit::Contain, false, out, info));
    CHECK(out.width == 12 && out.height == 8);
    CHECK_DCT(info.dct_scale == 1);
}

void test_orientation() {
    // Rotate90CW: 64 x 96 upright. The box is upright too.
    const std::vector<uint8_t> f = photo(6, true);
    Image out;
    ScaledDecodeInfo info;
    CHECK(scaled(f, 32, 48, broimage::ScaleFit::Contain, false, out, info));
    CHECK(out.width == 32 && out.height == 48);
    CHECK(info.source_width == H && info.source_height == W);
    CHECK_DCT(info.path == ScaledDecodePath::Dct && info.dct_scale == 4);
    CHECK(diff_from_full(f, out) <= 24);

    CHECK(scaled(f, 20, 20, broimage::ScaleFit::Contain, true, out, info));
    CHECK(info.path == ScaledDecodePath::ExifThumbnail);
    CHECK(out.width == 13 && out.height == 20);

    Image thumb;
    CHECK(broimage::decode_exif_thumbnail(f.data(), f.size(), thumb));
    CHECK(thumb.width == 27 && thumb.height == TW);
}

void test_exif_thumbnail() {
    const std::vector<uint8_t> f = photo(1, true);
    const uint8_t* jpeg = nullptr;
    std::size_t n = 0;
    CHECK(broimage::find_exif_thumbnail(f.data(), f.size(), jpeg, n));
    CHECK(jpeg && n == thumbnail_jpeg().size() && jpeg[0] == 0xFF && jpeg[1] == 0xD8);
    Image t;
    CHECK(broimage::decode_exif_thumbnail(f.data(), f.size(), t));
    CHECK(t.width == TW && t.height == 27);

    const std::vector<uint8_t> none = photo(1, false);
    CHECK(!broimage::find_exif_thumbnail(none.data(), none.size(), jpeg, n));
    std::string why;
    CHECK(!broimage::decode_exif_thumbnail(none.data(), none.size(), t, &why));
    CHECK(t.empty() && !why.empty());
    // Without a thumbnail a tiny box goes to the IDCT.
    Image out;
    ScaledDecodeInfo info;
    CHECK(scaled(none, 20, 20, broimage::ScaleFit::Contain, true, out, info));
    CHECK_DCT(info.path == ScaledDecodePath::Dct);
}

void test_other_formats() {
    std::vector<uint8_t> rgba(50 * 40 * 4, 200), png;
    CHECK(broimage::encode_png_memory(png, rgba.data(), 50, 40, 4));
    Image out;
    ScaledDecodeInfo info;
    CHECK(scaled(png, 25, 25, broimage::ScaleFit::Contain, true, out, info));
    CHECK(out.width == 25 && out.height == 20);
    CHECK(info.path == ScaledDecodePath::Full && info.decoded_width == 50);
    CHECK(out.pixels[0] == 200);

    const std::vector<uint8_t> junk = {1, 2, 3, 4, 5, 6, 7, 8};
    std::string why;
    ScaledDecodeOptions o;
    o.width = 10;
    CHECK(!broimage::decode_memory_scaled(junk.data(), junk.size(), o, out, &why));
    CHECK(out.empty() && !why.empty());
}

// ---- CR3 -------------------------------------------------------------------

void be32(std::vector<uint8_t>& v, uint32_t x) {
    for (int s = 24; s >= 0; s -= 8) v.push_back(uint8_t(x >> s));
}

std::vector<uint8_t> box(const char* type, const std::vector<uint8_t>& body, const uint8_t* uuid = nullptr) {
    std::vector<uint8_t> b;
    be32(b, uint32_t(8 + (uuid ? 16 : 0) + body.size()));
    b.insert(b.end(), type, type + 4);
    if (uuid) b.insert(b.end(), uuid, uuid + 16);
    b.insert(b.end(), body.begin(), body.end());
    return b;
}

std::vector<uint8_t> cat(std::initializer_list<std::vector<uint8_t>> parts) {
    std::vector<uint8_t> out;
    for (const auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

std::vector<uint8_t> jpeg_of(int w, int h, uint8_t red) {
    std::vector<uint8_t> px(std::size_t(w) * h * 3, 40), jpg;
    for (std::size_t i = 0; i < px.size(); i += 3) px[i] = red;
    broimage::encode_jpeg_memory(jpg, px.data(), w, h, 3, 95);
    return jpg;
}

// A CR3 as Canon lays it out (decode_cr3.h): a full-size 96 x 64 JPEG in mdat
// (red 200), a 48 x 32 PRVW (red 120), a 40 x 30 THMB padded to 4:3 (red
// 230), and CMT1 saying `orientation`.
std::vector<uint8_t> cr3(int orientation) {
    static const uint8_t canon[16] = {0x85, 0xc0, 0xb6, 0x87, 0x82, 0x0f, 0x11, 0xe0,
                                      0x81, 0x11, 0xf4, 0xce, 0x46, 0x2b, 0x6a, 0x48};
    static const uint8_t prvw_uuid[16] = {0xea, 0xf4, 0x2b, 0x5e, 0x1c, 0x98, 0x4b, 0x88,
                                          0xb9, 0xfb, 0xb7, 0xdc, 0x40, 0x6e, 0x4d, 0x16};
    std::vector<uint8_t> cmt1 = {'I', 'I', 42, 0};
    put32(cmt1, 8);
    put16(cmt1, 1);
    entry(cmt1, 0x0112, 3, uint32_t(orientation));
    put32(cmt1, 0);

    const std::vector<uint8_t> thumb_jpg = thumbnail_jpeg();
    std::vector<uint8_t> thmb = {0, 0, 0, 0, 0, TW, 0, TH};
    be32(thmb, uint32_t(thumb_jpg.size()));
    thmb.insert(thmb.end(), {0, 1, 0, 0});
    thmb.insert(thmb.end(), thumb_jpg.begin(), thumb_jpg.end());

    const std::vector<uint8_t> prev_jpg = jpeg_of(48, 32, 120);
    std::vector<uint8_t> prvw = {0, 0, 0, 0, 0, 1, 0, 48, 0, 32, 0, 1};
    be32(prvw, uint32_t(prev_jpg.size()));
    prvw.insert(prvw.end(), prev_jpg.begin(), prev_jpg.end());

    const std::vector<uint8_t> full_jpg = jpeg_of(W, H, 200);
    std::vector<uint8_t> tkhd(84, 0);
    tkhd[76 + 0] = 0; tkhd[76 + 1] = uint8_t(W);   // 16.16 fixed point
    tkhd[80 + 0] = 0; tkhd[80 + 1] = uint8_t(H);
    std::vector<uint8_t> stsz = {0, 0, 0, 0}, co64 = {0, 0, 0, 0};
    be32(stsz, uint32_t(full_jpg.size()));
    be32(stsz, 1);
    be32(co64, 1);
    auto build = [&](uint32_t mdat_at) {
        std::vector<uint8_t> c = co64;
        be32(c, 0);
        be32(c, mdat_at);
        const auto trak = box("trak", cat({box("tkhd", tkhd),
                                          box("mdia", box("minf", box("stbl", cat({box("stsz", stsz), box("co64", c)}))))}));
        const auto moov = box("moov", cat({box("uuid", cat({box("CMT1", cmt1), box("THMB", thmb)}), canon), trak}));
        const auto ftyp = box("ftyp", {'c', 'r', 'x', ' ', 0, 0, 0, 1, 'c', 'r', 'x', ' ', 'i', 's', 'o', 'm'});
        const auto prev = box("uuid", cat({std::vector<uint8_t>(8, 0), box("PRVW", prvw)}), prvw_uuid);
        return cat({ftyp, moov, prev});
    };
    std::vector<uint8_t> head = build(0);
    const uint32_t at = uint32_t(head.size() + 8);  // the JPEG starts after mdat's header
    std::vector<uint8_t> file = build(at);
    const auto mdat = box("mdat", full_jpg);
    file.insert(file.end(), mdat.begin(), mdat.end());
    return file;
}

int red_at_center(const Image& img) {
    return img.pixels[(std::size_t(img.height / 2) * img.width + img.width / 2) * 4];
}

void test_cr3() {
    const std::vector<uint8_t> f = cr3(1);
    CHECK(broimage::can_decode("cr3") && broimage::can_decode("image/x-canon-cr3"));
    int w = 0, h = 0, c = 0;
    CHECK(broimage::probe_dimensions_memory(f.data(), f.size(), &w, &h, &c) && w == W && h == H);

    // decode_memory: the full-size JPEG.
    Image full;
    CHECK(broimage::decode_memory_oriented(f.data(), f.size(), full));
    CHECK(full.width == W && full.height == H && std::abs(red_at_center(full) - 200) < 12);

    Image out;
    ScaledDecodeInfo info;
    // 20 px: the THMB, its bars cut off.
    CHECK(scaled(f, 20, 20, broimage::ScaleFit::Contain, true, out, info));
    CHECK(info.path == ScaledDecodePath::RawPreview && out.width == 20 && out.height == 13);
    CHECK(info.source_width == W && info.source_height == H);
    CHECK(std::abs(red_at_center(out) - 230) < 12);
    // 44 px: wider than THMB's 40, within PRVW's 48 x 32.
    CHECK(scaled(f, 44, 44, broimage::ScaleFit::Contain, true, out, info));
    CHECK(info.path == ScaledDecodePath::RawPreview && out.width == 44 && out.height == 29);
    CHECK(std::abs(red_at_center(out) - 120) < 12);
    // 64 px: the full-size JPEG, at 6/8 (72 x 48 covers 64 x 43).
    CHECK(scaled(f, 64, 64, broimage::ScaleFit::Contain, true, out, info));
    CHECK(info.path == ScaledDecodePath::RawPreview && out.width == 64 && out.height == 43);
    CHECK_DCT(info.dct_scale == 6);
    CHECK(std::abs(red_at_center(out) - 200) < 12);

    // Upright by CMT1.
    const std::vector<uint8_t> r = cr3(8);
    CHECK(scaled(r, 40, 40, broimage::ScaleFit::Contain, true, out, info));
    CHECK(out.width == 27 && out.height == 40 && info.source_width == H && info.source_height == W);
    broimage::ExifInfo e;
    CHECK(broimage::read_exif(r.data(), r.size(), e) && e.orientation && *e.orientation == 8);

    // From a file: the head is enough for the PRVW, the full-size JPEG needs it all.
    const std::string path = "test_decode_scaled.cr3";
    if (std::FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(f.data(), 1, f.size(), fp);
        std::fclose(fp);
        ScaledDecodeOptions o;
        o.width = o.height = 44;
        CHECK(broimage::decode_file_scaled(path, o, out, nullptr, &info) && out.width == 44);
        CHECK(info.path == ScaledDecodePath::RawPreview && std::abs(red_at_center(out) - 120) < 12);
        o.width = o.height = 64;
        CHECK(broimage::decode_file_scaled(path, o, out, nullptr, &info) && out.width == 64);
        CHECK(std::abs(red_at_center(out) - 200) < 12);
        std::remove(path.c_str());
    }
}

// The JPEG decoder against stb_image: the same pixels within IDCT rounding.
void test_matches_stb() {
    const std::vector<uint8_t> f = photo(1, false);
    Image ours;
    CHECK(broimage::decode_memory(f.data(), f.size(), ours));
    int w = 0, h = 0, c = 0;
    stbi_uc* px = stbi_load_from_memory(f.data(), int(f.size()), &w, &h, &c, 4);
    CHECK(px && w == ours.width && h == ours.height);
    if (px && w == ours.width && h == ours.height) {
        int worst = 0;
        for (std::size_t i = 0; i < ours.pixels.size(); ++i)
            worst = std::max(worst, std::abs(int(px[i]) - int(ours.pixels[i])));
        CHECK(worst <= 8);
    }
    stbi_image_free(px);
}

// The 8-bit area resize against the float one it stands in for.
void test_area_u8() {
    uint32_t seed = 12345;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return uint8_t(seed >> 24); };
    const struct { int sw, sh, dw, dh, ch; } cases[] = {
        {97, 61, 23, 17, 4}, {97, 61, 23, 17, 3}, {64, 64, 64, 32, 1}, {640, 480, 641 / 3, 160, 4},
        {2100, 1100, 700, 367, 4},  // big enough to split across threads
    };
    for (const auto& k : cases) {
        std::vector<uint8_t> src(std::size_t(k.sw) * k.sh * k.ch);
        for (auto& v : src) v = rnd();
        std::vector<float> sf(src.begin(), src.end()), df(std::size_t(k.dw) * k.dh * k.ch);
        broimage::resize_hwc_f32(sf.data(), k.sw, k.sh, k.ch, df.data(), k.dw, k.dh, broimage::Filter::Area);
        std::vector<uint8_t> d(df.size());
        broimage::resize_hwc_u8(src.data(), k.sw, k.sh, k.ch, d.data(), k.dw, k.dh, broimage::Filter::Area);
        int worst = 0;
        for (std::size_t i = 0; i < d.size(); ++i)
            worst = std::max(worst, std::abs(int(d[i]) - int(std::lround(std::clamp(df[i], 0.0f, 255.0f)))));
        CHECK(worst <= 1);
    }
}

}  // namespace

int main() {
    test_paths();
    test_orientation();
    test_exif_thumbnail();
    test_other_formats();
    test_cr3();
    test_matches_stb();
    test_area_u8();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test_decode_scaled: ok\n");
    return 0;
}
