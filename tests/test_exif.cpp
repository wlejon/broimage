// EXIF metadata (exif.h): one EXIF block with every field read_exif knows,
// carried in each container (JPEG APP1, TIFF IFD0, WebP EXIF chunk, PNG
// eXIf), in both byte orders; the exposure display string; absent fields
// stay absent; read_exif_orientation finds the tag in every container.
#include "broimage/decode.h"
#include "broimage/encode.h"
#include "broimage/exif.h"

#include "tiff_writer.h"

#include <cmath>
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

using tiffw::Entry;

bool approx(const std::optional<double>& v, double want, double eps = 1e-6) {
    return v && std::fabs(*v - want) < eps;
}

std::vector<Entry> exif_entries() {
    return {tiffw::rationals(0x829A, {1, 250}), tiffw::rationals(0x829D, {28, 10}), tiffw::shorts(0x8827, {400}),
            tiffw::ascii(0x9003, "2024:05:06 14:30:00"), tiffw::ascii(0x9011, "+02:00"),
            tiffw::rationals(0x920A, {50, 1}), tiffw::shorts(0xA405, {75}), tiffw::ascii(0xA433, "Canon"),
            tiffw::ascii(0xA434, "RF50mm F1.8 STM")};
}

std::vector<Entry> gps_entries() {
    return {tiffw::ascii(1, "N"), tiffw::rationals(2, {40, 1, 26, 1, 4614, 100}),
            tiffw::ascii(3, "W"), tiffw::rationals(4, {79, 1, 58, 1, 5604, 100}),
            Entry{5, 1, {}, {1}}, tiffw::rationals(6, {125, 10})};
}

// A TIFF-structured block: `image` entries (and pixel bytes) for a real
// TIFF, the Exif and GPS sub-IFDs, IFD0 with Make/Model/Orientation.
std::vector<uint8_t> build_block(bool le, std::vector<Entry> ifd0, const std::vector<uint8_t>& pixels = {},
                                 std::vector<Entry> ex = exif_entries(), std::vector<Entry> gps = gps_entries()) {
    tiffw::Buf b;
    b.le = le;
    b.b = le ? std::vector<uint8_t>{'I', 'I'} : std::vector<uint8_t>{'M', 'M'};
    b.u16(42);
    b.u32(0);
    if (!pixels.empty()) {
        ifd0.push_back(tiffw::longs(273, {uint32_t(b.b.size())}));
        ifd0.push_back(tiffw::longs(279, {uint32_t(pixels.size())}));
        b.bytes(pixels);
    }
    if (!ex.empty()) ifd0.push_back(tiffw::longs(0x8769, {tiffw::write_ifd(b, ex)}));
    if (!gps.empty()) ifd0.push_back(tiffw::longs(0x8825, {tiffw::write_ifd(b, gps)}));
    const uint32_t ifd = tiffw::write_ifd(b, ifd0);
    b.put32_at(4, ifd);
    return b.b;
}

std::vector<Entry> camera() {
    return {tiffw::ascii(0x010F, "Canon"), tiffw::ascii(0x0110, "EOS R5"), tiffw::shorts(0x0112, {6})};
}

void check_full(const broimage::ExifInfo& e, const char* where) {
    std::fprintf(stderr, "  checking %s\n", where);
    CHECK(e.make == "Canon");
    CHECK(e.model == "EOS R5");
    CHECK(e.lens_make == "Canon");
    CHECK(e.lens_model == "RF50mm F1.8 STM");
    CHECK(approx(e.exposure_time, 0.004));
    CHECK(e.exposure_time_text == "1/250");
    CHECK(approx(e.f_number, 2.8));
    CHECK(e.iso && *e.iso == 400);
    CHECK(approx(e.focal_length, 50));
    CHECK(approx(e.focal_length_35mm, 75));
    CHECK(e.date_taken == "2024-05-06T14:30:00+02:00");
    CHECK(e.offset_time == "+02:00");
    CHECK(e.sub_sec_time.empty());
    CHECK(approx(e.latitude, 40 + 26 / 60.0 + 46.14 / 3600));
    CHECK(approx(e.longitude, -(79 + 58 / 60.0 + 56.04 / 3600)));
    CHECK(approx(e.altitude, -12.5));
    CHECK(e.orientation && *e.orientation == 6);
}

uint32_t crc32(const uint8_t* p, std::size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1)));
    }
    return ~c;
}

std::vector<uint8_t> in_jpeg(const std::vector<uint8_t>& block) {
    const uint8_t rgb[4 * 4 * 3] = {};
    std::vector<uint8_t> jpg;
    broimage::encode_jpeg_memory(jpg, rgb, 4, 4, 3, 90);
    std::vector<uint8_t> app1 = {0xFF, 0xE1, 0, 0, 'E', 'x', 'i', 'f', 0, 0};
    app1.insert(app1.end(), block.begin(), block.end());
    const std::size_t len = app1.size() - 2;
    app1[2] = uint8_t(len >> 8);
    app1[3] = uint8_t(len);
    // After an APP0 JFIF and an XMP-style APP1, as cameras write them.
    std::vector<uint8_t> xmp = {0xFF, 0xE1, 0, 12, 'h', 't', 't', 'p', ':', '/', '/', 'n', 's', '.'};
    std::vector<uint8_t> out(jpg.begin(), jpg.begin() + 2);
    std::size_t app0_end = 2;
    if (jpg[2] == 0xFF && jpg[3] == 0xE0) app0_end = 4 + (std::size_t(jpg[4]) << 8 | jpg[5]);
    out.insert(out.end(), jpg.begin() + 2, jpg.begin() + long(app0_end));
    out.insert(out.end(), xmp.begin(), xmp.end());
    out.insert(out.end(), app1.begin(), app1.end());
    out.insert(out.end(), jpg.begin() + long(app0_end), jpg.end());
    return out;
}

std::vector<uint8_t> in_webp(const std::vector<uint8_t>& block, bool prefixed) {
    std::vector<uint8_t> payload;
    if (prefixed) payload = {'E', 'x', 'i', 'f', 0, 0};
    payload.insert(payload.end(), block.begin(), block.end());
    std::vector<uint8_t> w = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P'};
    auto chunk = [&](const char* id, const std::vector<uint8_t>& data) {
        w.insert(w.end(), id, id + 4);
        const uint32_t n = uint32_t(data.size());
        w.insert(w.end(), {uint8_t(n), uint8_t(n >> 8), uint8_t(n >> 16), uint8_t(n >> 24)});
        w.insert(w.end(), data.begin(), data.end());
        if (n & 1) w.push_back(0);
    };
    chunk("VP8X", {0x08, 0, 0, 0, 3, 0, 0, 3, 0, 0});  // EXIF flag, 4x4 canvas
    chunk("ICCP", {1, 2, 3});                           // an odd-sized chunk before it
    chunk("EXIF", payload);
    const uint32_t riff = uint32_t(w.size() - 8);
    w[4] = uint8_t(riff); w[5] = uint8_t(riff >> 8); w[6] = uint8_t(riff >> 16); w[7] = uint8_t(riff >> 24);
    return w;
}

std::vector<uint8_t> in_png(const std::vector<uint8_t>& block) {
    const uint8_t rgba[2 * 2 * 4] = {};
    std::vector<uint8_t> png;
    broimage::encode_png_memory(png, rgba, 2, 2, 4);
    std::vector<uint8_t> chunk = {uint8_t(block.size() >> 24), uint8_t(block.size() >> 16),
                                  uint8_t(block.size() >> 8), uint8_t(block.size()), 'e', 'X', 'I', 'f'};
    chunk.insert(chunk.end(), block.begin(), block.end());
    const uint32_t crc = crc32(chunk.data() + 4, chunk.size() - 4);
    chunk.insert(chunk.end(), {uint8_t(crc >> 24), uint8_t(crc >> 16), uint8_t(crc >> 8), uint8_t(crc)});
    std::vector<uint8_t> out(png.begin(), png.begin() + 33);  // signature + IHDR
    out.insert(out.end(), chunk.begin(), chunk.end());
    out.insert(out.end(), png.begin() + 33, png.end());
    return out;
}

void test_containers() {
    for (bool le : {true, false}) {
        const auto block = build_block(le, camera());
        broimage::ExifInfo e;
        auto jpg = in_jpeg(block);
        CHECK(broimage::read_exif(jpg.data(), jpg.size(), e));
        check_full(e, le ? "JPEG (II)" : "JPEG (MM)");
        broimage::Image img;
        CHECK(broimage::decode_memory(jpg.data(), jpg.size(), img) && img.width == 4);  // still a JPEG

        // A TIFF image: 2x1 RGB, the camera tags in its own IFD0.
        auto image = camera();
        for (const Entry& x : std::vector<Entry>{tiffw::longs(256, {2}), tiffw::longs(257, {1}),
                                                  tiffw::shorts(258, {8, 8, 8}), tiffw::shorts(262, {2}),
                                                  tiffw::shorts(277, {3})})
            image.push_back(x);
        const auto tif = build_block(le, image, {10, 20, 30, 40, 50, 60});
        CHECK(broimage::read_exif(tif.data(), tif.size(), e));
        check_full(e, "TIFF");
        CHECK(broimage::decode_memory_oriented(tif.data(), tif.size(), img) && img.width == 1 && img.height == 2);

        for (bool prefixed : {false, true}) {
            auto webp = in_webp(block, prefixed);
            CHECK(broimage::read_exif(webp.data(), webp.size(), e));
            check_full(e, "WebP");
            CHECK(broimage::read_exif_orientation(webp.data(), webp.size()) == broimage::ExifOrientation::Rotate90CW);
        }
        auto png = in_png(block);
        CHECK(broimage::read_exif(png.data(), png.size(), e));
        check_full(e, "PNG");
        CHECK(broimage::read_exif_orientation(png.data(), png.size()) == broimage::ExifOrientation::Rotate90CW);
        CHECK(broimage::decode_memory(png.data(), png.size(), img) && img.width == 2);
        CHECK(broimage::decode_memory_oriented(png.data(), png.size(), img) && img.width == 2 && img.height == 2);
    }
}

std::string exposure(uint32_t num, uint32_t den) {
    const auto block = build_block(true, {}, {}, {tiffw::rationals(0x829A, {num, den})}, {});
    broimage::ExifInfo e;
    broimage::read_exif(block.data(), block.size(), e);
    return e.exposure_time_text;
}

void test_exposure_text() {
    CHECK(exposure(1, 250) == "1/250");
    CHECK(exposure(10, 2500) == "1/250");
    CHECK(exposure(1, 3) == "1/3");
    CHECK(exposure(1, 2) == "1/2");
    CHECK(exposure(3, 10) == "0.3");
    CHECK(exposure(5, 2) == "2.5");
    CHECK(exposure(30, 1) == "30");
    CHECK(exposure(1, 1) == "1");
    CHECK(exposure(10, 75) == "1/8");  // 0.1333: under a quarter second rounds to 1/N
    CHECK(exposure(0, 1).empty());
}

void test_absent_and_missing() {
    // Make only: everything else absent.
    const auto block = build_block(false, {tiffw::ascii(0x010F, "Nikon  ")}, {}, {}, {});
    broimage::ExifInfo e;
    CHECK(broimage::read_exif(block.data(), block.size(), e));
    CHECK(e.make == "Nikon" && e.model.empty() && e.lens_model.empty());
    CHECK(!e.exposure_time && e.exposure_time_text.empty() && !e.f_number && !e.iso && !e.focal_length &&
          !e.focal_length_35mm && e.date_taken.empty() && e.offset_time.empty() && e.sub_sec_time.empty() &&
          !e.latitude && !e.longitude && !e.altitude && !e.orientation);
    // A date without an offset; the all-zero placeholder; ISOSpeed fallback;
    // a coordinate without its reference.
    const auto b2 = build_block(true, {},  {},
                                {tiffw::ascii(0x9003, "2023:12:31 23:59:58"), tiffw::longs(0x8833, {3200})},
                                {tiffw::rationals(2, {1, 1, 0, 1, 0, 1}), tiffw::rationals(6, {100, 1})});
    CHECK(broimage::read_exif(b2.data(), b2.size(), e));
    CHECK(e.date_taken == "2023-12-31T23:59:58" && e.offset_time.empty());
    CHECK(e.iso && *e.iso == 3200);
    CHECK(!e.latitude && approx(e.altitude, 100));
    const auto b3 = build_block(true, {}, {}, {tiffw::ascii(0x9003, "0000:00:00 00:00:00")}, {});
    CHECK(broimage::read_exif(b3.data(), b3.size(), e) && e.date_taken.empty());

    // No EXIF at all.
    std::vector<uint8_t> jpg;
    const uint8_t rgb[12] = {};
    broimage::encode_jpeg_memory(jpg, rgb, 2, 2, 3, 90);
    CHECK(!broimage::read_exif(jpg.data(), jpg.size(), e));
    CHECK(broimage::read_exif_orientation(jpg.data(), jpg.size()) == broimage::ExifOrientation::Normal);
    const uint8_t junk[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    CHECK(!broimage::read_exif(junk, sizeof junk, e));
    CHECK(!broimage::read_exif(nullptr, 0, e));

    // Hostile: every truncation and corruption of a full JPEG is survived.
    const auto full = in_jpeg(build_block(true, camera()));
    for (std::size_t n = 0; n < full.size(); n += 3) broimage::read_exif(full.data(), n, e);
    for (std::size_t i = 0; i < 400 && i < full.size(); ++i) {
        auto bad = full;
        bad[i] ^= 0x5A;
        broimage::read_exif(bad.data(), bad.size(), e);
    }
}

void test_subsec_time() {
    for (bool le : {true, false}) {
        // Tag 0x9003, 0x9291 and 0x9011 together.
        const auto b1 = build_block(le, {}, {},
                                    {tiffw::ascii(0x9003, "2024:05:06 14:30:00"),
                                     tiffw::ascii(0x9291, "25"),
                                     tiffw::ascii(0x9011, "+02:00")},
                                    {});
        broimage::ExifInfo e;
        CHECK(broimage::read_exif(b1.data(), b1.size(), e));
        CHECK(e.date_taken == "2024-05-06T14:30:00.25+02:00");
        CHECK(e.sub_sec_time == "25");
        CHECK(e.offset_time == "+02:00");

        // Negative offset with fraction.
        const auto b_neg = build_block(le, {}, {},
                                       {tiffw::ascii(0x9003, "2024:05:06 14:30:00"),
                                        tiffw::ascii(0x9291, "25"),
                                        tiffw::ascii(0x9011, "-05:00")},
                                       {});
        CHECK(broimage::read_exif(b_neg.data(), b_neg.size(), e));
        CHECK(e.date_taken == "2024-05-06T14:30:00.25-05:00");
        CHECK(e.sub_sec_time == "25");
        CHECK(e.offset_time == "-05:00");

        // SubSec without offset; exact digits kept (no padding or rounding).
        const auto b2 = build_block(le, {}, {},
                                    {tiffw::ascii(0x9003, "2024:05:06 14:30:00"),
                                     tiffw::ascii(0x9291, "005")},
                                    {});
        CHECK(broimage::read_exif(b2.data(), b2.size(), e));
        CHECK(e.date_taken == "2024-05-06T14:30:00.005");
        CHECK(e.sub_sec_time == "005");
        CHECK(e.offset_time.empty());

        // Trailing spaces are trimmed.
        const auto b_spaces = build_block(le, {}, {},
                                          {tiffw::ascii(0x9003, "2024:05:06 14:30:00"),
                                           tiffw::ascii(0x9291, "42   "),
                                           tiffw::ascii(0x9011, "+01:00")},
                                          {});
        CHECK(broimage::read_exif(b_spaces.data(), b_spaces.size(), e));
        CHECK(e.date_taken == "2024-05-06T14:30:00.42+01:00");
        CHECK(e.sub_sec_time == "42");
        CHECK(e.offset_time == "+01:00");

        // Trailing NULs in undefined/raw entry trimmed.
        const auto b_nuls = build_block(le, {}, {},
                                        {tiffw::ascii(0x9003, "2024:05:06 14:30:00"),
                                         tiffw::undefined(0x9291, {'8', '5', 0, 0}),
                                         tiffw::ascii(0x9011, "+01:00")},
                                        {});
        CHECK(broimage::read_exif(b_nuls.data(), b_nuls.size(), e));
        CHECK(e.date_taken == "2024-05-06T14:30:00.85+01:00");
        CHECK(e.sub_sec_time == "85");
        CHECK(e.offset_time == "+01:00");

        // Non-digit SubSecTime is ignored.
        for (const char* bad : {"bad", "25a", " 25", "+25", "-25", "1.5", "   ", ""}) {
            const auto b_bad = build_block(le, {}, {},
                                           {tiffw::ascii(0x9003, "2024:05:06 14:30:00"),
                                            tiffw::ascii(0x9291, bad),
                                            tiffw::ascii(0x9011, "+02:00")},
                                           {});
            CHECK(broimage::read_exif(b_bad.data(), b_bad.size(), e));
            CHECK(e.date_taken == "2024-05-06T14:30:00+02:00");
            CHECK(e.sub_sec_time.empty());
            CHECK(e.offset_time == "+02:00");
        }
    }
}

void test_file() {
    const auto tif = build_block(true, camera(), {});
    const std::string path = "broimage_test_exif.tif";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (!f) return;
    std::fwrite(tif.data(), 1, tif.size(), f);
    std::fclose(f);
    broimage::ExifInfo e;
    CHECK(broimage::read_exif_file(path, e));
    check_full(e, "TIFF file");
    CHECK(broimage::read_exif_orientation_file(path) == broimage::ExifOrientation::Rotate90CW);
    std::remove(path.c_str());
}

} // namespace

int main() {
    test_containers();
    test_exposure_text();
    test_absent_and_missing();
    test_subsec_time();
    test_file();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test_exif: all passed\n");
    return 0;
}
