#include "broimage/buffer.h"
#include "broimage/decode.h"
#include "broimage/encode.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

static std::string tmp_path(const char* name) {
    const char* tmp = std::getenv("TMP");
    if (!tmp) tmp = std::getenv("TEMP");
    if (!tmp) tmp = "/tmp";
    return std::string(tmp) + "/broimage_test_" + name;
}

int main() {
    // Encode a synthetic 4x3 RGBA gradient to PNG, decode it, verify pixels match.
    const int W = 4, H = 3;
    std::vector<uint8_t> src(W * H * 4);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            uint8_t* p = src.data() + (y * W + x) * 4;
            p[0] = static_cast<uint8_t>(x * 64);
            p[1] = static_cast<uint8_t>(y * 80);
            p[2] = static_cast<uint8_t>((x + y) * 32);
            p[3] = 255;
        }
    }

    const std::string path = tmp_path("rgba.png");
    CHECK(broimage::encode_png_file(path, src.data(), W, H, 4));

    broimage::Image img;
    std::string err;
    CHECK(broimage::decode_file(path, img, &err));
    CHECK(img.width == W);
    CHECK(img.height == H);
    CHECK(img.channels == 4);
    CHECK(img.pixels.size() == src.size());
    for (size_t i = 0; i < src.size(); ++i) CHECK(img.pixels[i] == src[i]);

    // encode_png_memory + decode_memory round trip.
    std::vector<uint8_t> mem;
    CHECK(broimage::encode_png_memory(mem, src.data(), W, H, 4));
    CHECK(!mem.empty());

    broimage::Image img2;
    CHECK(broimage::decode_memory(mem.data(), mem.size(), img2));
    CHECK(img2.width == W && img2.height == H && img2.channels == 4);
    for (size_t i = 0; i < src.size(); ++i) CHECK(img2.pixels[i] == src[i]);

    // Decode failure -> 1x1 white fallback.
    broimage::Image bad;
    std::string bad_err;
    bool ok = broimage::decode_file("does-not-exist-broimage.png", bad, &bad_err);
    CHECK(!ok);
    CHECK(bad.width == 1 && bad.height == 1);
    CHECK(bad.pixels.size() == 4);
    CHECK(bad.pixels[0] == 255 && bad.pixels[3] == 255);
    CHECK(!bad_err.empty());

    // JPEG round trip (lossy — just check decode succeeds, dims match, file exists).
    const std::string jpath = tmp_path("rgb.jpg");
    std::vector<uint8_t> rgb(W * H * 3);
    for (int i = 0; i < W * H; ++i) {
        rgb[i * 3 + 0] = src[i * 4 + 0];
        rgb[i * 3 + 1] = src[i * 4 + 1];
        rgb[i * 3 + 2] = src[i * 4 + 2];
    }
    CHECK(broimage::encode_jpeg_file(jpath, rgb.data(), W, H, 3, 90));
    broimage::Image jimg;
    CHECK(broimage::decode_file(jpath, jimg));
    CHECK(jimg.width == W && jimg.height == H);

    // ----- EXIF orientation --------------------------------------------------
    // Build a tiny synthetic JPEG header containing only SOI + APP1 with one
    // EXIF orientation tag. We don't need entropy-coded image data — the
    // parser stops at SOS (which we don't emit), and bails earlier when the
    // tag is found.
    auto make_exif_jpeg = [](uint16_t orient_value) {
        std::vector<uint8_t> buf;
        // SOI.
        buf.push_back(0xFF); buf.push_back(0xD8);
        // APP1 marker.
        buf.push_back(0xFF); buf.push_back(0xE1);
        // Build payload: "Exif\0\0" + TIFF header (little-endian) + IFD0 with 1 entry.
        std::vector<uint8_t> p;
        const char id[] = { 'E','x','i','f',0,0 };
        p.insert(p.end(), id, id + 6);
        // TIFF little-endian header.
        p.push_back('I'); p.push_back('I');
        p.push_back(0x2A); p.push_back(0x00);   // magic 0x2A
        // Offset to IFD0 = 8 (right after TIFF header).
        p.push_back(0x08); p.push_back(0x00); p.push_back(0x00); p.push_back(0x00);
        // Number of IFD0 entries = 1.
        p.push_back(0x01); p.push_back(0x00);
        // Entry: tag 0x0112 (orientation), type 3 (SHORT), count 1, value = orient.
        p.push_back(0x12); p.push_back(0x01); // tag
        p.push_back(0x03); p.push_back(0x00); // type
        p.push_back(0x01); p.push_back(0x00); p.push_back(0x00); p.push_back(0x00); // count
        p.push_back(static_cast<uint8_t>(orient_value & 0xFF));
        p.push_back(static_cast<uint8_t>((orient_value >> 8) & 0xFF));
        p.push_back(0x00); p.push_back(0x00);
        // Next IFD offset = 0.
        p.push_back(0x00); p.push_back(0x00); p.push_back(0x00); p.push_back(0x00);
        // Segment length includes the length bytes themselves (2) + payload.
        const uint16_t seg_len = static_cast<uint16_t>(2 + p.size());
        buf.push_back(static_cast<uint8_t>((seg_len >> 8) & 0xFF));
        buf.push_back(static_cast<uint8_t>(seg_len & 0xFF));
        buf.insert(buf.end(), p.begin(), p.end());
        return buf;
    };

    for (int o = 1; o <= 8; ++o) {
        const auto jpg = make_exif_jpeg(static_cast<uint16_t>(o));
        const auto read = broimage::read_exif_orientation(jpg.data(), jpg.size());
        CHECK(static_cast<int>(read) == o);
    }
    // No-EXIF JPEG: parser returns Normal.
    {
        std::vector<uint8_t> bare = { 0xFF, 0xD8, 0xFF, 0xD9 };
        CHECK(broimage::read_exif_orientation(bare.data(), bare.size()) ==
              broimage::ExifOrientation::Normal);
    }
    // Non-JPEG bytes: Normal.
    {
        std::vector<uint8_t> png_sig = { 0x89, 'P', 'N', 'G' };
        CHECK(broimage::read_exif_orientation(png_sig.data(), png_sig.size()) ==
              broimage::ExifOrientation::Normal);
    }

    // apply_exif_orientation: synthesize a 2x3 RGBA image with distinct pixel
    // identifiers, then check each orientation matches the hand-computed map.
    auto make_img = []() {
        broimage::Image im;
        im.width = 2; im.height = 3; im.channels = 4;
        im.pixels.resize(2 * 3 * 4);
        // pixel id encoded in R; G = x, B = y, A = 255.
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 2; ++x) {
                uint8_t* p = im.pixels.data() + (y * 2 + x) * 4;
                p[0] = static_cast<uint8_t>(y * 2 + x);
                p[1] = static_cast<uint8_t>(x);
                p[2] = static_cast<uint8_t>(y);
                p[3] = 255;
            }
        }
        return im;
    };
    auto id_at = [](const broimage::Image& im, int x, int y) {
        return im.pixels[(y * im.width + x) * 4 + 0];
    };

    // FlipH: (x, y) -> id(W-1-x, y).
    {
        auto im = make_img();
        broimage::apply_exif_orientation(im, broimage::ExifOrientation::FlipH);
        CHECK(im.width == 2 && im.height == 3);
        for (int y = 0; y < 3; ++y) for (int x = 0; x < 2; ++x) {
            CHECK(id_at(im, x, y) == y * 2 + (1 - x));
        }
    }
    // Rotate90CW: dst (x, y) <- src (y, H-1-x); dims swap to (3, 2).
    {
        auto im = make_img();
        broimage::apply_exif_orientation(im, broimage::ExifOrientation::Rotate90CW);
        CHECK(im.width == 3 && im.height == 2);
        for (int y = 0; y < 2; ++y) for (int x = 0; x < 3; ++x) {
            const int sx = y;
            const int sy = 3 - 1 - x;
            CHECK(id_at(im, x, y) == sy * 2 + sx);
        }
    }
    // Rotate90CCW: dst (x, y) <- src (W-1-y, x); dims swap to (3, 2).
    {
        auto im = make_img();
        broimage::apply_exif_orientation(im, broimage::ExifOrientation::Rotate90CCW);
        CHECK(im.width == 3 && im.height == 2);
        for (int y = 0; y < 2; ++y) for (int x = 0; x < 3; ++x) {
            const int sx = 2 - 1 - y;
            const int sy = x;
            CHECK(id_at(im, x, y) == sy * 2 + sx);
        }
    }

    return g_failed == 0 ? 0 : 1;
}
