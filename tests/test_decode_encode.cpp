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

    return g_failed == 0 ? 0 : 1;
}
