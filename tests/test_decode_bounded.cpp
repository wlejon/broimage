// decode_memory_bounded / gif_frame_count: limits checked before decoding,
// every frame of an animated GIF, a still image as one frame.
#include "broimage/decode.h"
#include "broimage/encode.h"

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

// A 16 x 12 GIF89a, three frames of 70 ms (red, green, blue), looping.
const uint8_t kAnimGif[] = {
    0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x10, 0x00, 0x0c, 0x00, 0xf0, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x21, 0xff, 0x0b, 0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30, 0x03, 0x01, 0x00,
    0x00, 0x00, 0x21, 0xf9, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x0c,
    0x00, 0x00, 0x02, 0x0c, 0x84, 0x8f, 0xa9, 0xcb, 0xed, 0x0f, 0xa3, 0x9c, 0xb4, 0xda, 0x6b, 0x0a, 0x00, 0x21,
    0xf9, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x80, 0x00,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x02, 0x0c, 0x84, 0x8f, 0xa9, 0xcb, 0xed, 0x0f, 0xa3, 0x9c, 0xb4, 0xda, 0x6b,
    0x0a, 0x00, 0x21, 0xf9, 0x04, 0x00, 0x07, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x0c,
    0x00, 0x80, 0x00, 0x00, 0xff, 0x00, 0x00, 0x00, 0x02, 0x0c, 0x84, 0x8f, 0xa9, 0xcb, 0xed, 0x0f, 0xa3, 0x9c,
    0xb4, 0xda, 0x6b, 0x0a, 0x00, 0x3b};

} // namespace

int main() {
    // The block walk counts the frames without decoding.
    CHECK(broimage::gif_frame_count(kAnimGif, sizeof kAnimGif) == 3);
    CHECK(broimage::gif_frame_count(kAnimGif, 10) == 0);
    const uint8_t notGif[] = {0x89, 'P', 'N', 'G'};
    CHECK(broimage::gif_frame_count(notGif, sizeof notGif) == 0);

    // Every frame, with its delay, each a solid colour.
    broimage::Animation a;
    std::string err;
    CHECK(broimage::decode_memory_bounded(kAnimGif, sizeof kAnimGif, {}, a, &err));
    CHECK(a.width == 16 && a.height == 12);
    CHECK(a.frames.size() == 3);
    const uint8_t want[3][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
    for (size_t i = 0; i < a.frames.size() && i < 3; ++i) {
        CHECK(a.frames[i].rgba.size() == size_t(16 * 12 * 4));
        CHECK(a.frames[i].delay_ms == 70);
        bool solid = true;
        for (size_t p = 0; p + 3 < a.frames[i].rgba.size(); p += 4)
            solid = solid && a.frames[i].rgba[p] == want[i][0] && a.frames[i].rgba[p + 1] == want[i][1] &&
                    a.frames[i].rgba[p + 2] == want[i][2] && a.frames[i].rgba[p + 3] == 255;
        CHECK(solid);
    }

    // Limits are checked against the headers: refused, and nothing produced.
    broimage::DecodeLimits tight;
    tight.max_bytes = 16 * 12 * 4 * 2;  // two frames' worth, three needed
    CHECK(!broimage::decode_memory_bounded(kAnimGif, sizeof kAnimGif, tight, a, &err));
    CHECK(a.frames.empty() && !err.empty());
    tight = {};
    tight.max_width = 15;
    CHECK(!broimage::decode_memory_bounded(kAnimGif, sizeof kAnimGif, tight, a, &err));
    tight = {};
    tight.max_height = 11;
    CHECK(!broimage::decode_memory_bounded(kAnimGif, sizeof kAnimGif, tight, a, &err));
    tight = {};
    tight.max_bytes = 16 * 12 * 4 * 3;  // exactly enough
    CHECK(broimage::decode_memory_bounded(kAnimGif, sizeof kAnimGif, tight, a, &err));
    CHECK(a.frames.size() == 3);

    // A still PNG: one frame, the pixels exactly.
    const int W = 5, H = 4;
    std::vector<uint8_t> src(size_t(W * H * 4));
    for (size_t i = 0; i < src.size(); ++i) src[i] = uint8_t(i * 13 + 7);
    std::vector<uint8_t> png;
    CHECK(broimage::encode_png_memory(png, src.data(), W, H, 4));
    CHECK(broimage::decode_memory_bounded(png.data(), png.size(), {}, a, &err));
    CHECK(a.width == W && a.height == H && a.frames.size() == 1);
    CHECK(!a.frames.empty() && a.frames[0].rgba == src);
    tight = {};
    tight.max_bytes = size_t(W * H * 4) - 1;
    CHECK(!broimage::decode_memory_bounded(png.data(), png.size(), tight, a, &err));

    // Garbage and empty input.
    const uint8_t junk[] = {1, 2, 3, 4, 5, 6, 7, 8};
    CHECK(!broimage::decode_memory_bounded(junk, sizeof junk, {}, a, &err));
    CHECK(!broimage::decode_memory_bounded(nullptr, 0, {}, a, &err));
    // A GIF cut off in its first frame.
    CHECK(!broimage::decode_memory_bounded(kAnimGif, 40, {}, a, &err));

    if (g_failed == 0) std::printf("broimage_test_decode_bounded: OK\n");
    return g_failed == 0 ? 0 : 1;
}
