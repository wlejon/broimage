// decode_memory_bounded / gif_frame_count: limits checked before decoding,
// every frame of an animated GIF, a still image as one frame.
#include "broimage/decode.h"
#include "broimage/encode.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
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

// A baseline TIFF: w x h, `spp` 8-bit samples per pixel from `px`, in
// strips of `rows_per_strip`, little or big endian, `extra` as ExtraSamples.
std::vector<uint8_t> make_tiff(int w, int h, int spp, const std::vector<uint8_t>& px, int rows_per_strip, bool le,
                               int extra, int photometric) {
    std::vector<uint8_t> t;
    auto put16 = [&](uint32_t v) {
        if (le) t.insert(t.end(), {uint8_t(v), uint8_t(v >> 8)});
        else t.insert(t.end(), {uint8_t(v >> 8), uint8_t(v)});
    };
    auto put32 = [&](uint32_t v) {
        if (le) { put16(v & 0xFFFF); put16(v >> 16); }
        else { put16(v >> 16); put16(v & 0xFFFF); }
    };
    t.insert(t.end(), le ? std::initializer_list<uint8_t>{'I', 'I'} : std::initializer_list<uint8_t>{'M', 'M'});
    put16(42);
    put32(8);
    const int strips = (h + rows_per_strip - 1) / rows_per_strip;
    struct E { uint16_t tag, type; uint32_t count, value; };
    // Strip data goes after the IFD and the two strip arrays.
    const uint32_t entries = extra ? 10 : 9;
    const uint32_t ifd_end = 8 + 2 + entries * 12 + 4;
    const uint32_t offs_at = ifd_end, counts_at = offs_at + 4 * strips, data_at = counts_at + 4 * strips;
    std::vector<E> e = {{256, 4, 1, uint32_t(w)}, {257, 4, 1, uint32_t(h)}, {258, 3, 1, 8}, {259, 3, 1, 1},
                        {262, 3, 1, uint32_t(photometric)},
                        {273, 4, uint32_t(strips), strips == 1 ? data_at : offs_at},
                        {277, 3, 1, uint32_t(spp)}, {278, 4, 1, uint32_t(rows_per_strip)},
                        {279, 4, uint32_t(strips), strips == 1 ? uint32_t(w * h * spp) : counts_at}};
    if (extra) e.push_back({338, 3, 1, uint32_t(extra)});
    put16(entries);
    for (const E& x : e) {
        put16(x.tag);
        put16(x.type);
        put32(x.count);
        if (x.type == 3 && x.count == 1) { put16(x.value); put16(0); }
        else put32(x.value);
    }
    put32(0);
    uint32_t at = data_at;
    for (int s = 0; s < strips; ++s) {
        put32(at);
        at += uint32_t(std::min(rows_per_strip, h - s * rows_per_strip) * w * spp);
    }
    for (int s = 0; s < strips; ++s) put32(uint32_t(std::min(rows_per_strip, h - s * rows_per_strip) * w * spp));
    t.insert(t.end(), px.begin(), px.end());
    return t;
}

} // namespace

int main() {
    // Baseline TIFF: RGBA with unassociated alpha, three strips, both byte orders.
    {
        const int W = 6, H = 5;
        std::vector<uint8_t> px(size_t(W * H * 4));
        for (size_t i = 0; i < px.size(); ++i) px[i] = uint8_t(i * 7 + 3);
        for (bool le : {true, false}) {
            const std::vector<uint8_t> tiff = make_tiff(W, H, 4, px, 2, le, 2, 2);
            broimage::Animation a;
            std::string err;
            CHECK(broimage::decode_memory_bounded(tiff.data(), tiff.size(), {}, a, &err));
            CHECK(a.width == W && a.height == H && a.frames.size() == 1);
            CHECK(!a.frames.empty() && a.frames[0].rgba == px);
            broimage::DecodeLimits tight;
            tight.max_bytes = size_t(W * H * 4) - 1;
            CHECK(!broimage::decode_memory_bounded(tiff.data(), tiff.size(), tight, a, &err));
            // Truncated strips are refused, not read past.
            CHECK(!broimage::decode_memory_bounded(tiff.data(), tiff.size() - 3, {}, a, &err));
        }
        // RGB, one strip, and gray: expanded to opaque RGBA.
        std::vector<uint8_t> rgb(size_t(W * H * 3));
        for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t(i * 5);
        std::vector<uint8_t> tiff = make_tiff(W, H, 3, rgb, H, true, 0, 2);
        broimage::Animation a;
        CHECK(broimage::decode_memory_bounded(tiff.data(), tiff.size(), {}, a));
        bool ok = a.frames.size() == 1;
        for (size_t i = 0; ok && i < size_t(W * H); ++i)
            ok = a.frames[0].rgba[i * 4] == rgb[i * 3] && a.frames[0].rgba[i * 4 + 2] == rgb[i * 3 + 2] &&
                 a.frames[0].rgba[i * 4 + 3] == 255;
        CHECK(ok);
        std::vector<uint8_t> gray(size_t(W * H));
        for (size_t i = 0; i < gray.size(); ++i) gray[i] = uint8_t(i * 9);
        tiff = make_tiff(W, H, 1, gray, H, false, 0, 1);
        CHECK(broimage::decode_memory_bounded(tiff.data(), tiff.size(), {}, a));
        CHECK(a.frames.size() == 1 && a.frames[0].rgba[4 * 3] == gray[3] && a.frames[0].rgba[4 * 3 + 1] == gray[3]);
        // Premultiplied (associated) alpha is straightened.
        std::vector<uint8_t> pm = {100, 50, 0, 128};
        tiff = make_tiff(1, 1, 4, pm, 1, true, 1, 2);
        CHECK(broimage::decode_memory_bounded(tiff.data(), tiff.size(), {}, a));
        CHECK(a.frames.size() == 1 && a.frames[0].rgba[0] == 199 && a.frames[0].rgba[1] == 100 &&
              a.frames[0].rgba[3] == 128);
    }

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
