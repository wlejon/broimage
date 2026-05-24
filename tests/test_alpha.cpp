#include "broimage/alpha.h"

#include <cstdint>
#include <cstdio>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

int main() {
    // Premultiply: a=0 zeroes RGB, a=255 leaves RGB unchanged.
    const uint8_t in[3 * 4] = {
        200, 100,  50, 255, // opaque -> unchanged
        200, 100,  50,   0, // transparent -> zeroed
        200, 100,  50, 128, // half -> ~(channel*128+127)/255
    };
    uint8_t out[3 * 4];
    broimage::premultiply_alpha_rgba8(in, out, 3);
    CHECK(out[0] == 200 && out[1] == 100 && out[2] == 50 && out[3] == 255);
    CHECK(out[4] ==   0 && out[5] ==   0 && out[6] ==  0 && out[7] ==   0);
    CHECK(out[8] == static_cast<uint8_t>((200 * 128 + 127) / 255));
    CHECK(out[11] == 128);

    // Round trip: premultiply then unpremultiply restores opaque + half (within
    // 1-bit rounding noise) and produces zero for fully transparent (canonical).
    uint8_t back[3 * 4];
    broimage::unpremultiply_alpha_rgba8(out, back, 3);
    CHECK(back[0] == 200 && back[3] == 255);
    // Transparent pixel: RGB is zeroed (canonical for un-premul of a=0).
    CHECK(back[4] == 0 && back[5] == 0 && back[6] == 0 && back[7] == 0);
    // Half-alpha: channel reconstructed within 1.
    const int orig = 200;
    const int prem = (orig * 128 + 127) / 255;
    const int rec  = (prem * 255 + 64) / 128;
    CHECK(back[8] == rec);
    CHECK(back[11] == 128);

    // Alpha-correct resize: 2x2 RGBA with 3 opaque red pixels and one
    // transparent pixel with garbage RGB. Naive resize would average the
    // garbage into the result; alpha-aware resize must not.
    //
    //   (200, 0, 0, 255)   (200, 0, 0, 255)
    //   (200, 0, 0, 255)   ( 0, 255, 0,  0)   <- garbage green, transparent
    //
    // Downscaled to 1x1, the center pixel sees the three opaque reds plus the
    // transparent corner; alpha-correct: rgb stays ~red, alpha ~ 3*255/4 = ~191.
    // Naive bilinear: rgb is averaged with the green, producing a brownish tint.
    const uint8_t img2[2 * 2 * 4] = {
        200,   0,   0, 255,   200,   0,   0, 255,
        200,   0,   0, 255,     0, 255,   0,   0,
    };
    uint8_t one[4];
    broimage::resize_rgba8_alpha(img2, 2, 2, one, 1, 1, broimage::Filter::Bilinear);
    // Expectation: dominant red, very little green leak, alpha around 191.
    CHECK(one[0] > 140);    // red stays high
    CHECK(one[1] < 30);     // green leak is small (would be ~64 with naive)
    CHECK(one[2] == 0);
    CHECK(one[3] > 180 && one[3] < 210);

    // Letterbox alpha: square output, transparent source square fits exactly.
    // 2x4 (tall) -> 4x4 dst, pad = transparent black.
    std::vector<uint8_t> tall(2 * 4 * 4, 0);
    for (int i = 0; i < 8; ++i) {
        tall[i * 4 + 0] = 128; tall[i * 4 + 3] = 255;
    }
    std::vector<uint8_t> lb(4 * 4 * 4, 0);
    int ox = 0, oy = 0, ow = 0, oh = 0;
    broimage::letterbox_rgba8_alpha(tall.data(), 2, 4, lb.data(), 4, 4,
                                    0, 0, 0, 0,
                                    broimage::Filter::Bilinear,
                                    &ox, &oy, &ow, &oh);
    // scale s = min(4/2, 4/4) = 1, new_w = 2, new_h = 4, off_x = 1, off_y = 0.
    CHECK(ox == 1 && oy == 0 && ow == 2 && oh == 4);
    // Pad columns (x=0 and x=3): alpha=0 with our chosen pad color.
    for (int y = 0; y < 4; ++y) {
        CHECK(lb[(y * 4 + 0) * 4 + 3] == 0);
        CHECK(lb[(y * 4 + 3) * 4 + 3] == 0);
    }
    // Content columns are opaque-ish.
    for (int y = 0; y < 4; ++y) {
        CHECK(lb[(y * 4 + 1) * 4 + 3] > 200);
        CHECK(lb[(y * 4 + 2) * 4 + 3] > 200);
    }

    return g_failed == 0 ? 0 : 1;
}
