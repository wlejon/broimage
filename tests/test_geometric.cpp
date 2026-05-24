#include "broimage/geometric.h"

#include <cmath>
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

static bool nearf(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol;
}

int main() {
    // 2x2 single-channel float upsampled to 4x4 bilinear: corners should
    // equal the source corner samples (within the center-pixel rule).
    const float src2[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
    float dst4[16];
    broimage::resize_hwc_f32(src2, 2, 2, 1, dst4, 4, 4, broimage::Filter::Bilinear);
    // Corner samples are clamped, so dst4[0] ≈ src2[0] = 0, dst4[3] ≈ src2[1] = 1.
    CHECK(nearf(dst4[0], 0.0f, 0.05f));
    CHECK(nearf(dst4[3], 1.0f, 0.05f));
    CHECK(nearf(dst4[12], 1.0f, 0.05f));
    CHECK(nearf(dst4[15], 0.0f, 0.05f));

    // Nearest-neighbor identity: same dims is a copy.
    float dst2[4];
    broimage::resize_hwc_f32(src2, 2, 2, 1, dst2, 2, 2, broimage::Filter::Nearest);
    for (int i = 0; i < 4; ++i) CHECK(nearf(dst2[i], src2[i]));

    // Bicubic identity at same dims (should be close to identity, not exact).
    float dst2b[4];
    broimage::resize_hwc_f32(src2, 2, 2, 1, dst2b, 2, 2, broimage::Filter::Bicubic);
    for (int i = 0; i < 4; ++i) CHECK(nearf(dst2b[i], src2[i], 1e-3f));

    // u8 path: downscale a 4x4 solid color, every pixel preserved.
    std::vector<uint8_t> red(4 * 4 * 4, 0);
    for (int i = 0; i < 16; ++i) { red[i*4+0] = 200; red[i*4+3] = 255; }
    std::vector<uint8_t> red_small(2 * 2 * 4);
    broimage::resize_hwc_u8(red.data(), 4, 4, 4, red_small.data(), 2, 2);
    for (int i = 0; i < 4; ++i) {
        CHECK(red_small[i*4+0] == 200);
        CHECK(red_small[i*4+3] == 255);
    }

    // Letterbox: tall source into a square dst leaves horizontal bars.
    // src 4x8 (tall) -> dst 4x4. Scale = 4/8 = 0.5; new_w = 2, new_h = 4.
    // Content rect: x=1, y=0, w=2, h=4. Bar columns: x in {0, 3}.
    std::vector<uint8_t> tall(4 * 8 * 4, 0);
    for (int i = 0; i < 32; ++i) { tall[i*4+0] = 100; tall[i*4+1] = 100; tall[i*4+2] = 100; tall[i*4+3] = 255; }
    std::vector<uint8_t> lb(4 * 4 * 4, 0);
    int ox = 0, oy = 0, ow = 0, oh = 0;
    broimage::letterbox_hwc_u8(tall.data(), 4, 8, 4, lb.data(), 4, 4,
                               7, 7, 7, 255, broimage::Filter::Bilinear,
                               &ox, &oy, &ow, &oh);
    CHECK(ox == 1 && oy == 0 && ow == 2 && oh == 4);
    // Left bar (x=0) and right bar (x=3) every row: pad color.
    for (int y = 0; y < 4; ++y) {
        const uint8_t* l = lb.data() + (y * 4 + 0) * 4;
        const uint8_t* r = lb.data() + (y * 4 + 3) * 4;
        CHECK(l[0] == 7 && l[1] == 7 && l[2] == 7);
        CHECK(r[0] == 7 && r[1] == 7 && r[2] == 7);
    }

    // Center crop.
    std::vector<uint8_t> grid(4 * 4 * 3);
    for (int i = 0; i < 16; ++i) {
        grid[i*3+0] = static_cast<uint8_t>(i);
        grid[i*3+1] = static_cast<uint8_t>(i * 2);
        grid[i*3+2] = static_cast<uint8_t>(i * 3);
    }
    std::vector<uint8_t> cc(2 * 2 * 3);
    broimage::center_crop_hwc_u8(grid.data(), 4, 4, 3, cc.data(), 2, 2);
    // crop origin (1, 1): expected source indices 5, 6, 9, 10.
    CHECK(cc[0] == 5  && cc[3] == 6);
    CHECK(cc[6] == 9  && cc[9] == 10);

    // Flip horizontal: 2x1, RGB.
    const uint8_t row[6] = { 1, 2, 3,  4, 5, 6 };
    uint8_t flipped[6];
    broimage::flip_horizontal_hwc_u8(row, flipped, 2, 1, 3);
    CHECK(flipped[0] == 4 && flipped[3] == 1);

    // Flip vertical: 1x2, RGB.
    const uint8_t col[6] = { 1, 2, 3,  4, 5, 6 };
    uint8_t fv[6];
    broimage::flip_vertical_hwc_u8(col, fv, 1, 2, 3);
    CHECK(fv[0] == 4 && fv[3] == 1);

    // Rotate 90 CCW.
    // 2x3 (w=2,h=3) gray (channels=1):
    //   0 1
    //   2 3
    //   4 5
    // After 90 CCW (turns=1), dst dims (h=2,w=3 swapped) -> (3,2) i.e. dw=3, dh=2:
    //   1 3 5
    //   0 2 4
    const uint8_t small[6] = { 0, 1, 2, 3, 4, 5 };
    uint8_t rot[6];
    broimage::rotate_90_hwc_u8(small, 2, 3, 1, rot, 1);
    CHECK(rot[0] == 1 && rot[1] == 3 && rot[2] == 5);
    CHECK(rot[3] == 0 && rot[4] == 2 && rot[5] == 4);

    return g_failed == 0 ? 0 : 1;
}
