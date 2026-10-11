// sharpness_u8: exact values on patterns worked out by hand (a flat field, a
// step edge, a one-pixel checkerboard, which the Laplacian sees and Sobel
// does not), a blurred copy measuring lower, the region matching a cropped
// copy, grey RGB(A) matching luma, strides, and the same answer for every
// thread count.

#include "broimage/sharpness.h"

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

namespace {

bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol * std::fmax(1.0, std::fabs(b)); }

bool same(const broimage::SharpnessResult& a, const broimage::SharpnessResult& b) {
    return a.laplacian_variance == b.laplacian_variance && a.tenengrad == b.tenengrad &&
           a.luma_mean == b.luma_mean && a.luma_std == b.luma_std && a.pixels == b.pixels;
}

// Deterministic texture, luma: smooth waves plus pseudo-random grain.
std::vector<uint8_t> texture(int w, int h) {
    std::vector<uint8_t> img(static_cast<size_t>(w) * h);
    uint32_t s = 12345;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            s = s * 1664525u + 1013904223u;
            const double v = 128 + 50 * std::sin(x * 0.31) * std::cos(y * 0.23) + ((s >> 24) & 63) - 32;
            img[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(std::fmin(std::fmax(v, 0.0), 255.0));
        }
    return img;
}

// 3 x 3 box blur (edges copied).
std::vector<uint8_t> blur(const std::vector<uint8_t>& src, int w, int h) {
    std::vector<uint8_t> out(src);
    for (int y = 1; y + 1 < h; ++y)
        for (int x = 1; x + 1 < w; ++x) {
            int sum = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) sum += src[static_cast<size_t>(y + dy) * w + x + dx];
            out[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>((sum + 4) / 9);
        }
    return out;
}

void testFlat() {
    std::vector<uint8_t> img(64 * 48, 90);
    const auto r = broimage::sharpness_u8(img.data(), 64, 48, 1);
    CHECK(r.laplacian_variance == 0 && r.tenengrad == 0 && r.luma_std == 0);
    CHECK(r.luma_mean == 90);
    CHECK(r.pixels == 62 * 46);
}

void testStepEdge() {
    // Columns < 20 are 0, the rest 200. The two columns either side of the
    // edge carry Laplacian -200 / +200 and Sobel gx 800; everything else 0.
    const int W = 50, H = 30;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) img[static_cast<size_t>(y) * W + x] = x < 20 ? 0 : 200;
    const auto r = broimage::sharpness_u8(img.data(), W, H, 1);
    const double n = double(W - 2) * (H - 2);
    CHECK(near(r.laplacian_variance, 2.0 * (H - 2) * 40000.0 / n));
    CHECK(near(r.tenengrad, 2.0 * (H - 2) * 640000.0 / n));
}

void testCheckerboard() {
    // Laplacian +-1020 everywhere; Sobel's two columns either side are equal,
    // so no gradient at all.
    const int W = 32, H = 32;
    std::vector<uint8_t> img(static_cast<size_t>(W) * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) img[static_cast<size_t>(y) * W + x] = ((x + y) & 1) ? 255 : 0;
    const auto r = broimage::sharpness_u8(img.data(), W, H, 1);
    const double n = double(W - 2) * (H - 2);
    // 30 x 30 interior: as many +1020 as -1020, mean 0.
    CHECK(near(r.laplacian_variance, 1020.0 * 1020.0));
    CHECK(r.tenengrad == 0);
    CHECK(near(r.luma_mean, 127.5));
    CHECK(r.pixels == static_cast<int64_t>(n));
}

void testBlurLowers() {
    const int W = 200, H = 150;
    const auto img = texture(W, H);
    const auto soft = blur(img, W, H);
    const auto a = broimage::sharpness_u8(img.data(), W, H, 1);
    const auto b = broimage::sharpness_u8(soft.data(), W, H, 1);
    CHECK(a.laplacian_variance > 4 * b.laplacian_variance);
    CHECK(a.tenengrad > 1.5 * b.tenengrad);
}

void testRegionAndChannels() {
    const int W = 120, H = 90;
    const auto img = texture(W, H);
    // A region equals a cropped copy.
    const int rx = 17, ry = 9, rw = 61, rh = 44;
    std::vector<uint8_t> crop(static_cast<size_t>(rw) * rh);
    for (int y = 0; y < rh; ++y)
        for (int x = 0; x < rw; ++x) crop[static_cast<size_t>(y) * rw + x] = img[static_cast<size_t>(y + ry) * W + x + rx];
    const auto a = broimage::sharpness_u8(img.data(), W, H, 1, 0, rx, ry, rw, rh);
    const auto b = broimage::sharpness_u8(crop.data(), rw, rh, 1);
    CHECK(same(a, b));
    // A region past the edge is clamped; 0 x 0 means to the edge.
    CHECK(same(broimage::sharpness_u8(img.data(), W, H, 1, 0, 100, 80, 500, 500),
               broimage::sharpness_u8(img.data(), W, H, 1, 0, 100, 80)));
    // Under 3 x 3: nothing counted.
    CHECK(broimage::sharpness_u8(img.data(), W, H, 1, 0, 5, 5, 2, 10).pixels == 0);
    // Grey RGB / RGBA (and luma + alpha) in a padded stride measure as the luma.
    const int stride = W * 4 + 12;
    std::vector<uint8_t> rgba(static_cast<size_t>(stride) * H, 7), rgb(static_cast<size_t>(W) * H * 3),
        ga(static_cast<size_t>(W) * H * 2);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint8_t v = img[static_cast<size_t>(y) * W + x];
            uint8_t* p = rgba.data() + static_cast<size_t>(y) * stride + x * 4;
            p[0] = p[1] = p[2] = v;
            p[3] = 255;
            uint8_t* q = rgb.data() + (static_cast<size_t>(y) * W + x) * 3;
            q[0] = q[1] = q[2] = v;
            ga[(static_cast<size_t>(y) * W + x) * 2] = v;
            ga[(static_cast<size_t>(y) * W + x) * 2 + 1] = 3;
        }
    const auto g = broimage::sharpness_u8(img.data(), W, H, 1);
    CHECK(same(g, broimage::sharpness_u8(rgba.data(), W, H, 4, stride)));
    CHECK(same(g, broimage::sharpness_u8(rgb.data(), W, H, 3)));
    CHECK(same(g, broimage::sharpness_u8(ga.data(), W, H, 2)));
    // Colour weights: pure green reads brighter than pure blue.
    std::vector<uint8_t> green(16 * 16 * 3, 0), blue(16 * 16 * 3, 0);
    for (int i = 0; i < 16 * 16; ++i) green[i * 3 + 1] = 200, blue[i * 3 + 2] = 200;
    CHECK(broimage::sharpness_u8(green.data(), 16, 16, 3).luma_mean == (150 * 200 + 128) >> 8);
    CHECK(broimage::sharpness_u8(blue.data(), 16, 16, 3).luma_mean == (29 * 200 + 128) >> 8);
}

void testThreadsAgree() {
    const int W = 1500, H = 1000;   // over a megapixel: the default splits it
    const auto img = texture(W, H);
    const auto one = broimage::sharpness_u8(img.data(), W, H, 1, 0, 0, 0, 0, 0, 1);
    for (int t : {0, 2, 3, 7, 8, 64})
        CHECK(same(one, broimage::sharpness_u8(img.data(), W, H, 1, 0, 0, 0, 0, 0, t)));
    // A region with fewer interior rows than threads.
    CHECK(same(broimage::sharpness_u8(img.data(), W, H, 1, 0, 10, 10, 900, 5, 1),
               broimage::sharpness_u8(img.data(), W, H, 1, 0, 10, 10, 900, 5, 8)));
}

} // namespace

int main() {
    testFlat();
    testStepEdge();
    testCheckerboard();
    testBlurLowers();
    testRegionAndChannels();
    testThreadsAgree();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("sharpness: all passed\n");
    return 0;
}
