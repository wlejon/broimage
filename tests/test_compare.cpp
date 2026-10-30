// compare_u8: equal images (zero difference, infinite PSNR, SSIM 1), a known
// brightening (its signed mean, rms, PSNR, the share changed under and over
// the threshold), a change confined to one cell of a grid, a blur lowering
// the SSIM while a uniform shift keeps it high, cell bounds covering the
// image, and strides.

#include "broimage/compare.h"

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

// RGBA texture: smooth waves plus grain, alpha 255.
std::vector<uint8_t> texture(int w, int h) {
    std::vector<uint8_t> img(static_cast<size_t>(w) * h * 4);
    uint32_t s = 777;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            s = s * 1664525u + 1013904223u;
            const double v = 120 + 60 * std::sin(x * 0.37) * std::cos(y * 0.29) + ((s >> 24) & 31) - 16;
            uint8_t* p = &img[(static_cast<size_t>(y) * w + x) * 4];
            p[0] = static_cast<uint8_t>(v);
            p[1] = static_cast<uint8_t>(v * 0.8);
            p[2] = static_cast<uint8_t>(255 - v);
            p[3] = 255;
        }
    return img;
}

void testEqual() {
    const auto a = texture(40, 30);
    const auto r = broimage::compare_u8(a.data(), a.data(), 40, 30, 4);
    CHECK(r.all.rms == 0 && r.all.max_abs == 0 && r.all.changed == 0);
    CHECK(std::isinf(r.all.psnr));
    CHECK(near(r.all.ssim, 1.0));
    CHECK(r.cells.size() == 1 && r.cells[0].w == 40 && r.cells[0].h == 30);
}

void testBrighten() {
    const int w = 32, h = 16;
    auto a = texture(w, h);
    // Clear headroom, then add 10 to red only.
    for (size_t i = 0; i < a.size(); i += 4) a[i] = static_cast<uint8_t>(a[i] % 200);
    auto b = a;
    for (size_t i = 0; i < b.size(); i += 4) b[i] += 10;
    const auto r = broimage::compare_u8(a.data(), b.data(), w, h, 4);
    CHECK(near(r.all.mean[0], 10.0) && r.all.mean[1] == 0 && r.all.mean[2] == 0 && r.all.mean[3] == 0);
    // One channel in four at 10: mse 25, rms 5.
    CHECK(near(r.all.rms, 5.0));
    CHECK(near(r.all.psnr, 10.0 * std::log10(255.0 * 255.0 / 25.0)));
    CHECK(r.all.max_abs == 10);
    CHECK(r.all.changed == 1.0);
    const auto loose = broimage::compare_u8(a.data(), b.data(), w, h, 4, 0, 0, 1, 1, 10);
    CHECK(loose.all.changed == 0.0);
}

void testCell() {
    const int w = 64, h = 48;
    const auto a = texture(w, h);
    auto b = a;
    // Darken the bottom-right cell of a 2 x 2 grid.
    for (int y = h / 2; y < h; ++y)
        for (int x = w / 2; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                uint8_t& v = b[(static_cast<size_t>(y) * w + x) * 4 + c];
                v = static_cast<uint8_t>(v / 2);
            }
    const auto r = broimage::compare_u8(a.data(), b.data(), w, h, 4, 0, 0, 2, 2);
    CHECK(r.cells.size() == 4);
    for (int i = 0; i < 3; ++i) CHECK(r.cells[i].stats.rms == 0 && near(r.cells[i].stats.ssim, 1.0));
    CHECK(r.cells[3].stats.rms > 10 && r.cells[3].stats.mean[0] < -10 && r.cells[3].stats.changed > 0.9);
    CHECK(r.cells[3].x == 32 && r.cells[3].y == 24 && r.cells[3].w == 32 && r.cells[3].h == 24);
    // The whole is the cells pooled: a quarter of the pixels changed.
    CHECK(near(r.all.changed, r.cells[3].stats.changed / 4));
}

void testStructure() {
    // Grain, all fine structure: a blur takes most of it away.
    const int w = 48, h = 48;
    std::vector<uint8_t> a(static_cast<size_t>(w) * h * 4, 255);
    uint32_t s = 99;
    for (size_t i = 0; i < a.size(); ++i) {
        s = s * 1664525u + 1013904223u;
        if (i % 4 != 3) a[i] = static_cast<uint8_t>(60 + ((s >> 24) % 120));
    }
    auto shifted = a, blurred = a;
    for (size_t i = 0; i < shifted.size(); ++i)
        if (i % 4 != 3) shifted[i] = static_cast<uint8_t>(std::fmin(a[i] + 20, 255));
    for (int y = 1; y + 1 < h; ++y)
        for (int x = 1; x + 1 < w; ++x)
            for (int c = 0; c < 3; ++c) {
                int s = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) s += a[(static_cast<size_t>(y + dy) * w + x + dx) * 4 + c];
                blurred[(static_cast<size_t>(y) * w + x) * 4 + c] = static_cast<uint8_t>((s + 4) / 9);
            }
    const auto rs = broimage::compare_u8(a.data(), shifted.data(), w, h, 4);
    const auto rb = broimage::compare_u8(a.data(), blurred.data(), w, h, 4);
    CHECK(rs.all.ssim > 0.95);
    CHECK(rb.all.ssim < 0.6);
}

void testGridAndStride() {
    const int w = 37, h = 23;
    const auto a = texture(w, h);
    auto b = a;
    b[5 * 4] ^= 0x40;
    const auto r = broimage::compare_u8(a.data(), b.data(), w, h, 4, 0, 0, 3, 5);
    CHECK(r.cells.size() == 15);
    int area = 0;
    for (const auto& c : r.cells) area += c.w * c.h;
    CHECK(area == w * h);
    // A padded copy compares the same.
    const int stride = w * 4 + 12;
    std::vector<uint8_t> pa(static_cast<size_t>(stride) * h), pb(pa.size());
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w * 4; ++x) {
            pa[static_cast<size_t>(y) * stride + x] = a[static_cast<size_t>(y) * w * 4 + x];
            pb[static_cast<size_t>(y) * stride + x] = b[static_cast<size_t>(y) * w * 4 + x];
        }
    const auto rp = broimage::compare_u8(pa.data(), pb.data(), w, h, 4, stride, stride, 3, 5);
    CHECK(rp.all.rms == r.all.rms && rp.all.ssim == r.all.ssim && rp.all.max_abs == 64);
}

} // namespace

int main() {
    testEqual();
    testBrighten();
    testCell();
    testStructure();
    testGridAndStride();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("compare: all passed\n");
    return 0;
}
