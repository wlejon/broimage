// block_motion_u8 / block_residual_multi_u8: a textured frame shifted by a
// known whole- and sub-pixel motion is followed to it, a flat frame is
// untextured, a shift past the search is lost, a block whose own motion
// explains it leaves almost no residual, and every thread count gives the
// same bits.

#include "broimage/motion.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

namespace {

constexpr int W = 320, H = 240;

// Smooth texture, sampled at (x - sx, y - sy): the picture moved by (sx, sy).
std::vector<uint8_t> frame(double sx, double sy) {
    std::vector<uint8_t> img(size_t(W) * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const double u = x - sx, v = y - sy;
            const double t = 128 + 45 * std::sin(u * 0.21 + 0.4 * std::sin(v * 0.13)) * std::cos(v * 0.17) +
                             30 * std::sin((u + v) * 0.09);
            img[size_t(y) * W + x] = uint8_t(std::lround(std::fmin(std::fmax(t, 0.0), 255.0)));
        }
    return img;
}

// motion.js's half(): box downsample by 2, rounded.
std::vector<uint8_t> half(const std::vector<uint8_t>& s, int w, int h) {
    const int w2 = w >> 1, h2 = h >> 1;
    std::vector<uint8_t> o(size_t(w2) * h2);
    for (int y = 0; y < h2; ++y)
        for (int x = 0; x < w2; ++x) {
            const size_t r0 = size_t(2 * y) * w, r1 = r0 + w;
            o[size_t(y) * w2 + x] = uint8_t((s[r0 + 2 * x] + s[r0 + 2 * x + 1] + s[r1 + 2 * x] + s[r1 + 2 * x + 1] + 2) >> 2);
        }
    return o;
}

struct Pyr { std::vector<uint8_t> f, h, q; };
Pyr pyramid(std::vector<uint8_t> f) {
    Pyr p;
    p.f = std::move(f);
    p.h = half(p.f, W, H);
    p.q = half(p.h, W / 2, H / 2);
    return p;
}

broimage::BlockMotionGrid grid() {
    broimage::BlockMotionGrid g;
    g.width = W; g.height = H;
    g.nx = (W / 2 - 2 * g.margin) / g.block;
    g.ny = (H / 2 - 2 * g.margin) / g.block;
    return g;
}

std::vector<double> match(const Pyr& a, const Pyr& b, int threads) {
    const auto g = grid();
    std::vector<double> out(size_t(6) * g.nx * g.ny);
    broimage::block_motion_u8(a.f.data(), b.f.data(), a.h.data(), b.h.data(), a.q.data(), b.q.data(), g, 1.5,
                              out.data(), threads);
    return out;
}

bool sameBits(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

} // namespace

int main() {
    const auto g = grid();
    const int nb = g.nx * g.ny;
    const Pyr a = pyramid(frame(0, 0));

    // A sub-pixel shift: every textured block follows it.
    {
        const Pyr b = pyramid(frame(3.4, -1.7));
        const auto r = match(a, b, 1);
        int tracked = 0, close = 0;
        for (int i = 0; i < nb; ++i) {
            const double* o = &r[size_t(6) * i];
            if (o[5] != 0) continue;
            ++tracked;
            if (std::fabs(o[0] - 3.4) < 0.15 && std::fabs(o[1] + 1.7) < 0.15) ++close;
        }
        CHECK(tracked > nb * 2 / 3);
        CHECK(close > tracked * 9 / 10);
        // Thread invariance, to the bit.
        for (int t : {2, 3, 7, 16}) {
            const auto rt = match(a, b, t);
            CHECK(sameBits(r.data(), rt.data(), r.size() * sizeof(double)));
        }

        // The residual: each block's own motion explains it.
        std::vector<float> mv(size_t(2) * nb), d(nb), d1;
        for (int i = 0; i < nb; ++i) {
            const double* o = &r[size_t(6) * i];
            const bool ok = o[5] == 0;
            mv[2 * i] = ok ? float(o[0]) : NAN;
            mv[2 * i + 1] = ok ? float(o[1]) : NAN;
            d[i] = float(o[3]);
        }
        d1 = d;
        broimage::block_residual_multi_u8(a.h.data(), b.h.data(), g, mv.data(), d1.data(), 1);
        int small = 0;
        for (int i = 0; i < nb; ++i) small += d1[i] < 3;
        CHECK(small > nb * 9 / 10);
        for (int t : {2, 5, 16}) {
            auto dt = d;
            broimage::block_residual_multi_u8(a.h.data(), b.h.data(), g, mv.data(), dt.data(), t);
            CHECK(sameBits(d1.data(), dt.data(), d1.size() * sizeof(float)));
        }
        // A block whose D is NaN is left alone.
        auto dn = d;
        dn[5] = NAN;
        broimage::block_residual_multi_u8(a.h.data(), b.h.data(), g, mv.data(), dn.data(), 1);
        CHECK(std::isnan(dn[5]));
    }

    // Past the search (quarter-res radius 4 = 16 px): lost.
    {
        const Pyr b = pyramid(frame(22, 0));
        const auto r = match(a, b, 0);
        int lost = 0;
        for (int i = 0; i < nb; ++i) lost += r[size_t(6) * i + 5] == 1;
        CHECK(lost > nb * 8 / 10);
    }

    // A flat frame: untextured, P the brightness step, no motion.
    {
        const Pyr f0 = pyramid(std::vector<uint8_t>(size_t(W) * H, 100));
        const Pyr f1 = pyramid(std::vector<uint8_t>(size_t(W) * H, 104));
        const auto r = match(f0, f1, 0);
        bool ok = true;
        for (int i = 0; i < nb; ++i) {
            const double* o = &r[size_t(6) * i];
            ok = ok && o[5] == 2 && o[0] == 0 && o[1] == 0 && o[2] == 4 && o[3] == 0 && o[4] == 0;
        }
        CHECK(ok);
    }

    // The luma planes: motion.js's luma and half() exactly, odd sizes, any
    // thread count, a stride.
    for (const int w : {37, 64, 321})
        for (const int h : {1, 7, 30, 243}) {
            const int stride = w * 4 + 12;
            std::vector<uint8_t> rgba(size_t(stride) * h);
            uint32_t r = 7;
            for (auto& v : rgba) { r = r * 1664525u + 1013904223u; v = uint8_t(r >> 24); }
            std::vector<uint8_t> lf(size_t(w) * h);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const uint8_t* p = &rgba[size_t(y) * stride + 4 * x];
                    lf[size_t(y) * w + x] = uint8_t((54 * p[0] + 183 * p[1] + 19 * p[2]) >> 8);
                }
            const auto lh = half(lf, w, h), lq = half(lh, w / 2, h / 2);
            for (int t : {1, 3, 16}) {
                std::vector<uint8_t> f(lf.size()), hh(lh.size()), q(lq.size());
                broimage::motion_luma_u8(rgba.data(), w, h, stride, f.data(), hh.data(), q.data(), t);
                CHECK(f == lf);
                CHECK(hh == lh);
                CHECK(q == lq);
            }
        }

    if (g_failed) { std::fprintf(stderr, "%d check(s) failed\n", g_failed); return 1; }
    std::printf("test_motion: ok\n");
    return 0;
}
