// Block motion between two frames (include/broimage/motion.h): a port of
// broworkshop lib/kit/motion.js's matchBlock and multiResidual that gives
// the same numbers. A read outside a plane is NaN, as a typed array's is in
// JS; blocks whose whole footprint lies inside every plane (all of them, for
// the oracle's margins) take the unchecked path.

#include "broimage/motion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <thread>
#include <vector>

namespace broimage {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// A luma plane, read as JS reads a Uint8Array: out of range is NaN.
struct Plane {
    const uint8_t* p;
    int64_t n;
    template <bool Checked>
    double at(int64_t i) const {
        if constexpr (Checked) {
            if (i < 0 || i >= n) return kNaN;
        }
        return p[i];
    }
};

// Math.max(0, v): NaN stays NaN (std::max would give 0).
double max0(double v) { return std::isnan(v) ? v : std::max(0.0, v); }

// Math.max(-1, Math.min(1, v)), NaN passing through.
double clampUnit(double v) {
    if (std::isnan(v)) return v;
    return std::max(-1.0, std::min(1.0, v));
}

template <bool Checked>
double ssd(const Plane& a, const Plane& b, int W, int ax, int ay, int bx, int by, int n) {
    if constexpr (!Checked) {
        int64_t s = 0;
        for (int j = 0; j < n; ++j) {
            const uint8_t* pa = a.p + int64_t(ay + j) * W + ax;
            const uint8_t* pb = b.p + int64_t(by + j) * W + bx;
            for (int i = 0; i < n; ++i) { const int d = int(pa[i]) - int(pb[i]); s += d * d; }
        }
        return double(s);
    } else {
        double s = 0;
        for (int j = 0; j < n; ++j) {
            int64_t ia = int64_t(ay + j) * W + ax, ib = int64_t(by + j) * W + bx;
            for (int i = 0; i < n; ++i, ++ia, ++ib) { const double d = a.at<true>(ia) - b.at<true>(ib); s += d * d; }
        }
        return s;
    }
}

struct Ctx {
    Plane fa, fb, ha, hb, qa, qb;
    int W, W2;
    int B, margin, R;
    double lostTexture;
};

// matchBlock: x0, y0 the block's corner at half resolution; lk scratch of
// 3 (2B)^2 doubles.
template <bool Checked>
void matchBlock(const Ctx& c, int x0, int y0, double* lk, double* out) {
    const int B = c.B, R = c.R, W = c.W, W2 = c.W2;
    const int F = 2 * B, FF = F * F, X0 = 2 * x0, Y0 = 2 * y0, Wq = W2 >> 1;
    const Plane &fa = c.fa, &fb = c.fb;
    double s = 0, s2 = 0;
    for (int j = 0; j < F; ++j) {
        const int64_t r = int64_t(Y0 + j) * W + X0;
        for (int i = 0; i < F; ++i) { const double v = fa.at<Checked>(r + i); s += v; s2 += v * v; }
    }
    const double sig = std::sqrt(max0(s2 / FF - (s / FF) * (s / FF)));
    out[4] = sig;
    double dx = 0, dy = 0;
    int status = 0;
    if (sig < c.lostTexture) status = 2;
    else {
        const int qx = x0 >> 1, qy = y0 >> 1;
        double best = kInf;
        int bx = 0, by = 0;
        for (int v = -R; v <= R; ++v)
            for (int u = -R; u <= R; ++u) {
                const double cst = ssd<Checked>(c.qa, c.qb, Wq, qx, qy, qx + u, qy + v, 4) + 2 * (std::abs(u) + std::abs(v));
                if (cst < best) { best = cst; bx = u; by = v; }
            }
        if (std::abs(bx) == R || std::abs(by) == R) status = 1;
        else {
            best = kInf;
            int hx = 2 * bx, hy = 2 * by;
            for (int v = 2 * by - 2; v <= 2 * by + 2; ++v)
                for (int u = 2 * bx - 2; u <= 2 * bx + 2; ++u) {
                    const double cst = ssd<Checked>(c.ha, c.hb, W2, x0, y0, x0 + u, y0 + v, B) + 2 * (std::abs(u) + std::abs(v));
                    if (cst < best) { best = cst; hx = u; hy = v; }
                }
            const double fxb = 2 * hx, fyb = 2 * hy;
            dx = fxb; dy = fyb;
            double hxx = 0, hxy = 0, hyy = 0, gxs = 0, gys = 0;
            for (int j = 0, n = 0; j < F; ++j)
                for (int i = 0; i < F; ++i, ++n) {
                    const int64_t o = int64_t(Y0 + j) * W + X0 + i;
                    const double gx = 0.5 * (fa.at<Checked>(o + 1) - fa.at<Checked>(o - 1));
                    const double gy = 0.5 * (fa.at<Checked>(o + W) - fa.at<Checked>(o - W));
                    lk[3 * n] = gx; lk[3 * n + 1] = gy; lk[3 * n + 2] = fa.at<Checked>(o);
                    hxx += gx * gx; hxy += gx * gy; hyy += gy * gy; gxs += gx; gys += gy;
                }
            const double det = hxx * hyy - hxy * hxy;
            if (det > 1e-6 * (hxx + hyy) * (hxx + hyy) && det > 1) {
                for (int it = 0; it < 4; ++it) {
                    const double ixd = std::floor(dx), iyd = std::floor(dy), wx = dx - ixd, wy = dy - iyd;
                    const int ix = int(ixd), iy = int(iyd);
                    const double w00 = (1 - wx) * (1 - wy), w10 = wx * (1 - wy), w01 = (1 - wx) * wy, w11 = wx * wy;
                    double ex = 0, ey = 0, em = 0;
                    for (int j = 0, n = 0; j < F; ++j) {
                        int64_t o = int64_t(Y0 + j + iy) * W + X0 + ix;
                        for (int i = 0; i < F; ++i, ++n, ++o) {
                            const double e = w00 * fb.at<Checked>(o) + w10 * fb.at<Checked>(o + 1) +
                                             w01 * fb.at<Checked>(o + W) + w11 * fb.at<Checked>(o + W + 1) - lk[3 * n + 2];
                            em += e; ex += lk[3 * n] * e; ey += lk[3 * n + 1] * e;
                        }
                    }
                    em /= FF;
                    ex -= em * gxs; ey -= em * gys;
                    const double ux = (hyy * ex - hxy * ey) / det, uy = (hxx * ey - hxy * ex) / det;
                    dx -= clampUnit(ux); dy -= clampUnit(uy);
                    if (ux * ux + uy * uy < 1e-4) break;
                }
                if (std::abs(dx - fxb) > 2 || std::abs(dy - fyb) > 2) { dx = fxb; dy = fyb; }
            }
        }
    }
    // Brightness change along the motion, and the structure left. (A NaN
    // motion makes JS index with NaN: every read undefined, P and D NaN.)
    const double ixd = std::floor(dx), iyd = std::floor(dy), wx = dx - ixd, wy = dy - iyd;
    double m = kNaN, rms = kNaN;
    if (std::isfinite(ixd) && std::isfinite(iyd)) {
        const int ix = int(ixd), iy = int(iyd);
        const double w00 = (1 - wx) * (1 - wy), w10 = wx * (1 - wy), w01 = (1 - wx) * wy, w11 = wx * wy;
        double sb = 0, sd = 0;
        for (int j = 0; j < F; ++j) {
            int64_t o = int64_t(Y0 + j + iy) * W + X0 + ix, oa = int64_t(Y0 + j) * W + X0;
            for (int i = 0; i < F; ++i, ++o, ++oa) {
                const double d = w00 * fb.at<Checked>(o) + w10 * fb.at<Checked>(o + 1) + w01 * fb.at<Checked>(o + W) +
                                 w11 * fb.at<Checked>(o + W + 1) - fa.at<Checked>(oa);
                sb += d; sd += d * d;
            }
        }
        m = sb / FF;
        rms = std::sqrt(max0(sd / FF - m * m));
    }
    out[0] = dx; out[1] = dy; out[2] = m; out[3] = rms; out[5] = status;
}

// Whether every read matchBlock can make for the block at (x0, y0) lies in
// the planes: the search reaches 2 (R - 1) + 2 half-res px, Lucas-Kanade 2
// full-res px past that, and the bilinear taps and gradients one more.
bool inside(const Ctx& c, int H, int x0, int y0) {
    const int B = c.B, R = c.R, F = 2 * B;
    const int reach = 2 * (2 * (R - 1) + 2) + 2 + 1;   // full-res px
    const int X0 = 2 * x0, Y0 = 2 * y0;
    if (X0 - reach < 0 || Y0 - reach < 0 || X0 + F + reach >= c.W || Y0 + F + reach >= H) return false;
    return true;
}

int decideThreads(int threads, int rows) {
    if (threads <= 0) {
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        threads = int(std::min(hw, 16u));
    }
    return std::max(1, std::min(threads, rows));
}

template <class Fn>
void forRows(int rows, int threads, Fn&& fn) {
    threads = decideThreads(threads, rows);
    if (threads == 1) { fn(0, rows); return; }
    std::vector<std::thread> pool;
    auto span = [&](int t) { return int(int64_t(rows) * t / threads); };
    for (int t = 1; t < threads; ++t) pool.emplace_back([&, t] { fn(span(t), span(t + 1)); });
    fn(span(0), span(1));
    for (std::thread& th : pool) th.join();
}

// Box downsample by 2 of rows [y0, y1) of the output.
void halfRows(const uint8_t* src, int W, uint8_t* dst, int w, int y0, int y1) {
    for (int y = y0; y < y1; ++y) {
        const uint8_t* r0 = src + size_t(2 * y) * W;
        const uint8_t* r1 = r0 + W;
        uint8_t* o = dst + size_t(y) * w;
        for (int x = 0; x < w; ++x)
            o[x] = uint8_t((r0[2 * x] + r0[2 * x + 1] + r1[2 * x] + r1[2 * x + 1] + 2) >> 2);
    }
}

} // namespace

void motion_luma_u8(const uint8_t* rgba, int W, int H, int stride, uint8_t* full, uint8_t* half,
                    uint8_t* quarter, int threads) {
    if (!rgba || !full || W <= 0 || H <= 0) return;
    if (stride <= 0) stride = W * 4;
    const int W2 = W >> 1, H2 = H >> 1, Wq = W2 >> 1, Hq = H2 >> 1;
    // Bands of quarter rows (4 full rows each), so each band's half and
    // quarter rows come only from its own full rows; the full rows past the
    // last whole band in a final pass.
    const int bands = Hq;
    forRows(std::max(bands, 1), threads, [&](int b0, int b1) {
        const int f0 = bands ? 4 * b0 : 0, f1 = bands ? 4 * b1 : H;
        for (int y = f0; y < f1; ++y) {
            const uint8_t* p = rgba + size_t(y) * stride;
            uint8_t* o = full + size_t(y) * W;
            for (int x = 0; x < W; ++x, p += 4) o[x] = uint8_t((54 * p[0] + 183 * p[1] + 19 * p[2]) >> 8);
        }
        if (!bands) return;
        if (half) halfRows(full, W, half, W2, 2 * b0, 2 * b1);
        if (half && quarter) halfRows(half, W2, quarter, Wq, b0, b1);
    });
    for (int y = 4 * bands; y < H; ++y) {
        const uint8_t* p = rgba + size_t(y) * stride;
        uint8_t* o = full + size_t(y) * W;
        for (int x = 0; x < W; ++x, p += 4) o[x] = uint8_t((54 * p[0] + 183 * p[1] + 19 * p[2]) >> 8);
    }
    if (half && 2 * bands < H2) halfRows(full, W, half, W2, 2 * bands, H2);
}

void block_motion_u8(const uint8_t* full_a, const uint8_t* full_b,
                     const uint8_t* half_a, const uint8_t* half_b,
                     const uint8_t* quarter_a, const uint8_t* quarter_b,
                     const BlockMotionGrid& g, double lost_texture,
                     double* out, int threads) {
    if (g.width <= 0 || g.height <= 0 || g.nx <= 0 || g.ny <= 0 || g.block <= 0) return;
    const int W = g.width, H = g.height, W2 = W >> 1, H2 = H >> 1, Wq = W2 >> 1, Hq = H2 >> 1;
    const Ctx c{{full_a, int64_t(W) * H}, {full_b, int64_t(W) * H},
                {half_a, int64_t(W2) * H2}, {half_b, int64_t(W2) * H2},
                {quarter_a, int64_t(Wq) * Hq}, {quarter_b, int64_t(Wq) * Hq},
                W, W2, g.block, g.margin, g.radius, lost_texture};
    forRows(g.ny, threads, [&](int y0, int y1) {
        std::vector<double> lk(size_t(3) * 4 * g.block * g.block);
        for (int by = y0; by < y1; ++by)
            for (int bx = 0; bx < g.nx; ++bx) {
                const int x = g.margin + bx * g.block, y = g.margin + by * g.block;
                double* o = out + size_t(6) * (size_t(by) * g.nx + bx);
                if (inside(c, H, x, y)) matchBlock<false>(c, x, y, lk.data(), o);
                else matchBlock<true>(c, x, y, lk.data(), o);
            }
    });
}

void block_residual_multi_u8(const uint8_t* half_a, const uint8_t* half_b,
                             const BlockMotionGrid& g, const float* motion,
                             float* d, int threads) {
    if (g.width <= 0 || g.height <= 0 || g.nx <= 0 || g.ny <= 0 || g.block <= 0) return;
    const int W2 = g.width >> 1, H2 = g.height >> 1, B = g.block, nx = g.nx, ny = g.ny;
    const Plane ha{half_a, int64_t(W2) * H2}, hb{half_b, int64_t(W2) * H2};
    forRows(ny, threads, [&](int r0, int r1) {
        double hyp[2 * 10];
        for (int cy = r0; cy < r1; ++cy)
            for (int cx = 0; cx < nx; ++cx) {
                const int bi = cy * nx + cx;
                if (!std::isfinite(d[bi])) continue;
                int nh = 1;
                hyp[0] = hyp[1] = 0;
                for (int v = -1; v <= 1; ++v)
                    for (int u = -1; u <= 1; ++u) {
                        const int x = cx + u, y = cy + v;
                        if (x < 0 || y < 0 || x >= nx || y >= ny) continue;
                        const int n = y * nx + x;
                        if (std::isfinite(motion[2 * n])) {
                            hyp[2 * nh] = double(motion[2 * n]) / 2;
                            hyp[2 * nh + 1] = double(motion[2 * n + 1]) / 2;
                            ++nh;
                        }
                    }
                const int x0 = g.margin + cx * B, y0 = g.margin + cy * B;
                double s = 0, s2 = 0;
                for (int j = 0; j < B; ++j)
                    for (int i = 0; i < B; ++i) {
                        const double av = ha.at<true>(int64_t(y0 + j) * W2 + x0 + i);
                        double best = kInf, be = 0;
                        for (int h = 0; h < nh; ++h) {
                            const double X = x0 + i + hyp[2 * h], Y = y0 + j + hyp[2 * h + 1];
                            const double fx = std::floor(X), fy = std::floor(Y), wx = X - fx, wy = Y - fy;
                            double e = kNaN;
                            if (std::isfinite(fx) && std::isfinite(fy)) {
                                const int64_t o = int64_t(fy) * W2 + int64_t(fx);
                                e = (1 - wy) * ((1 - wx) * hb.at<true>(o) + wx * hb.at<true>(o + 1)) +
                                    wy * ((1 - wx) * hb.at<true>(o + W2) + wx * hb.at<true>(o + W2 + 1)) - av;
                            }
                            if (std::abs(e) < best) { best = std::abs(e); be = e; }
                        }
                        s += be; s2 += be * be;
                    }
                const double m = s / (B * B);
                d[bi] = float(std::sqrt(max0(s2 / (B * B) - m * m)));
            }
    });
}

} // namespace broimage
