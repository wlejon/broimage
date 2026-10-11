// Similarity alignment over a mask (coarse-to-fine Gauss-Newton with a
// photometric gain/bias and Huber weights) and the affine warp that renders
// the result. See include/broimage/align.h.
//
// Coordinates are pixel centres: pixel i sits at i + 0.5, so a 2x box
// downsample halves every coordinate exactly and a transform estimated at one
// pyramid level carries to the next by scaling its translation.

#include "broimage/align.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace broimage {

namespace {

struct Plane {
    int w = 0, h = 0;
    std::vector<float> v;
    float at(int x, int y) const { return v[static_cast<size_t>(y) * w + x]; }
};

struct Level {
    Plane ref, mov, gx, gy;           // luma 0..1; the moving image's gradient
    std::vector<uint8_t> mask;        // ref pixels used
};

Plane lumaOf(const uint8_t* src, int w, int h, int channels, int stride) {
    Plane p;
    p.w = w;
    p.h = h;
    p.v.resize(static_cast<size_t>(w) * h);
    const int rowBytes = stride ? stride : w * channels;
    for (int y = 0; y < h; ++y) {
        const uint8_t* row = src + static_cast<size_t>(y) * rowBytes;
        float* out = p.v.data() + static_cast<size_t>(y) * w;
        for (int x = 0; x < w; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * channels;
            float l = channels >= 3 ? (0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2]) : px[0];
            out[x] = l * (1.0f / 255.0f);
        }
    }
    return p;
}

Plane half(const Plane& s) {
    Plane d;
    d.w = s.w / 2;
    d.h = s.h / 2;
    d.v.resize(static_cast<size_t>(d.w) * d.h);
    for (int y = 0; y < d.h; ++y)
        for (int x = 0; x < d.w; ++x)
            d.v[static_cast<size_t>(y) * d.w + x] =
                0.25f * (s.at(2 * x, 2 * y) + s.at(2 * x + 1, 2 * y) +
                         s.at(2 * x, 2 * y + 1) + s.at(2 * x + 1, 2 * y + 1));
    return d;
}

std::vector<uint8_t> halfMask(const std::vector<uint8_t>& m, int w, int h) {
    const int dw = w / 2, dh = h / 2;
    std::vector<uint8_t> d(static_cast<size_t>(dw) * dh);
    for (int y = 0; y < dh; ++y)
        for (int x = 0; x < dw; ++x) {
            const size_t i = static_cast<size_t>(2 * y) * w + 2 * x;
            const int n = (m[i] != 0) + (m[i + 1] != 0) + (m[i + w] != 0) + (m[i + w + 1] != 0);
            d[static_cast<size_t>(y) * dw + x] = n >= 2 ? 1 : 0;
        }
    return d;
}

void gradients(const Plane& p, Plane& gx, Plane& gy) {
    gx.w = gy.w = p.w;
    gx.h = gy.h = p.h;
    gx.v.assign(p.v.size(), 0.0f);
    gy.v.assign(p.v.size(), 0.0f);
    for (int y = 0; y < p.h; ++y)
        for (int x = 0; x < p.w; ++x) {
            const int xl = std::max(x - 1, 0), xr = std::min(x + 1, p.w - 1);
            const int yu = std::max(y - 1, 0), yd = std::min(y + 1, p.h - 1);
            const size_t i = static_cast<size_t>(y) * p.w + x;
            gx.v[i] = (p.at(xr, y) - p.at(xl, y)) / static_cast<float>(xr - xl);
            gy.v[i] = (p.at(x, yd) - p.at(x, yu)) / static_cast<float>(yd - yu);
        }
}

// Bilinear sample at a pixel-centre position (px, py); false outside the
// image (a margin of half a pixel is allowed, edge pixels clamped).
inline bool sample(const Plane& p, double px, double py, float* out) {
    if (px < 0.0 || py < 0.0 || px > p.w || py > p.h) return false;
    const double u = px - 0.5, v = py - 0.5;
    int x0 = static_cast<int>(std::floor(u)), y0 = static_cast<int>(std::floor(v));
    const float fx = static_cast<float>(u - x0), fy = static_cast<float>(v - y0);
    const int x1 = std::min(std::max(x0 + 1, 0), p.w - 1);
    const int y1 = std::min(std::max(y0 + 1, 0), p.h - 1);
    x0 = std::min(std::max(x0, 0), p.w - 1);
    y0 = std::min(std::max(y0, 0), p.h - 1);
    const float a = p.at(x0, y0), b = p.at(x1, y0), c = p.at(x0, y1), d = p.at(x1, y1);
    *out = (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
    return true;
}

// The warp under estimation, as A = [[1 + a, -b], [b, 1 + a]] about c plus t.
struct Params {
    double a = 0, b = 0, tx = 0, ty = 0, gain = 1, bias = 0;
};

Params fromSimilarity(const Similarity& s, double f) {
    Params p;
    p.a = s.scale * std::cos(s.rotation) - 1.0;
    p.b = s.scale * std::sin(s.rotation);
    p.tx = s.tx * f;
    p.ty = s.ty * f;
    return p;
}

Similarity toSimilarity(const Params& p, double f) {
    Similarity s;
    s.scale = std::hypot(1.0 + p.a, p.b);
    s.rotation = std::atan2(p.b, 1.0 + p.a);
    s.tx = p.tx / f;
    s.ty = p.ty / f;
    return s;
}

// Solve n x n (n <= 6) by Gaussian elimination with partial pivoting.
bool solve(double A[6][6], double b[6], int n, double x[6]) {
    for (int c = 0; c < n; ++c) {
        int piv = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(A[r][c]) > std::fabs(A[piv][c])) piv = r;
        if (std::fabs(A[piv][c]) < 1e-12) return false;
        if (piv != c) {
            for (int k = 0; k < n; ++k) std::swap(A[c][k], A[piv][k]);
            std::swap(b[c], b[piv]);
        }
        for (int r = c + 1; r < n; ++r) {
            const double f = A[r][c] / A[c][c];
            for (int k = c; k < n; ++k) A[r][k] -= f * A[c][k];
            b[r] -= f * b[c];
        }
    }
    for (int r = n - 1; r >= 0; --r) {
        double s = b[r];
        for (int k = r + 1; k < n; ++k) s -= A[r][k] * x[k];
        x[r] = s / A[r][r];
    }
    return true;
}

struct Stats {
    double rms = 0, ncc = 0, coverage = 0;
};

// Residual statistics and NCC of the current warp over the mask.
Stats measure(const Level& L, const Params& p, double cx, double cy, size_t maskCount) {
    double sr = 0, sm = 0, srr = 0, smm = 0, srm = 0, se = 0;
    size_t n = 0;
    for (int y = 0; y < L.ref.h; ++y)
        for (int x = 0; x < L.ref.w; ++x) {
            if (!L.mask[static_cast<size_t>(y) * L.ref.w + x]) continue;
            const double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
            const double qx = cx + (1 + p.a) * dx - p.b * dy + p.tx;
            const double qy = cy + p.b * dx + (1 + p.a) * dy + p.ty;
            float m;
            if (!sample(L.mov, qx, qy, &m)) continue;
            const double r = L.ref.at(x, y);
            const double e = p.gain * m + p.bias - r;
            sr += r; sm += m; srr += r * r; smm += m * m; srm += r * m; se += e * e;
            ++n;
        }
    Stats s;
    if (!n) return s;
    s.rms = std::sqrt(se / n);
    s.coverage = maskCount ? static_cast<double>(n) / maskCount : 0;
    const double vr = srr - sr * sr / n, vm = smm - sm * sm / n;
    s.ncc = (vr > 0 && vm > 0) ? (srm - sr * sm / n) / std::sqrt(vr * vm) : 0;
    return s;
}

// Exhaustive whole-pixel shift search around p's translation, best NCC wins.
void searchShift(const Level& L, Params& p, double cx, double cy, int radius, size_t maskCount) {
    double best = -2;
    int bx = 0, by = 0;
    for (int sy = -radius; sy <= radius; ++sy)
        for (int sx = -radius; sx <= radius; ++sx) {
            Params q = p;
            q.tx += sx;
            q.ty += sy;
            const Stats s = measure(L, q, cx, cy, maskCount);
            // A shift that leaves most of the mask outside the moving image
            // cannot win on a lucky correlation of what is left.
            const double score = s.coverage >= 0.5 ? s.ncc : -2;
            if (score > best) { best = score; bx = sx; by = sy; }
        }
    p.tx += bx;
    p.ty += by;
}

// Gauss-Newton on one level. Returns the steps taken; *converged when the
// last step was below tolerance.
int refine(const Level& L, Params& p, double cx, double cy, const AlignOptions& o, bool* converged) {
    // Columns for a and b are scaled by the level's half-diagonal so the
    // normal equations are well conditioned whatever the image size.
    const double rho = 0.5 * std::hypot(static_cast<double>(L.ref.w), static_cast<double>(L.ref.h));
    int idx[6], n = 0;
    if (o.similarity) { idx[n++] = 0; idx[n++] = 1; }
    idx[n++] = 2; idx[n++] = 3;
    if (o.photometric) { idx[n++] = 4; idx[n++] = 5; }

    std::vector<float> absr;
    double huber = 1e9;   // no down-weighting until residuals are known
    *converged = false;
    int it = 0;
    for (; it < o.max_iterations; ++it) {
        double H[6][6] = {}, g[6] = {};
        absr.clear();
        for (int y = 0; y < L.ref.h; ++y)
            for (int x = 0; x < L.ref.w; ++x) {
                if (!L.mask[static_cast<size_t>(y) * L.ref.w + x]) continue;
                const double dx = x + 0.5 - cx, dy = y + 0.5 - cy;
                const double qx = cx + (1 + p.a) * dx - p.b * dy + p.tx;
                const double qy = cy + p.b * dx + (1 + p.a) * dy + p.ty;
                float m, mx, my;
                if (!sample(L.mov, qx, qy, &m) || !sample(L.gx, qx, qy, &mx) || !sample(L.gy, qx, qy, &my))
                    continue;
                const double r = p.gain * m + p.bias - L.ref.at(x, y);
                absr.push_back(static_cast<float>(std::fabs(r)));
                const double ar = std::fabs(r);
                const double w = ar <= huber ? 1.0 : huber / ar;
                const double J[6] = {
                    p.gain * (mx * dx + my * dy) / rho,
                    p.gain * (-mx * dy + my * dx) / rho,
                    p.gain * mx,
                    p.gain * my,
                    m,
                    1.0,
                };
                for (int i = 0; i < n; ++i) {
                    const double ji = w * J[idx[i]];
                    g[i] += ji * r;
                    for (int k = i; k < n; ++k) H[i][k] += ji * J[idx[k]];
                }
            }
        if (absr.size() < 16) break;
        for (int i = 0; i < n; ++i) {
            for (int k = 0; k < i; ++k) H[i][k] = H[k][i];
            H[i][i] *= 1.0 + 1e-6;   // a touch of damping against a flat direction
        }
        double rhs[6], d[6];
        for (int i = 0; i < n; ++i) rhs[i] = -g[i];
        if (!solve(H, rhs, n, d)) break;
        double step[6] = {};
        for (int i = 0; i < n; ++i) step[idx[i]] = d[i];
        p.a += step[0] / rho;
        p.b += step[1] / rho;
        p.tx += step[2];
        p.ty += step[3];
        p.gain += step[4];
        p.bias += step[5];

        // Huber threshold for the next step from this one's residuals (MAD).
        const size_t mid = absr.size() / 2;
        std::nth_element(absr.begin(), absr.begin() + mid, absr.end());
        huber = std::max(1.345 * 1.4826 * absr[mid], 1e-4);

        const double moved = std::hypot(step[2], step[3]) +
                             rho * (std::fabs(step[0] / rho) + std::fabs(step[1] / rho));
        if (moved < 1e-3) { *converged = true; ++it; break; }
    }
    return it;
}

} // namespace

void similarity_matrix(const Similarity& s, double cx, double cy, double m[6]) {
    const double c = s.scale * std::cos(s.rotation), sn = s.scale * std::sin(s.rotation);
    m[0] = c;  m[1] = -sn; m[2] = cx - c * cx + sn * cy + s.tx;
    m[3] = sn; m[4] = c;   m[5] = cy - sn * cx - c * cy + s.ty;
}

AlignResult align_similarity(const uint8_t* ref, const uint8_t* mov,
                             int w, int h, int channels,
                             const uint8_t* mask,
                             const Similarity& init,
                             const AlignOptions& options,
                             int ref_stride_bytes,
                             int mov_stride_bytes) {
    AlignResult res;
    res.transform = init;
    if (!ref || !mov || w <= 0 || h <= 0 || channels <= 0) return res;

    std::vector<Level> levels(1);
    levels[0].ref = lumaOf(ref, w, h, channels, ref_stride_bytes);
    levels[0].mov = lumaOf(mov, w, h, channels, mov_stride_bytes);
    levels[0].mask.assign(static_cast<size_t>(w) * h, 1);
    if (mask) for (size_t i = 0; i < levels[0].mask.size(); ++i) levels[0].mask[i] = mask[i] ? 1 : 0;
    while (static_cast<int>(levels.size()) <= options.levels) {
        const Level& up = levels.back();
        if (std::min(up.ref.w, up.ref.h) / 2 < 24) break;
        Level d;
        d.ref = half(up.ref);
        d.mov = half(up.mov);
        d.mask = halfMask(up.mask, up.ref.w, up.ref.h);
        levels.push_back(std::move(d));
    }
    for (Level& L : levels) gradients(L.mov, L.gx, L.gy);

    auto countOf = [](const std::vector<uint8_t>& m) {
        size_t n = 0;
        for (uint8_t v : m) n += v != 0;
        return n;
    };
    const size_t fullCount = countOf(levels[0].mask);
    if (fullCount < 16) return res;

    const double cx = 0.5 * w, cy = 0.5 * h;
    Params p;
    for (int li = static_cast<int>(levels.size()) - 1; li >= 0; --li) {
        Level& L = levels[li];
        // Exactly 2^-li even for odd sizes: a level's pixel i covers the
        // finer level's 2i and 2i + 1, whose centre is 2 (i + 0.5).
        const double f = std::ldexp(1.0, -li);
        const size_t count = countOf(L.mask);
        if (count < 16) continue;
        if (li == static_cast<int>(levels.size()) - 1) {
            p = fromSimilarity(init, f);
            if (options.search > 0) searchShift(L, p, cx * f, cy * f, options.search, count);
        } else {
            // Scale and rotation carry over; the translation scales with the level.
            p.tx *= 2.0;
            p.ty *= 2.0;
        }
        bool conv = false;
        res.iterations += refine(L, p, cx * f, cy * f, options, &conv);
        if (li == 0) res.converged = conv;
    }

    res.transform = toSimilarity(p, 1.0);
    res.gain = p.gain;
    res.bias = p.bias;
    const Stats s = measure(levels[0], p, cx, cy, fullCount);
    res.rms = s.rms;
    res.ncc = s.ncc;
    res.coverage = s.coverage;
    return res;
}

void warp_affine_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                        uint8_t* dst, int dst_w, int dst_h,
                        const double m[6],
                        Filter filter,
                        const uint8_t* fill,
                        int src_stride_bytes,
                        int dst_stride_bytes) {
    if (!src || !dst || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0 || channels <= 0) return;
    const int ss = src_stride_bytes ? src_stride_bytes : src_w * channels;
    const int ds = dst_stride_bytes ? dst_stride_bytes : dst_w * channels;
    auto px = [&](int x, int y) {
        x = std::min(std::max(x, 0), src_w - 1);
        y = std::min(std::max(y, 0), src_h - 1);
        return src + static_cast<size_t>(y) * ss + static_cast<size_t>(x) * channels;
    };
    auto cubic = [](double t, double w[4]) {   // Catmull-Rom
        const double t2 = t * t, t3 = t2 * t;
        w[0] = -0.5 * t3 + t2 - 0.5 * t;
        w[1] = 1.5 * t3 - 2.5 * t2 + 1.0;
        w[2] = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
        w[3] = 0.5 * t3 - 0.5 * t2;
    };
    for (int y = 0; y < dst_h; ++y) {
        uint8_t* out = dst + static_cast<size_t>(y) * ds;
        const double py = y + 0.5;
        for (int x = 0; x < dst_w; ++x, out += channels) {
            const double pxc = x + 0.5;
            const double sx = m[0] * pxc + m[1] * py + m[2];
            const double sy = m[3] * pxc + m[4] * py + m[5];
            if (sx < 0.0 || sy < 0.0 || sx > src_w || sy > src_h) {
                if (fill) std::memcpy(out, fill, channels);
                else std::memset(out, 0, channels);
                continue;
            }
            const double u = sx - 0.5, v = sy - 0.5;
            if (filter == Filter::Nearest) {
                std::memcpy(out, px(static_cast<int>(std::floor(sx)), static_cast<int>(std::floor(sy))), channels);
                continue;
            }
            const int x0 = static_cast<int>(std::floor(u)), y0 = static_cast<int>(std::floor(v));
            const double fx = u - x0, fy = v - y0;
            if (filter == Filter::Bicubic) {
                double wx[4], wy[4];
                cubic(fx, wx);
                cubic(fy, wy);
                for (int c = 0; c < channels; ++c) {
                    double acc = 0;
                    for (int j = 0; j < 4; ++j) {
                        double row = 0;
                        for (int i = 0; i < 4; ++i) row += wx[i] * px(x0 - 1 + i, y0 - 1 + j)[c];
                        acc += wy[j] * row;
                    }
                    out[c] = static_cast<uint8_t>(std::min(std::max(acc + 0.5, 0.0), 255.0));
                }
                continue;
            }
            const uint8_t *a = px(x0, y0), *b = px(x0 + 1, y0), *c0 = px(x0, y0 + 1), *d = px(x0 + 1, y0 + 1);
            for (int c = 0; c < channels; ++c) {
                const double top = a[c] + (b[c] - a[c]) * fx;
                const double bot = c0[c] + (d[c] - c0[c]) * fx;
                out[c] = static_cast<uint8_t>(top + (bot - top) * fy + 0.5);
            }
        }
    }
}

} // namespace broimage
