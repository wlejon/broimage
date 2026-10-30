// compare_u8: the difference statistics and luma SSIM of two images, overall
// and per grid cell (include/broimage/compare.h).

#include "broimage/compare.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace broimage {

namespace {

constexpr double kC1 = (0.01 * 255) * (0.01 * 255);
constexpr double kC2 = (0.03 * 255) * (0.03 * 255);

struct Acc {
    int64_t sum[4] = {0, 0, 0, 0};
    int64_t sq = 0;
    int max_abs = 0;
    int64_t changed = 0;
    int64_t pixels = 0;
    double ssim = 0.0;
    int64_t windows = 0;
};

uint8_t lumaOf(const uint8_t* p, int channels) {
    if (channels < 3) return p[0];
    return static_cast<uint8_t>((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
}

// The SSIM of luma planes la, lb (row length w) over one window.
double windowSsim(const uint8_t* la, const uint8_t* lb, int w, int x0, int y0, int ww, int wh) {
    int64_t sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (int y = y0; y < y0 + wh; ++y) {
        const uint8_t* ra = la + static_cast<size_t>(y) * w;
        const uint8_t* rb = lb + static_cast<size_t>(y) * w;
        for (int x = x0; x < x0 + ww; ++x) {
            const int va = ra[x], vb = rb[x];
            sa += va;
            sb += vb;
            saa += va * va;
            sbb += vb * vb;
            sab += va * vb;
        }
    }
    const double n = static_cast<double>(ww) * wh;
    const double ma = sa / n, mb = sb / n;
    const double va = saa / n - ma * ma, vb = sbb / n - mb * mb, cov = sab / n - ma * mb;
    return ((2 * ma * mb + kC1) * (2 * cov + kC2)) / ((ma * ma + mb * mb + kC1) * (va + vb + kC2));
}

CompareStats finish(const Acc& a, int channels) {
    CompareStats s;
    if (a.pixels == 0) return s;
    const double n = static_cast<double>(a.pixels);
    for (int c = 0; c < channels; ++c) s.mean[c] = a.sum[c] / n;
    const double mse = a.sq / (n * channels);
    s.rms = std::sqrt(mse);
    s.psnr = mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : std::numeric_limits<double>::infinity();
    s.max_abs = a.max_abs;
    s.changed = a.changed / n;
    s.ssim = a.windows > 0 ? a.ssim / static_cast<double>(a.windows) : 1.0;
    return s;
}

} // namespace

CompareResult compare_u8(const uint8_t* a, const uint8_t* b, int w, int h, int channels,
                         int a_stride, int b_stride, int rows, int cols, int threshold) {
    CompareResult out;
    if (!a || !b || w <= 0 || h <= 0 || channels < 1 || channels > 4) return out;
    if (a_stride <= 0) a_stride = w * channels;
    if (b_stride <= 0) b_stride = w * channels;
    rows = std::clamp(rows, 1, h);
    cols = std::clamp(cols, 1, w);

    // Each column's and row's cell, and the cells' bounds.
    std::vector<int> cellX(w), cellY(h);
    for (int c = 0; c < cols; ++c)
        for (int x = static_cast<int>(static_cast<int64_t>(c) * w / cols); x < static_cast<int64_t>(c + 1) * w / cols; ++x) cellX[x] = c;
    for (int r = 0; r < rows; ++r)
        for (int y = static_cast<int>(static_cast<int64_t>(r) * h / rows); y < static_cast<int64_t>(r + 1) * h / rows; ++y) cellY[y] = r;
    std::vector<Acc> acc(static_cast<size_t>(rows) * cols);
    Acc all;

    std::vector<uint8_t> la(static_cast<size_t>(w) * h), lb(la.size());
    for (int y = 0; y < h; ++y) {
        const uint8_t* ra = a + static_cast<size_t>(y) * a_stride;
        const uint8_t* rb = b + static_cast<size_t>(y) * b_stride;
        for (int x = 0; x < w; ++x) {
            const uint8_t* pa = ra + static_cast<size_t>(x) * channels;
            const uint8_t* pb = rb + static_cast<size_t>(x) * channels;
            Acc& k = acc[static_cast<size_t>(cellY[y]) * cols + cellX[x]];
            int worst = 0;
            for (int c = 0; c < channels; ++c) {
                const int d = pb[c] - pa[c];
                k.sum[c] += d;
                k.sq += d * d;
                worst = std::max(worst, d < 0 ? -d : d);
            }
            k.max_abs = std::max(k.max_abs, worst);
            k.changed += worst > threshold;
            ++k.pixels;
            la[static_cast<size_t>(y) * w + x] = lumaOf(pa, channels);
            lb[static_cast<size_t>(y) * w + x] = lumaOf(pb, channels);
        }
    }

    // SSIM windows, each counted in the cell holding its centre.
    const int ww = std::min(8, w), wh = std::min(8, h);
    for (int y0 = 0; y0 + wh <= h; y0 += 4) {
        for (int x0 = 0; x0 + ww <= w; x0 += 4) {
            const double s = windowSsim(la.data(), lb.data(), w, x0, y0, ww, wh);
            Acc& k = acc[static_cast<size_t>(cellY[y0 + wh / 2]) * cols + cellX[x0 + ww / 2]];
            k.ssim += s;
            ++k.windows;
            all.ssim += s;
            ++all.windows;
        }
    }

    out.cells.resize(acc.size());
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            Acc& k = acc[static_cast<size_t>(r) * cols + c];
            CompareCell& cell = out.cells[static_cast<size_t>(r) * cols + c];
            cell.x = static_cast<int>(static_cast<int64_t>(c) * w / cols);
            cell.y = static_cast<int>(static_cast<int64_t>(r) * h / rows);
            cell.w = static_cast<int>(static_cast<int64_t>(c + 1) * w / cols) - cell.x;
            cell.h = static_cast<int>(static_cast<int64_t>(r + 1) * h / rows) - cell.y;
            if (k.windows == 0) {
                k.ssim = windowSsim(la.data(), lb.data(), w, cell.x, cell.y, cell.w, cell.h);
                k.windows = 1;
            }
            for (int ch = 0; ch < channels; ++ch) all.sum[ch] += k.sum[ch];
            all.sq += k.sq;
            all.max_abs = std::max(all.max_abs, k.max_abs);
            all.changed += k.changed;
            all.pixels += k.pixels;
            cell.stats = finish(k, channels);
        }
    }
    out.all = finish(all, channels);
    return out;
}

} // namespace broimage
