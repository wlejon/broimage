// Focus measures over a region's luma: variance of the Laplacian and
// Tenengrad (mean squared Sobel gradient). See include/broimage/sharpness.h.
//
// Each thread takes a band of rows, converts the rows it needs (its band plus
// one above and one below) to integer luma once, and sums in 64-bit integers;
// the bands' sums add exactly, so the result does not depend on the split.

#include "broimage/sharpness.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

namespace broimage {

namespace {

struct Sums {
    int64_t lap = 0, lap2 = 0, ten = 0, y = 0, y2 = 0, n = 0;
    void add(const Sums& o) {
        lap += o.lap; lap2 += o.lap2; ten += o.ten; y += o.y; y2 += o.y2; n += o.n;
    }
};

inline int lumaAt(const uint8_t* p, int channels) {
    if (channels < 3) return p[0];
    return (77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8;
}

// Rows [y0, y1) of the region's interior (image rows), columns rx+1 .. rx+rw-2.
Sums band(const uint8_t* src, int channels, int rowBytes, int rx, int rw, int y0, int y1) {
    Sums s;
    const int rows = y1 - y0 + 2;
    std::vector<int32_t> luma(static_cast<size_t>(rows) * rw);
    for (int r = 0; r < rows; ++r) {
        const uint8_t* row = src + static_cast<size_t>(y0 - 1 + r) * rowBytes + static_cast<size_t>(rx) * channels;
        int32_t* out = luma.data() + static_cast<size_t>(r) * rw;
        for (int x = 0; x < rw; ++x) out[x] = lumaAt(row + static_cast<size_t>(x) * channels, channels);
    }
    for (int r = 1; r + 1 < rows; ++r) {
        const int32_t* a = luma.data() + static_cast<size_t>(r - 1) * rw;   // row above
        const int32_t* c = a + rw;                                          // this row
        const int32_t* b = c + rw;                                          // row below
        int64_t lap = 0, lap2 = 0, ten = 0, ys = 0, y2 = 0;
        for (int x = 1; x + 1 < rw; ++x) {
            const int32_t v = c[x];
            const int32_t l = 4 * v - c[x - 1] - c[x + 1] - a[x] - b[x];
            const int32_t gx = (a[x + 1] + 2 * c[x + 1] + b[x + 1]) - (a[x - 1] + 2 * c[x - 1] + b[x - 1]);
            const int32_t gy = (b[x - 1] + 2 * b[x] + b[x + 1]) - (a[x - 1] + 2 * a[x] + a[x + 1]);
            lap += l;
            lap2 += int64_t(l) * l;
            ten += int64_t(gx) * gx + int64_t(gy) * gy;
            ys += v;
            y2 += int64_t(v) * v;
        }
        s.lap += lap; s.lap2 += lap2; s.ten += ten; s.y += ys; s.y2 += y2;
        s.n += rw - 2;
    }
    return s;
}

} // namespace

SharpnessResult sharpness_u8(const uint8_t* src, int w, int h, int channels,
                             int stride_bytes, int rx, int ry, int rw, int rh, int threads) {
    SharpnessResult res;
    if (!src || w <= 0 || h <= 0 || channels < 1 || channels > 4) return res;
    rx = std::clamp(rx, 0, w);
    ry = std::clamp(ry, 0, h);
    rw = rw <= 0 ? w - rx : std::min(rw, w - rx);
    rh = rh <= 0 ? h - ry : std::min(rh, h - ry);
    if (rw < 3 || rh < 3) return res;
    const int rowBytes = stride_bytes ? stride_bytes : w * channels;

    // Interior rows of the region.
    const int first = ry + 1, last = ry + rh - 1;   // [first, last)
    const int interior = last - first;
    if (threads <= 0) {
        const int64_t work = int64_t(rw) * rh;
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        threads = work < (int64_t(1) << 20) ? 1 : int(std::min(hw, 8u));
    }
    threads = std::max(1, std::min(threads, interior));

    Sums total;
    if (threads == 1) {
        total = band(src, channels, rowBytes, rx, rw, first, last);
    } else {
        std::vector<Sums> parts(threads);
        std::vector<std::thread> pool;
        auto span = [&](int t) { return first + int(int64_t(interior) * t / threads); };
        for (int t = 1; t < threads; ++t)
            pool.emplace_back([&, t] { parts[t] = band(src, channels, rowBytes, rx, rw, span(t), span(t + 1)); });
        parts[0] = band(src, channels, rowBytes, rx, rw, span(0), span(1));
        for (std::thread& th : pool) th.join();
        for (const Sums& p : parts) total.add(p);
    }
    if (total.n <= 0) return res;

    const double n = double(total.n);
    const double lapMean = double(total.lap) / n;
    const double yMean = double(total.y) / n;
    res.laplacian_variance = std::max(0.0, double(total.lap2) / n - lapMean * lapMean);
    res.tenengrad = double(total.ten) / n;
    res.luma_mean = yMean;
    res.luma_std = std::sqrt(std::max(0.0, double(total.y2) / n - yMean * yMean));
    res.pixels = total.n;
    return res;
}

} // namespace broimage
