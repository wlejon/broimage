// Sharpness: how much fine detail a region of an 8-bit image holds, the
// focus measure for picking the sharpest frame of a burst.
//
// Two classic measures over the region's luma, computed in one pass:
//   - variance of the Laplacian (the 4-neighbour kernel 4c - n - s - e - w):
//     a blurred region has a flat Laplacian, a sharp one a wide spread;
//   - Tenengrad: the mean squared Sobel gradient magnitude, gx^2 + gy^2, a
//     little less sensitive to pixel noise than the Laplacian.
// Both grow with contrast as well as with focus, so they compare frames of
// the same subject (a burst), not unrelated photos; dividing by the region's
// luma variance (luma_std^2) takes the contrast out when that matters.
//
// Luma is BT.601 in integers ((77 R + 150 G + 29 B + 128) >> 8), and every
// sum is an integer, so the result is the same on every machine and for any
// thread count. Only pixels whose 3 x 3 neighbourhood lies inside the region
// count (a region under 3 x 3 has none and returns zeros).
#pragma once

#include <cstdint>

namespace broimage {

struct SharpnessResult {
    double laplacian_variance = 0.0;  // variance of the Laplacian, luma 0..255 units
    double tenengrad = 0.0;           // mean of Sobel gx^2 + gy^2
    double luma_mean = 0.0;           // over the counted pixels
    double luma_std = 0.0;            // their standard deviation (contrast)
    int64_t pixels = 0;               // pixels counted: (rw - 2) * (rh - 2)
};

// src: HWC u8 rows of w x h pixels, 1 (luma), 2 (luma + alpha), 3 (RGB) or 4
// (RGBA) channels; stride_bytes 0 = tightly packed. The region (rx, ry, rw, rh)
// is clamped to the image; rw or rh <= 0 means to the image's edge. threads:
// 0 = decide (one thread under a megapixel, else up to 8), else that many.
SharpnessResult sharpness_u8(const uint8_t* src, int w, int h, int channels,
                             int stride_bytes = 0,
                             int rx = 0, int ry = 0, int rw = 0, int rh = 0,
                             int threads = 0);

} // namespace broimage
