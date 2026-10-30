// Compare: how an 8-bit image b differs from a reference a of the same size,
// the measured oracle for a change of look (a render against the one before
// it). Overall and over a rows x cols grid of cells, so a change is placed
// as well as sized (the sky's band, the horizon's, the ground's):
//   - mean: b - a per channel, signed (brighter, bluer), 0..255 units;
//   - rms over every channel, and the PSNR it makes (dB, infinite when equal);
//   - max_abs, the largest channel difference;
//   - changed: the share of pixels with a channel differing by more than
//     `threshold`;
//   - ssim: the structural similarity of their luma (Wang et al. 2004) over
//     8 x 8 windows stepped by 4, constants (0.01 * 255)^2 and
//     (0.03 * 255)^2: 1 for the same structure, falling as edges and texture
//     move or blur, however the brightness shifts. A cell takes the windows
//     whose centres fall in it (a cell no window centre reaches, the cell as
//     one window).
//
// Luma is BT.601 in integers ((77 R + 150 G + 29 B + 128) >> 8); 1 and 2
// channel images are luma (+ alpha) already. Difference sums are integers
// and the SSIM sums run in one order, so the result is the same on every
// machine.
#pragma once

#include <cstdint>
#include <vector>

namespace broimage {

struct CompareStats {
    double mean[4] = {0, 0, 0, 0};   // b - a per channel
    double rms = 0.0;
    double psnr = 0.0;               // +inf when equal
    int max_abs = 0;
    double changed = 0.0;            // share of pixels, 0..1
    double ssim = 1.0;
};

struct CompareCell {
    int x = 0, y = 0, w = 0, h = 0;
    CompareStats stats;
};

struct CompareResult {
    CompareStats all;
    std::vector<CompareCell> cells;  // rows x cols, row-major
};

// a, b: HWC u8 rows of w x h pixels with `channels` (1..4) each; strides in
// bytes, 0 = tightly packed. rows, cols: the grid (clamped to the image's
// height and width). threshold: the difference a pixel's channel must pass
// to count as changed.
CompareResult compare_u8(const uint8_t* a, const uint8_t* b, int w, int h, int channels,
                         int a_stride = 0, int b_stride = 0,
                         int rows = 1, int cols = 1, int threshold = 8);

} // namespace broimage
