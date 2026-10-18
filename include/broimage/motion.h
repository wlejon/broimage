// Block motion between two frames: the hot half of broworkshop's motion
// oracle (lib/kit/motion.js), which follows every block of a frame to the
// next and measures what its motion leaves unexplained.
//
// The frames come as luma planes at three scales: full (W x H), half
// (W/2 x H/2) and quarter (W/4 x H/4), each a box downsample of the one
// above. Blocks are B x B at half resolution (2B x 2B full), on a grid of
// nx x ny starting `margin` half-res px in from the top-left corner.
//
// The arithmetic is double precision in the order the JS reference does it,
// with no contraction into fused multiply-adds, so the results are the
// JS's to the bit, on every machine and for any thread count (each block is
// independent of the others).
#pragma once

#include <cstdint>

namespace broimage {

struct BlockMotionGrid {
    int width = 0, height = 0;   // full-resolution plane
    int nx = 0, ny = 0;          // blocks
    int block = 8;               // B: block size at half resolution
    int margin = 16;             // half-res px before the first block
    int radius = 4;              // coarse search radius at quarter resolution
};

// A frame's three luma planes from RGBA8 rows: full[i] = (54 R + 183 G +
// 19 B) >> 8, then half and quarter, each pixel the rounded mean of the 2 x 2
// above it ((a + b + c + d + 2) >> 2), the odd last row or column left out.
// rgba: width x height pixels, stride_bytes apart (0: packed). half and
// quarter may be null.
void motion_luma_u8(const uint8_t* rgba, int width, int height, int stride_bytes,
                    uint8_t* full, uint8_t* half, uint8_t* quarter, int threads = 0);

// One frame pair's match, block by block. Per block b, out[6 b ..] =
//   dx, dy   motion (full-res px)
//   P        mean luma change along the motion
//   D        rms of what is left (less its mean)
//   sigma    the block's texture in frame a (luma std)
//   status   0 tracked, 1 lost (past the search), 2 untextured (sigma under
//            lost_texture: P and D at zero motion only)
// Coarse to fine: quarter resolution +-radius, half +-2, then up to four
// inverse-compositional Lucas-Kanade steps at full resolution.
// threads: 0 = decide, else that many.
void block_motion_u8(const uint8_t* full_a, const uint8_t* full_b,
                     const uint8_t* half_a, const uint8_t* half_b,
                     const uint8_t* quarter_a, const uint8_t* quarter_b,
                     const BlockMotionGrid& grid, double lost_texture,
                     double* out, int threads = 0);

// The structure a pair leaves when each half-res pixel of a block may take
// the best of several motions: zero, its own block's and its 8 neighbours'
// (those whose motion is finite). Per block whose d[b] is finite, d[b]
// becomes the rms of that residual less its mean; motion is per block
// (dx, dy) in full-res px, NaN when the block has none.
void block_residual_multi_u8(const uint8_t* half_a, const uint8_t* half_b,
                             const BlockMotionGrid& grid, const float* motion,
                             float* d, int threads = 0);

} // namespace broimage
