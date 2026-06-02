#pragma once

#include <cstdint>

namespace broimage {

// ----- Tiled processing: feather window + weighted accumulate ----------------
//
// The building blocks for "split a large image into overlapping tiles, run a
// local operator (an edge detector, a per-tile model pass, a tiled upscale) on
// each tile, then glue the per-tile outputs back into one seamless full-res
// map". Tiling keeps the working resolution — and therefore the memory — of
// each pass bounded regardless of how large the source is.
//
// The seam between two tiles is hidden by a *feather window*: a per-pixel
// weight that is ~1 in the tile interior and ramps smoothly to ~0 across each
// overlapped edge. Every tile contributes `value * window` into a shared
// accumulator and `window` into a parallel weight canvas; a final divide
// (`normalize_accumulator_f32`) turns the weighted sums back into a blended
// map. Where two feathered tiles overlap, their ramps sum toward 1, so the
// transition is gradual and the join is invisible for a locally-computed map.
//
// All buffers are float32. Maps are HWC interleaved (the natural layout for the
// grayscale edge/line maps and RGB control images these glue back together);
// the weight canvas is single-channel.

// Fill a single-channel feather window for a tile of `tw x th` pixels.
//
// `ov_l/ov_r/ov_t/ov_b` are the overlap widths (in pixels) on the left / right
// / top / bottom edges. An edge that ramps up over its overlap blends with the
// neighbour that covers the same region; an edge with overlap 0 keeps full
// weight right to the boundary — which is exactly what a tile sitting on the
// image border wants (no neighbour there to share with, so its border pixels
// must not be feathered away).
//
// The ramp is a raised cosine (smooth, C1) that stays strictly positive at the
// very edge, so even a region covered by a single tile's ramp normalizes
// cleanly. If the two overlaps on an axis would collide (`ov_l + ov_r > tw`)
// they are clamped to fit. `w` must hold at least `tw * th` floats.
void feather_window_f32(float* w, int tw, int th,
                        int ov_l, int ov_r, int ov_t, int ov_b);

// Scatter one tile into the full-size accumulators, weighted by `window`.
//
// For every tile pixel (tx, ty) landing at full-image pixel (dst_x+tx,
// dst_y+ty) that falls inside the [0, full_w) x [0, full_h) canvas:
//   acc [(fy*full_w + fx)*channels + c] += tile[(ty*tw + tx)*channels + c] * win
//   wacc[ fy*full_w + fx]               += win
// where `win = window[ty*tw + tx]`. Pixels that fall outside the canvas are
// skipped (tiles may be clipped at the right/bottom edge).
//
// `acc` is HWC interleaved (`full_w*full_h*channels` floats), `wacc` is
// single-channel (`full_w*full_h` floats); both must be zero-initialized before
// the first tile and reused across every tile of one pass. `tile` is HWC
// interleaved (`tw*th*channels`), `window` is single-channel (`tw*th`).
void accumulate_tile_f32(float* acc, float* wacc,
                         int full_w, int full_h, int channels,
                         const float* tile, int tw, int th,
                         int dst_x, int dst_y, const float* window);

// Resolve the accumulators into the final blended map, in place:
//   acc[i*channels + c] /= max(wacc[i], eps)
// `eps` guards pixels no tile covered (weight 0) from producing NaN/Inf; they
// resolve to 0. `acc` holds `n_pixels*channels` floats, `wacc` holds
// `n_pixels`.
void normalize_accumulator_f32(float* acc, const float* wacc,
                               int n_pixels, int channels, float eps = 1e-6f);

} // namespace broimage
