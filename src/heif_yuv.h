// YUV to RGBA8 for decoded HEIF items (heif.h): the colour maths of
// ITU-T H.273 (matrix coefficients, range) and, for PQ / HLG, a tone-map
// to SDR. Written for AV1 frames a host decoder hands over (YuvPlanes).

#pragma once

#include "broimage/heif.h"

#include <cstdint>

namespace broimage::detail {

// The colour description a frame is converted with (H.273 code points).
struct ColorSpec {
    int primaries = 2;
    int transfer = 2;
    int matrix = 2;
    bool full_range = false;
    uint16_t max_cll = 0;  // clli, nits; 0 when unknown
};

// Convert `f` into the straight-RGBA8 `canvas` (cw x ch, rows of cw * 4
// bytes) with its top-left at (x0, y0), clipped to the canvas; alpha 255.
void yuv_to_rgba(const YuvPlanes& f, const ColorSpec& c, uint8_t* canvas, int cw, int ch, int x0, int y0);

// The luma plane of `f` into the alpha channel of `canvas` (an alpha
// auxiliary image), scaled from limited range when `full_range` is false.
void yuv_to_alpha(const YuvPlanes& f, bool full_range, uint8_t* canvas, int cw, int ch, int x0, int y0);

bool is_hdr_transfer(int transfer);

}  // namespace broimage::detail
