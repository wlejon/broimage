#pragma once

#include "broimage/geometric.h"

#include <cstdint>

namespace broimage {

// ----- Alpha premultiplication -----------------------------------------------
//
// Why this exists: a straight-alpha RGBA image stores its color channels as-is
// regardless of alpha. When that image is filtered (resize, blur, blend) the
// transparent pixels' arbitrary RGB values leak into the result and produce
// the classic dark / colored fringe around composited edges.
//
// Premultiplying — R*=a/255, G*=a/255, B*=a/255 — makes transparent pixels
// contribute zero to filtered sums, fixing the fringing. The pair of helpers
// here let callers move into and out of premultiplied space; the
// `_alpha`-suffixed geometric variants do the round-trip internally.
//
// All ops operate on tightly-packed RGBA8 buffers (4 channels). Source and
// destination may alias.

void premultiply_alpha_rgba8(const uint8_t* src, uint8_t* dst, int pixel_count);
void unpremultiply_alpha_rgba8(const uint8_t* src, uint8_t* dst, int pixel_count);

// Resize an RGBA8 image with alpha-aware filtering: premultiplies the source
// to a scratch buffer, resizes, then unpremultiplies into `dst`. Use this
// for any RGBA composite where the transparent regions carry arbitrary RGB
// (decoded PNGs, sprite atlases, glyph alphas). Straight-alpha resize via
// `resize_hwc_u8` is fine when the input is known opaque or when the alpha
// is binary and the pad color doesn't matter.
void resize_rgba8_alpha(const uint8_t* src, int src_w, int src_h,
                        uint8_t*       dst, int dst_w, int dst_h,
                        Filter filter = Filter::Bilinear);

// Letterbox an RGBA8 source into an RGBA8 destination using the alpha-aware
// resize internally. `pad` channels are interpreted as straight RGBA — the
// pad region is written verbatim, not multiplied.
void letterbox_rgba8_alpha(const uint8_t* src, int src_w, int src_h,
                           uint8_t*       dst, int dst_w, int dst_h,
                           uint8_t pad_r, uint8_t pad_g, uint8_t pad_b, uint8_t pad_a,
                           Filter filter = Filter::Bilinear,
                           int* out_x = nullptr, int* out_y = nullptr,
                           int* out_w = nullptr, int* out_h = nullptr);

} // namespace broimage
