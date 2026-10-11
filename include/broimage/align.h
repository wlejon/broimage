// Image alignment: estimate the similarity transform (scale, rotation,
// translation) that maps a reference image onto a moving one, over a masked
// region, and warp an image by an affine transform.
//
// The use this was written for: stabilizing a subject (a bird) across the
// photos of a camera burst. A tracker supplies a mask and a rough position on
// each frame; align_similarity refines it to sub-pixel accuracy from the
// pixels under the mask, and warp_affine_hwc_u8 renders each frame with the
// subject held still.
#pragma once

#include "broimage/geometric.h"

#include <cstdint>

namespace broimage {

// A similarity about a centre c (the reference image's centre unless the
// caller moves it): a reference point p maps to the moving image at
//     c + scale * R(rotation) * (p - c) + (tx, ty)
// rotation in radians, counter-clockwise in image coordinates (y down, so it
// reads clockwise on screen). Identity is { 1, 0, 0, 0 }.
struct Similarity {
    double scale = 1.0;
    double rotation = 0.0;
    double tx = 0.0;
    double ty = 0.0;
};

struct AlignOptions {
    // Pyramid levels below full size (each half the size of the one above);
    // a level is not built once its shorter side would drop under 24 px.
    int levels = 4;
    // At the coarsest level, an exhaustive search of whole-pixel shifts this
    // far (in that level's pixels) around the initial translation, by
    // normalized cross-correlation, before refining. 0 skips it. It is what
    // lets a rough initial position (a mask's centroid) still converge.
    int search = 4;
    // Gauss-Newton iterations per level.
    int max_iterations = 40;
    // Estimate scale and rotation (false: translation only).
    bool similarity = true;
    // Estimate a gain and bias between the two images (exposure changes
    // between frames). false: pixel values are compared as they are.
    bool photometric = true;
};

struct AlignResult {
    Similarity transform;
    double gain = 1.0;      // moving * gain + bias ~ reference, under the mask
    double bias = 0.0;      //   (on luma in 0..1)
    double rms = 0.0;       // the residual's RMS at full size, after alignment (luma 0..1)
    double ncc = 0.0;       // normalized cross-correlation under the mask, -1..1
    double coverage = 0.0;  // share of the mask whose mapped point lies inside the moving image
    int iterations = 0;     // Gauss-Newton steps, all levels
    bool converged = false; // the full-size level's last step was below tolerance
};

// ref, mov: HWC u8 images of the same size, 1 (luma), 3 (RGB) or 4 (RGBA)
// channels; colour is compared as luma. mask: w*h bytes, nonzero = use the
// reference pixel, or null for the whole image. init: the starting estimate.
// Strides follow geometric.h (0 = tightly packed). A mask with fewer than 16
// pixels returns `init` with converged false.
AlignResult align_similarity(const uint8_t* ref, const uint8_t* mov,
                             int w, int h, int channels,
                             const uint8_t* mask,
                             const Similarity& init = Similarity(),
                             const AlignOptions& options = AlignOptions(),
                             int ref_stride_bytes = 0,
                             int mov_stride_bytes = 0);

// The 2x3 matrix m of a similarity about centre (cx, cy):
//     x' = m[0] * x + m[1] * y + m[2],   y' = m[3] * x + m[4] * y + m[5]
void similarity_matrix(const Similarity& s, double cx, double cy, double m[6]);

// Warp: every destination pixel centre (x + 0.5, y + 0.5) maps through m to
// a source position, sampled with `filter` (Nearest, Bilinear or Bicubic;
// others read as Bilinear). A position outside the source takes `fill`
// (channels bytes; null = zeros). For a strong reduction, resize the source
// first: the samplers here do not prefilter.
void warp_affine_hwc_u8(const uint8_t* src, int src_w, int src_h, int channels,
                        uint8_t* dst, int dst_w, int dst_h,
                        const double m[6],
                        Filter filter = Filter::Bilinear,
                        const uint8_t* fill = nullptr,
                        int src_stride_bytes = 0,
                        int dst_stride_bytes = 0);

} // namespace broimage
