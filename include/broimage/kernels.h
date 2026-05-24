#pragma once

#include "broimage/geometric.h"

#include <cstdint>
#include <vector>

namespace broimage {

// ----- gradient --------------------------------------------------------------
//
// Build a 1D RGBA8 LUT from color stops. Each stop is `{t, r, g, b, a}` with
// `t` in [0, 1] and 8-bit channel values (a defaults to 255 if Stop::a is
// negative — call sites that don't set a get fully opaque).
struct GradientStop {
    float t;
    float r, g, b, a;
};

// `out` is resized to `n * 4` bytes. n must be >= 2.
void gradient(const GradientStop* stops, int stop_count, int n,
              std::vector<uint8_t>& out);

// ----- lookup ----------------------------------------------------------------
//
// `dst[i*4..i*4+4] = lut[ clamp_or_wrap( (src[i] - lo) / (hi - lo) * (lut_n-1) ) ]`.
// `src` is a flat scalar field; `dst` is RGBA8 with `n_pixels * 4` bytes.
enum class LookupEdge { Clamp, Wrap };

void lookup_f32(const float* src, int n_pixels,
                const uint8_t* lut, int lut_n,
                uint8_t* dst,
                float lo, float hi,
                LookupEdge edge = LookupEdge::Clamp);

// ----- reduce ----------------------------------------------------------------
//
// Sub-sampling stride visits every Nth element (cheap minmax for huge buffers
// driving smoothed estimators). stride must be >= 1.
struct MinMax { float min; float max; };

MinMax reduce_minmax_f32(const float* src, int n, int stride = 1);
double reduce_sum_f32(const float* src, int n, int stride = 1);
double reduce_mean_f32(const float* src, int n, int stride = 1);

// `bins` must be >= 1, `hi` > `lo`. Output histogram is written into
// `out_counts` (caller-sized to `bins`).
void reduce_histogram_f32(const float* src, int n,
                          int bins, float lo, float hi,
                          uint32_t* out_counts,
                          int stride = 1);

// ----- map (element-wise unary, float32) ------------------------------------
//
// `dst[i] = f(src[i])`. dst and src may alias.

void map_affine_f32(const float* src, float* dst, int n,
                    float a, float b);
void map_affine_clamp_f32(const float* src, float* dst, int n,
                          float a, float b, float clamp_lo, float clamp_hi);
void map_abs_f32(const float* src, float* dst, int n);
void map_log_f32(const float* src, float* dst, int n);
void map_sqrt_f32(const float* src, float* dst, int n);
void map_exp_f32(const float* src, float* dst, int n);
void map_pow_f32(const float* src, float* dst, int n, float exponent);

// ----- combine (element-wise binary, float32) -------------------------------
//
// `dst[i] = f(a[i], b[i])`. dst may alias a or b.

void combine_add_f32(const float* a, const float* b, float* dst, int n);
void combine_sub_f32(const float* a, const float* b, float* dst, int n);
void combine_mul_f32(const float* a, const float* b, float* dst, int n);
void combine_min_f32(const float* a, const float* b, float* dst, int n);
void combine_max_f32(const float* a, const float* b, float* dst, int n);
void combine_lerp_f32(const float* a, const float* b, float* dst, int n, float t);
void combine_wsum_f32(const float* a, const float* b, float* dst, int n,
                      float wa, float wb);

// ----- stencil (2D convolution, float32) ------------------------------------
//
// `dst[y*w+x] = (sum over (ky,kx) of kernel[ky*kw+kx] * src[edge(y+ky-hkh, x+kx-hkw)])
//             * (1 / divisor) + bias`
//
// kernel width and height must be odd. `edge` controls out-of-range pixel
// handling: Clamp clamps to edge, Wrap wraps to the opposite side, Zero
// treats out-of-range as 0.
enum class StencilEdge { Clamp, Wrap, Zero };

void stencil_f32(const float* src, float* dst,
                 int src_w, int src_h,
                 const float* kernel, int kw, int kh,
                 float divisor = 1.0f, float bias = 0.0f,
                 StencilEdge edge = StencilEdge::Clamp);

// Multi-channel HWC float convolution: the same kernel is applied to each
// channel independently (per-channel `(sum k_i * src_i) / divisor + bias`).
// This is what you actually want for blur / sharpen / edge filters on an
// RGB(A) image — calling stencil_f32 once per channel works but forces the
// caller to deinterleave first.
void stencil_hwc_f32(const float* src, float* dst,
                     int src_w, int src_h, int channels,
                     const float* kernel, int kw, int kh,
                     float divisor = 1.0f, float bias = 0.0f,
                     StencilEdge edge = StencilEdge::Clamp);

// ----- resample --------------------------------------------------------------
//
// HWC float32 resize. This is the same kernel as `geometric::resize_hwc_f32`
// but exposed under the bro.image kernel-verb naming for the
// "kernels-on-buffers" surface that brokit's JS API uses.
//
// Reuses the geometric Filter enum so callers don't have to translate between
// "this is the resize op" and "this is the kernel verb spelling of the resize
// op". Bicubic is supported here as well.
void resample_f32(const float* src, int src_w, int src_h,
                  float*       dst, int dst_w, int dst_h,
                  int channels,
                  Filter filter = Filter::Bilinear);

} // namespace broimage
