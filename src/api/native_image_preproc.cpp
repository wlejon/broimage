// Layout shuffles, color-matrix apply, NCHW normalize (+ the mean/std presets),
// multi-channel stencil and the tiling accumulators.
//
// Ported from the QuickJS-era src/js/image_bindings.cpp (img_hwcToChw,
// img_chwToHwc, img_srgbToLinearU8ToF32, img_linearF32ToSrgbU8,
// img_applyColorMatrix3x3, img_applyColorMatrix3x4, img_u8NhwcToF32Nchw,
// img_nhwcToNchwF32, img_nchwToNhwcF32, img_f32NchwToU8Nhwc,
// img_normalizeNchw, the `presets` object, img_stencilHwc, img_featherWindow,
// img_accumulateTile, img_normalizeAccumulator).
//
// The bronze port kept lowercase-keyed `u8ToF32` / `nhwcToNchw` / `nchwToNhwc`
// and a positional `normalize`; those stay. These restore the old names, whose
// option objects use the uppercase N/C/H/W keys, so callers written against
// either generation work.

#include "host_image_internal.h"

#include <broimage/color.h>
#include <broimage/kernels.h>
#include <broimage/normalize.h>
#include <broimage/preproc.h>
#include <broimage/presets.h>
#include <broimage/tiling.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace broimage::api {

namespace {

bool pairOf(Value a, Value b, const char* who, TypedArrayView* dst,
            TypedArrayView* src, size_t dstBpe, size_t srcBpe) {
    if (!unpackTypedArray(a, "dst", dst)) return false;
    if (!unpackTypedArray(b, "src", src)) return false;
    if (dst->bytesPerElement != dstBpe) {
        ev::throwTypeError(std::string(who) + (dstBpe == 1
            ? ": dst must be a Uint8Array" : ": dst must be a Float32Array"));
        return false;
    }
    if (src->bytesPerElement != srcBpe) {
        ev::throwTypeError(std::string(who) + (srcBpe == 1
            ? ": src must be a Uint8Array" : ": src must be a Float32Array"));
        return false;
    }
    return true;
}

// ─── HWC <-> CHW (single image) ───────────────────────────────────────────

// hwcToChw / chwToHwc (dst, src, {width, height, channels}) — float32.
Value shuffleHwcChw(std::span<const Value> args, bool toChw) {
    const char* who = toChw ? "hwcToChw" : "chwToHwc";
    if (args.size() < 3)
        return ev::throwTypeError(std::string(who) + "(dst, src, {width, height, channels})");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], who, &dst, &src, 4, 4)) return ev::undefined();

    int32_t w = 0, h = 0, c = 0;
    if (!getPropI32(args[2], "width", &w, 0) || !getPropI32(args[2], "height", &h, 0) ||
        !getPropI32(args[2], "channels", &c, 0))
        return ev::undefined();
    if (w <= 0 || h <= 0 || c <= 0)
        return ev::throwRangeError(std::string(who) + ": dims/channels must be positive");
    const size_t need = static_cast<size_t>(w) * h * c * sizeof(float);
    if (src.byteLength < need || dst.byteLength < need)
        return ev::throwRangeError(std::string(who) + ": buffers too small for w*h*channels");

    if (!resolveViews({&dst, &src})) return ev::undefined();
    if (toChw) {
        broimage::hwc_to_chw_f32(reinterpret_cast<const float*>(src.data),
                                 reinterpret_cast<float*>(dst.data), w, h, c);
    } else {
        broimage::chw_to_hwc_f32(reinterpret_cast<const float*>(src.data),
                                 reinterpret_cast<float*>(dst.data), w, h, c);
    }
    return ev::undefined();
}

// ─── sRGB <-> linear across the u8 / f32 boundary ─────────────────────────

// srgbToLinearU8ToF32(dst, src) — one float out per input byte.
Value imageSrgbToLinearU8ToF32(Value, std::span<const Value> args) {
    if (args.size() < 2) return ev::throwTypeError("srgbToLinearU8ToF32(dst, src)");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], "srgbToLinearU8ToF32", &dst, &src, 4, 1))
        return ev::undefined();
    const int n = static_cast<int>(src.byteLength);
    if (dst.byteLength < static_cast<size_t>(n) * sizeof(float))
        return ev::throwRangeError("srgbToLinearU8ToF32: dst too small");
    if (!resolveViews({&dst, &src})) return ev::undefined();
    broimage::srgb_to_linear_u8_to_f32(src.data, reinterpret_cast<float*>(dst.data), n);
    return ev::undefined();
}

// linearF32ToSrgbU8(dst, src) — the inverse.
Value imageLinearF32ToSrgbU8(Value, std::span<const Value> args) {
    if (args.size() < 2) return ev::throwTypeError("linearF32ToSrgbU8(dst, src)");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], "linearF32ToSrgbU8", &dst, &src, 1, 4))
        return ev::undefined();
    const int n = static_cast<int>(src.byteLength / sizeof(float));
    if (dst.byteLength < static_cast<size_t>(n))
        return ev::throwRangeError("linearF32ToSrgbU8: dst too small");
    if (!resolveViews({&dst, &src})) return ev::undefined();
    broimage::linear_f32_to_srgb_u8(reinterpret_cast<const float*>(src.data), dst.data, n);
    return ev::undefined();
}

// ─── Color matrix ─────────────────────────────────────────────────────────

// applyColorMatrix3x3 / 3x4 (dst, src, {channels, matrix}) — float32 HWC; the
// 3x4 form's last column is a per-channel offset. Alpha passes through.
Value colorMatrix(std::span<const Value> args, bool is3x4) {
    const char* who = is3x4 ? "applyColorMatrix3x4" : "applyColorMatrix3x3";
    const int count = is3x4 ? 12 : 9;
    if (args.size() < 3)
        return ev::throwTypeError(std::string(who) + "(dst, src, {channels, matrix})");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], who, &dst, &src, 4, 4)) return ev::undefined();

    int32_t ch = 3;
    if (!getPropI32(args[2], "channels", &ch, 3)) return ev::undefined();
    if (ch != 3 && ch != 4)
        return ev::throwRangeError(std::string(who) + ": channels must be 3 or 4");

    float m[12];
    if (!getPropFloats(args[2], "matrix", m, count, nullptr)) {
        return ev::throwTypeError(std::string(who) + ": matrix must be " +
                                  std::to_string(count) + " numbers");
    }
    const int n = static_cast<int>(src.byteLength / sizeof(float) / static_cast<size_t>(ch));
    if (dst.byteLength < static_cast<size_t>(n) * ch * sizeof(float))
        return ev::throwRangeError(std::string(who) + ": dst too small");

    if (!resolveViews({&dst, &src})) return ev::undefined();
    if (is3x4) {
        broimage::apply_color_matrix_3x4_f32(reinterpret_cast<const float*>(src.data),
                                             reinterpret_cast<float*>(dst.data), n, ch, m);
    } else {
        broimage::apply_color_matrix_3x3_f32(reinterpret_cast<const float*>(src.data),
                                             reinterpret_cast<float*>(dst.data), n, ch, m);
    }
    return ev::undefined();
}

// ─── Batched layout / scale-bias ──────────────────────────────────────────

// u8NhwcToF32Nchw(dst, src, {N, H, W, C, scale, bias}) — the model-input
// shuffle. `scale` defaults to 1, NOT 1/255: the old contract made the caller
// say so, and the port's u8ToF32 (which defaults to 1/255) still does.
Value imageU8NhwcToF32Nchw(Value, std::span<const Value> args) {
    if (args.size() < 3)
        return ev::throwTypeError("u8NhwcToF32Nchw(dst, src, {N, H, W, C, scale, bias})");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], "u8NhwcToF32Nchw", &dst, &src, 4, 1)) return ev::undefined();

    int32_t n = 1, h = 0, w = 0, c = 0;
    double scale = 1.0, bias = 0.0;
    if (!getPropI32(args[2], "N", &n, 1) || !getPropI32(args[2], "H", &h, 0) ||
        !getPropI32(args[2], "W", &w, 0) || !getPropI32(args[2], "C", &c, 0) ||
        !getPropF64(args[2], "scale", &scale, 1.0) ||
        !getPropF64(args[2], "bias", &bias, 0.0))
        return ev::undefined();
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0)
        return ev::throwRangeError("u8NhwcToF32Nchw: N/H/W/C must be positive");
    const size_t need = static_cast<size_t>(n) * c * h * w;
    if (dst.byteLength < need * sizeof(float))
        return ev::throwRangeError("u8NhwcToF32Nchw: dst too small");
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::u8_nhwc_to_f32_nchw(src.data, n, h, w, c, static_cast<float>(scale),
                                  static_cast<float>(bias),
                                  reinterpret_cast<float*>(dst.data));
    return ev::undefined();
}

// f32NchwToU8Nhwc(dst, src, {N, C, H, W, scale, bias}) — the inverse, for
// turning a model output back into something drawable.
Value imageF32NchwToU8Nhwc(Value, std::span<const Value> args) {
    if (args.size() < 3)
        return ev::throwTypeError("f32NchwToU8Nhwc(dst, src, {N, C, H, W, scale, bias})");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], "f32NchwToU8Nhwc", &dst, &src, 1, 4)) return ev::undefined();

    int32_t n = 1, c = 0, h = 0, w = 0;
    double scale = 1.0, bias = 0.0;
    if (!getPropI32(args[2], "N", &n, 1) || !getPropI32(args[2], "C", &c, 0) ||
        !getPropI32(args[2], "H", &h, 0) || !getPropI32(args[2], "W", &w, 0) ||
        !getPropF64(args[2], "scale", &scale, 1.0) ||
        !getPropF64(args[2], "bias", &bias, 0.0))
        return ev::undefined();
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
        return ev::throwRangeError("f32NchwToU8Nhwc: N/C/H/W must be positive");
    if (dst.byteLength < static_cast<size_t>(n) * h * w * c)
        return ev::throwRangeError("f32NchwToU8Nhwc: dst too small");
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::f32_nchw_to_u8_nhwc(reinterpret_cast<const float*>(src.data), n, c, h, w,
                                  static_cast<float>(scale), static_cast<float>(bias),
                                  dst.data);
    return ev::undefined();
}

// nhwcToNchwF32 / nchwToNhwcF32 (dst, src, {N, H, W, C}) — plain float
// reshuffles, uppercase keys.
Value shuffleBatched(std::span<const Value> args, bool toNchw) {
    const char* who = toNchw ? "nhwcToNchwF32" : "nchwToNhwcF32";
    if (args.size() < 3)
        return ev::throwTypeError(std::string(who) + "(dst, src, {N, H, W, C})");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], who, &dst, &src, 4, 4)) return ev::undefined();

    int32_t n = 1, h = 0, w = 0, c = 0;
    if (!getPropI32(args[2], "N", &n, 1) || !getPropI32(args[2], "H", &h, 0) ||
        !getPropI32(args[2], "W", &w, 0) || !getPropI32(args[2], "C", &c, 0))
        return ev::undefined();
    if (n <= 0 || h <= 0 || w <= 0 || c <= 0)
        return ev::throwRangeError(std::string(who) + ": N/H/W/C must be positive");
    const size_t need = static_cast<size_t>(n) * c * h * w * sizeof(float);
    if (dst.byteLength < need)
        return ev::throwRangeError(std::string(who) + ": dst too small");

    if (!resolveViews({&dst, &src})) return ev::undefined();
    if (toNchw) {
        broimage::nhwc_to_nchw_f32(reinterpret_cast<const float*>(src.data), n, h, w, c,
                                   reinterpret_cast<float*>(dst.data));
    } else {
        broimage::nchw_to_nhwc_f32(reinterpret_cast<const float*>(src.data), n, c, h, w,
                                   reinterpret_cast<float*>(dst.data));
    }
    return ev::undefined();
}

// ─── Normalize ────────────────────────────────────────────────────────────

// normalizeNchw(dst, src, {N, C, H, W, mean, std}) — per-channel
// (x - mean) / std, with mean/std carried in the option object rather than as
// positional arguments. bro.image.presets supplies the usual triples.
Value imageNormalizeNchw(Value, std::span<const Value> args) {
    if (args.size() < 3)
        return ev::throwTypeError("normalizeNchw(dst, src, {N, C, H, W, mean, std})");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], "normalizeNchw", &dst, &src, 4, 4)) return ev::undefined();

    int32_t n = 1, c = 0, h = 0, w = 0;
    if (!getPropI32(args[2], "N", &n, 1) || !getPropI32(args[2], "C", &c, 0) ||
        !getPropI32(args[2], "H", &h, 0) || !getPropI32(args[2], "W", &w, 0))
        return ev::undefined();
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
        return ev::throwRangeError("normalizeNchw: N/C/H/W must be positive");

    std::vector<float> mean(static_cast<size_t>(c)), stdv(static_cast<size_t>(c));
    if (!getPropFloats(args[2], "mean", mean.data(), c, nullptr))
        return ev::throwTypeError("normalizeNchw: mean must be an array of length C");
    if (!getPropFloats(args[2], "std", stdv.data(), c, nullptr))
        return ev::throwTypeError("normalizeNchw: std must be an array of length C");

    const size_t need = static_cast<size_t>(n) * c * h * w * sizeof(float);
    if (src.byteLength < need || dst.byteLength < need)
        return ev::throwRangeError("normalizeNchw: buffers too small for N*C*H*W");
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::image_normalize_nchw_f32(reinterpret_cast<const float*>(src.data),
                                       mean.data(), stdv.data(), n, c, h, w,
                                       reinterpret_cast<float*>(dst.data));
    return ev::undefined();
}

// ─── Multi-channel stencil ────────────────────────────────────────────────

// stencilHwc(dst, src, {data, w, h}, {srcW, srcH, channels, divisor, bias,
//                                     edge}) — the same kernel per channel,
// which is what a blur / sharpen on an interleaved RGB(A) image needs.
Value imageStencilHwc(Value, std::span<const Value> args) {
    if (args.size() < 4)
        return ev::throwTypeError("stencilHwc(dst, src, kernel, params)");
    TypedArrayView dst, src;
    if (!pairOf(args[0], args[1], "stencilHwc", &dst, &src, 4, 4)) return ev::undefined();

    int32_t kw = 0, kh = 0;
    if (!getPropI32(args[2], "w", &kw, 0) || !getPropI32(args[2], "h", &kh, 0))
        return ev::undefined();
    TypedArrayView kdata;
    if (!unpackTypedArray(ev::getProperty(args[2], "data"), "kernel.data", &kdata))
        return ev::undefined();
    if (kdata.bytesPerElement != 4)
        return ev::throwTypeError("stencilHwc: kernel.data must be a Float32Array");
    if (kw <= 0 || kh <= 0 || (kw & 1) == 0 || (kh & 1) == 0)
        return ev::throwRangeError("stencilHwc: kernel w/h must be positive and odd");
    if (kdata.byteLength < static_cast<size_t>(kw) * kh * sizeof(float))
        return ev::throwRangeError("stencilHwc: kernel.data too small");

    int32_t sw = 0, sh = 0, ch = 1;
    if (!getPropI32(args[3], "srcW", &sw, 0) || !getPropI32(args[3], "srcH", &sh, 0) ||
        !getPropI32(args[3], "channels", &ch, 1))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || ch <= 0)
        return ev::throwRangeError("stencilHwc: srcW/srcH/channels must be positive");
    const size_t need = static_cast<size_t>(sw) * sh * ch * sizeof(float);
    if (src.byteLength < need || dst.byteLength < need)
        return ev::throwRangeError("stencilHwc: buffers too small for srcW*srcH*channels");

    std::string edge;
    if (!getPropStr(args[3], "edge", &edge)) return ev::undefined();
    broimage::StencilEdge be = broimage::StencilEdge::Clamp;
    if (edge == "wrap") be = broimage::StencilEdge::Wrap;
    else if (edge == "zero") be = broimage::StencilEdge::Zero;
    else if (!edge.empty() && edge != "clamp")
        return ev::throwTypeError("stencilHwc: edge must be clamp|wrap|zero");

    double divisor = 1.0, bias = 0.0;
    if (!getPropF64(args[3], "divisor", &divisor, 1.0) ||
        !getPropF64(args[3], "bias", &bias, 0.0))
        return ev::undefined();

    if (!resolveViews({&dst, &src, &kdata})) return ev::undefined();
    broimage::stencil_hwc_f32(reinterpret_cast<const float*>(src.data),
                              reinterpret_cast<float*>(dst.data), sw, sh, ch,
                              reinterpret_cast<const float*>(kdata.data), kw, kh,
                              static_cast<float>(divisor), static_cast<float>(bias), be);
    return ev::undefined();
}

// ─── Tiling ───────────────────────────────────────────────────────────────

// featherWindow(win, {tw, th, ovL, ovR, ovT, ovB}) — the raised-cosine ramp
// that makes overlapping tiles blend seamlessly.
Value imageFeatherWindow(Value, std::span<const Value> args) {
    if (args.size() < 2)
        return ev::throwTypeError("featherWindow(win, {tw, th, ovL, ovR, ovT, ovB})");
    TypedArrayView win;
    if (!unpackTypedArray(args[0], "win", &win)) return ev::undefined();
    if (win.bytesPerElement != 4)
        return ev::throwTypeError("featherWindow: win must be a Float32Array");

    int32_t tw = 0, th = 0, ovL = 0, ovR = 0, ovT = 0, ovB = 0;
    if (!getPropI32(args[1], "tw", &tw, 0) || !getPropI32(args[1], "th", &th, 0) ||
        !getPropI32(args[1], "ovL", &ovL, 0) || !getPropI32(args[1], "ovR", &ovR, 0) ||
        !getPropI32(args[1], "ovT", &ovT, 0) || !getPropI32(args[1], "ovB", &ovB, 0))
        return ev::undefined();
    if (tw <= 0 || th <= 0)
        return ev::throwRangeError("featherWindow: tw/th must be positive");
    if (win.byteLength < static_cast<size_t>(tw) * th * sizeof(float))
        return ev::throwRangeError("featherWindow: win too small for tw*th");
    if (!resolveViews({&win})) return ev::undefined();

    broimage::feather_window_f32(reinterpret_cast<float*>(win.data), tw, th,
                                 ovL, ovR, ovT, ovB);
    return ev::undefined();
}

// accumulateTile(acc, wacc, tile, window, {fullW, fullH, channels, tw, th,
//                                          dstX, dstY})
Value imageAccumulateTile(Value, std::span<const Value> args) {
    if (args.size() < 5)
        return ev::throwTypeError("accumulateTile(acc, wacc, tile, window, params)");
    TypedArrayView acc, wacc, tile, window;
    if (!unpackTypedArray(args[0], "acc", &acc)) return ev::undefined();
    if (!unpackTypedArray(args[1], "wacc", &wacc)) return ev::undefined();
    if (!unpackTypedArray(args[2], "tile", &tile)) return ev::undefined();
    if (!unpackTypedArray(args[3], "window", &window)) return ev::undefined();
    if (acc.bytesPerElement != 4 || wacc.bytesPerElement != 4 ||
        tile.bytesPerElement != 4 || window.bytesPerElement != 4)
        return ev::throwTypeError("accumulateTile: all buffers must be Float32Array");

    int32_t fw = 0, fh = 0, ch = 1, tw = 0, th = 0, dx = 0, dy = 0;
    if (!getPropI32(args[4], "fullW", &fw, 0) || !getPropI32(args[4], "fullH", &fh, 0) ||
        !getPropI32(args[4], "channels", &ch, 1) ||
        !getPropI32(args[4], "tw", &tw, 0) || !getPropI32(args[4], "th", &th, 0) ||
        !getPropI32(args[4], "dstX", &dx, 0) || !getPropI32(args[4], "dstY", &dy, 0))
        return ev::undefined();
    if (fw <= 0 || fh <= 0 || ch <= 0 || tw <= 0 || th <= 0)
        return ev::throwRangeError("accumulateTile: dims/channels must be positive");
    if (acc.byteLength < static_cast<size_t>(fw) * fh * ch * sizeof(float) ||
        wacc.byteLength < static_cast<size_t>(fw) * fh * sizeof(float))
        return ev::throwRangeError("accumulateTile: acc/wacc too small for fullW*fullH");
    if (tile.byteLength < static_cast<size_t>(tw) * th * ch * sizeof(float) ||
        window.byteLength < static_cast<size_t>(tw) * th * sizeof(float))
        return ev::throwRangeError("accumulateTile: tile/window too small for tw*th");
    if (!resolveViews({&acc, &wacc, &tile, &window})) return ev::undefined();

    broimage::accumulate_tile_f32(reinterpret_cast<float*>(acc.data),
                                  reinterpret_cast<float*>(wacc.data), fw, fh, ch,
                                  reinterpret_cast<const float*>(tile.data), tw, th,
                                  dx, dy, reinterpret_cast<const float*>(window.data));
    return ev::undefined();
}

// normalizeAccumulator(acc, wacc, {nPixels, channels, eps}) — resolve the
// weighted sums into the final blended map, in place.
Value imageNormalizeAccumulator(Value, std::span<const Value> args) {
    if (args.size() < 3)
        return ev::throwTypeError("normalizeAccumulator(acc, wacc, {nPixels, channels, eps})");
    TypedArrayView acc, wacc;
    if (!unpackTypedArray(args[0], "acc", &acc)) return ev::undefined();
    if (!unpackTypedArray(args[1], "wacc", &wacc)) return ev::undefined();
    if (acc.bytesPerElement != 4 || wacc.bytesPerElement != 4)
        return ev::throwTypeError("normalizeAccumulator: acc/wacc must be Float32Array");

    int32_t np = 0, ch = 1;
    double eps = 1e-6;
    if (!getPropI32(args[2], "nPixels", &np, 0) ||
        !getPropI32(args[2], "channels", &ch, 1) ||
        !getPropF64(args[2], "eps", &eps, 1e-6))
        return ev::undefined();
    if (np <= 0 || ch <= 0)
        return ev::throwRangeError("normalizeAccumulator: nPixels/channels must be positive");
    if (acc.byteLength < static_cast<size_t>(np) * ch * sizeof(float) ||
        wacc.byteLength < static_cast<size_t>(np) * sizeof(float))
        return ev::throwRangeError("normalizeAccumulator: buffers too small for nPixels");
    if (!resolveViews({&acc, &wacc})) return ev::undefined();

    broimage::normalize_accumulator_f32(reinterpret_cast<float*>(acc.data),
                                        reinterpret_cast<const float*>(wacc.data),
                                        np, ch, static_cast<float>(eps));
    return ev::undefined();
}

// { mean: [r, g, b], std: [r, g, b] } as plain JS arrays.
Value makePreset(const float* mean, const float* std_) {
    ObjectBuilder e;
    ev::Persistent m(hostArrayOf(3, [mean](size_t i) {
        return ev::fromDouble(static_cast<double>(mean[i]));
    }));
    e.set("mean", m.get());
    ev::Persistent s(hostArrayOf(3, [std_](size_t i) {
        return ev::fromDouble(static_cast<double>(std_[i]));
    }));
    e.set("std", s.get());
    return e.build();
}

} // namespace

void installPreprocOnto(Value imageObj) {
    ObjectBuilder b(imageObj);

    b.def("hwcToChw", 3, [](Value, std::span<const Value> a) { return shuffleHwcChw(a, true); });
    b.def("chwToHwc", 3, [](Value, std::span<const Value> a) { return shuffleHwcChw(a, false); });

    b.def("srgbToLinearU8ToF32", 2, imageSrgbToLinearU8ToF32);
    b.def("linearF32ToSrgbU8", 2, imageLinearF32ToSrgbU8);
    b.def("applyColorMatrix3x3", 3, [](Value, std::span<const Value> a) {
        return colorMatrix(a, false);
    });
    b.def("applyColorMatrix3x4", 3, [](Value, std::span<const Value> a) {
        return colorMatrix(a, true);
    });

    b.def("u8NhwcToF32Nchw", 3, imageU8NhwcToF32Nchw);
    b.def("f32NchwToU8Nhwc", 3, imageF32NchwToU8Nhwc);
    b.def("nhwcToNchwF32", 3, [](Value, std::span<const Value> a) {
        return shuffleBatched(a, true);
    });
    b.def("nchwToNhwcF32", 3, [](Value, std::span<const Value> a) {
        return shuffleBatched(a, false);
    });

    b.def("normalizeNchw", 3, imageNormalizeNchw);
    {
        ObjectBuilder presets;
        ev::Persistent clip(makePreset(broimage::CLIP_MEAN, broimage::CLIP_STD));
        presets.set("clip", clip.get());
        ev::Persistent inet(makePreset(broimage::IMAGENET_MEAN, broimage::IMAGENET_STD));
        presets.set("imagenet", inet.get());
        ev::Persistent sam(makePreset(broimage::SAM_MEAN, broimage::SAM_STD));
        presets.set("sam", sam.get());
        b.set("presets", presets.build());
    }

    b.def("stencilHwc", 4, imageStencilHwc);

    b.def("featherWindow", 2, imageFeatherWindow);
    b.def("accumulateTile", 5, imageAccumulateTile);
    b.def("normalizeAccumulator", 3, imageNormalizeAccumulator);
}

} // namespace broimage::api
