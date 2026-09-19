// The explicitly-typed geometric kernels and the alpha-aware ones.
//
// Ported from the QuickJS-era src/js/image_bindings.cpp (img_resizeU8,
// img_resizeF32, img_resizeChwF32, img_letterboxU8, img_padU8, img_cropU8,
// img_centerCropU8, img_flipHorizontalU8, img_flipVerticalU8, img_rotate90U8,
// img_premultiplyAlpha, img_unpremultiplyAlpha, img_resizeRgba8Alpha,
// img_letterboxRgba8Alpha).
//
// The bronze port replaced the *U8 names with untyped `resize` / `crop` /
// `pad` / ... and, for pad, swapped the `pad: [r,g,b,a]` array for four
// padR/padG/padB/padA scalars. Those stay; these restore the old names with
// the old option contract, so both generations of caller work.

#include "host_image_internal.h"

#include <broimage/alpha.h>
#include <broimage/geometric.h>

#include <cstdint>
#include <span>
#include <string>

namespace broimage::api {

namespace {

const float kOpaqueBlack[4] = {0.0f, 0.0f, 0.0f, 255.0f};
const float kTransparent[4] = {0.0f, 0.0f, 0.0f, 0.0f};

bool filterOf(Value opts, broimage::Filter* out, const char* who) {
    std::string s;
    if (!getPropStr(opts, "filter", &s, "bilinear")) {
        ev::throwTypeError(std::string(who) + ": filter must be a string");
        return false;
    }
    if (s.empty() || s == "bilinear") { *out = broimage::Filter::Bilinear; return true; }
    if (s == "nearest")  { *out = broimage::Filter::Nearest;  return true; }
    if (s == "bicubic")  { *out = broimage::Filter::Bicubic;  return true; }
    if (s == "lanczos3") { *out = broimage::Filter::Lanczos3; return true; }
    if (s == "area")     { *out = broimage::Filter::Area;     return true; }
    ev::throwTypeError(std::string(who) +
                       ": filter must be nearest|bilinear|bicubic|lanczos3|area");
    return false;
}

// { x, y, w, h } — the content rect a letterbox placed inside its destination.
Value makeRect(int x, int y, int w, int h) {
    ObjectBuilder r;
    r.set("x", static_cast<double>(x));
    r.set("y", static_cast<double>(y));
    r.set("w", static_cast<double>(w));
    r.set("h", static_cast<double>(h));
    return r.build();
}

bool pairU8(Value a, Value b, const char* who, TypedArrayView* dst,
            TypedArrayView* src, size_t bpe) {
    if (!unpackTypedArray(a, "dst", dst)) return false;
    if (!unpackTypedArray(b, "src", src)) return false;
    if (dst->bytesPerElement != bpe || src->bytesPerElement != bpe) {
        ev::throwTypeError(std::string(who) + (bpe == 1
            ? ": dst/src must be Uint8Array" : ": dst/src must be Float32Array"));
        return false;
    }
    return true;
}

// ─── resize ───────────────────────────────────────────────────────────────

// resizeU8(dst, src, {srcW, srcH, dstW, dstH, channels, srcStride, dstStride,
//                     filter})
Value imageResizeU8(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("resizeU8(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "resizeU8", &dst, &src, 1)) return ev::undefined();

    int32_t sw = 0, sh = 0, dw = 0, dh = 0, ch = 4, ss = 0, ds = 0;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "dstW", &dw, 0) || !getPropI32(args[2], "dstH", &dh, 0) ||
        !getPropI32(args[2], "channels", &ch, 4) ||
        !getPropI32(args[2], "srcStride", &ss, 0) || !getPropI32(args[2], "dstStride", &ds, 0))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || ch <= 0)
        return ev::throwRangeError("resizeU8: dims/channels must be positive");

    broimage::Filter f;
    if (!filterOf(args[2], &f, "resizeU8")) return ev::undefined();
    broimage::resize_hwc_u8(src.data, sw, sh, ch, dst.data, dw, dh, f, ss, ds);
    return ev::undefined();
}

// resizeF32(dst, src, {srcW, srcH, dstW, dstH, channels, filter}) — HWC float.
Value imageResizeF32(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("resizeF32(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "resizeF32", &dst, &src, 4)) return ev::undefined();

    int32_t sw = 0, sh = 0, dw = 0, dh = 0, ch = 1;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "dstW", &dw, 0) || !getPropI32(args[2], "dstH", &dh, 0) ||
        !getPropI32(args[2], "channels", &ch, 1))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || ch <= 0)
        return ev::throwRangeError("resizeF32: dims/channels must be positive");

    broimage::Filter f;
    if (!filterOf(args[2], &f, "resizeF32")) return ev::undefined();
    broimage::resize_hwc_f32(reinterpret_cast<const float*>(src.data), sw, sh, ch,
                             reinterpret_cast<float*>(dst.data), dw, dh, f);
    return ev::undefined();
}

// resizeChwF32(dst, src, {srcW, srcH, dstW, dstH, channels, filter}) — planar.
Value imageResizeChwF32(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("resizeChwF32(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "resizeChwF32", &dst, &src, 4)) return ev::undefined();

    int32_t sw = 0, sh = 0, dw = 0, dh = 0, ch = 1;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "dstW", &dw, 0) || !getPropI32(args[2], "dstH", &dh, 0) ||
        !getPropI32(args[2], "channels", &ch, 1))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || ch <= 0)
        return ev::throwRangeError("resizeChwF32: dims/channels must be positive");

    broimage::Filter f;
    if (!filterOf(args[2], &f, "resizeChwF32")) return ev::undefined();
    broimage::resize_chw_f32(reinterpret_cast<const float*>(src.data), sw, sh, ch,
                             reinterpret_cast<float*>(dst.data), dw, dh, f);
    return ev::undefined();
}

// ─── letterbox / pad / crop ───────────────────────────────────────────────

// letterboxU8(dst, src, {srcW, srcH, dstW, dstH, channels, pad, filter})
//   -> { x, y, w, h } of the content rect inside dst.
Value imageLetterboxU8(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("letterboxU8(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "letterboxU8", &dst, &src, 1)) return ev::undefined();

    int32_t sw = 0, sh = 0, dw = 0, dh = 0, ch = 4;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "dstW", &dw, 0) || !getPropI32(args[2], "dstH", &dh, 0) ||
        !getPropI32(args[2], "channels", &ch, 4))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || ch <= 0)
        return ev::throwRangeError("letterboxU8: dims/channels must be positive");

    float pad[4];
    if (!getPropFloats(args[2], "pad", pad, 4, kOpaqueBlack))
        return ev::throwTypeError("letterboxU8: pad must be [r, g, b, a]");
    broimage::Filter f;
    if (!filterOf(args[2], &f, "letterboxU8")) return ev::undefined();

    int ox = 0, oy = 0, ow = 0, oh = 0;
    broimage::letterbox_hwc_u8(src.data, sw, sh, ch, dst.data, dw, dh,
                               static_cast<uint8_t>(pad[0]), static_cast<uint8_t>(pad[1]),
                               static_cast<uint8_t>(pad[2]), static_cast<uint8_t>(pad[3]),
                               f, &ox, &oy, &ow, &oh);
    return makeRect(ox, oy, ow, oh);
}

// padU8(dst, src, {srcW, srcH, dstW, dstH, channels, offX, offY, pad,
//                  srcStride, dstStride})
Value imagePadU8(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("padU8(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "padU8", &dst, &src, 1)) return ev::undefined();

    int32_t sw = 0, sh = 0, dw = 0, dh = 0, ch = 4, ox = 0, oy = 0, ss = 0, ds = 0;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "dstW", &dw, 0) || !getPropI32(args[2], "dstH", &dh, 0) ||
        !getPropI32(args[2], "channels", &ch, 4) ||
        !getPropI32(args[2], "offX", &ox, 0) || !getPropI32(args[2], "offY", &oy, 0) ||
        !getPropI32(args[2], "srcStride", &ss, 0) || !getPropI32(args[2], "dstStride", &ds, 0))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || ch <= 0)
        return ev::throwRangeError("padU8: dims/channels must be positive");

    float pad[4];
    if (!getPropFloats(args[2], "pad", pad, 4, kOpaqueBlack))
        return ev::throwTypeError("padU8: pad must be [r, g, b, a]");
    broimage::pad_hwc_u8(src.data, sw, sh, ch, dst.data, dw, dh, ox, oy,
                         static_cast<uint8_t>(pad[0]), static_cast<uint8_t>(pad[1]),
                         static_cast<uint8_t>(pad[2]), static_cast<uint8_t>(pad[3]),
                         ss, ds);
    return ev::undefined();
}

// cropU8(dst, src, {srcW, srcH, channels, x, y, w, h, srcStride, dstStride})
Value imageCropU8(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("cropU8(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "cropU8", &dst, &src, 1)) return ev::undefined();

    int32_t sw = 0, sh = 0, ch = 4, x = 0, y = 0, w = 0, h = 0, ss = 0, ds = 0;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "channels", &ch, 4) ||
        !getPropI32(args[2], "x", &x, 0) || !getPropI32(args[2], "y", &y, 0) ||
        !getPropI32(args[2], "w", &w, 0) || !getPropI32(args[2], "h", &h, 0) ||
        !getPropI32(args[2], "srcStride", &ss, 0) || !getPropI32(args[2], "dstStride", &ds, 0))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || ch <= 0 || w <= 0 || h <= 0)
        return ev::throwRangeError("cropU8: dims/channels/rect must be positive");

    broimage::crop_hwc_u8(src.data, sw, sh, ch, dst.data, x, y, w, h, ss, ds);
    return ev::undefined();
}

// centerCropU8(dst, src, {srcW, srcH, channels, cropW, cropH, srcStride,
//                         dstStride})
Value imageCenterCropU8(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("centerCropU8(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "centerCropU8", &dst, &src, 1)) return ev::undefined();

    int32_t sw = 0, sh = 0, ch = 4, cw = 0, chh = 0, ss = 0, ds = 0;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "channels", &ch, 4) ||
        !getPropI32(args[2], "cropW", &cw, 0) || !getPropI32(args[2], "cropH", &chh, 0) ||
        !getPropI32(args[2], "srcStride", &ss, 0) || !getPropI32(args[2], "dstStride", &ds, 0))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || ch <= 0 || cw <= 0 || chh <= 0)
        return ev::throwRangeError("centerCropU8: dims/channels/crop must be positive");

    broimage::center_crop_hwc_u8(src.data, sw, sh, ch, dst.data, cw, chh, ss, ds);
    return ev::undefined();
}

// ─── flip / rotate ────────────────────────────────────────────────────────

// flipHorizontalU8 / flipVerticalU8 (dst, src, {w, h, channels, srcStride,
//                                               dstStride})
Value flipU8(std::span<const Value> args, bool horizontal) {
    const char* who = horizontal ? "flipHorizontalU8" : "flipVerticalU8";
    if (args.size() < 3) return ev::throwTypeError(std::string(who) + "(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], who, &dst, &src, 1)) return ev::undefined();

    int32_t w = 0, h = 0, ch = 4, ss = 0, ds = 0;
    if (!getPropI32(args[2], "w", &w, 0) || !getPropI32(args[2], "h", &h, 0) ||
        !getPropI32(args[2], "channels", &ch, 4) ||
        !getPropI32(args[2], "srcStride", &ss, 0) || !getPropI32(args[2], "dstStride", &ds, 0))
        return ev::undefined();
    if (w <= 0 || h <= 0 || ch <= 0)
        return ev::throwRangeError(std::string(who) + ": dims/channels must be positive");

    if (horizontal) {
        broimage::flip_horizontal_hwc_u8(src.data, dst.data, w, h, ch, ss, ds);
    } else {
        broimage::flip_vertical_hwc_u8(src.data, dst.data, w, h, ch, ss, ds);
    }
    return ev::undefined();
}

// rotate90U8(dst, src, {srcW, srcH, channels, turns, srcStride, dstStride})
//   `turns` counts 90-degree CCW turns; dst dims swap on an odd count.
Value imageRotate90U8(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("rotate90U8(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "rotate90U8", &dst, &src, 1)) return ev::undefined();

    int32_t sw = 0, sh = 0, ch = 4, turns = 1, ss = 0, ds = 0;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "channels", &ch, 4) || !getPropI32(args[2], "turns", &turns, 1) ||
        !getPropI32(args[2], "srcStride", &ss, 0) || !getPropI32(args[2], "dstStride", &ds, 0))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || ch <= 0)
        return ev::throwRangeError("rotate90U8: dims/channels must be positive");

    broimage::rotate_90_hwc_u8(src.data, sw, sh, ch, dst.data, turns, ss, ds);
    return ev::undefined();
}

// ─── alpha ────────────────────────────────────────────────────────────────

Value premultiply(std::span<const Value> args, bool forward) {
    const char* who = forward ? "premultiplyAlpha" : "unpremultiplyAlpha";
    if (args.size() < 2) return ev::throwTypeError(std::string(who) + "(dst, src)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], who, &dst, &src, 1)) return ev::undefined();

    const int n = static_cast<int>(src.byteLength / 4);
    if (dst.byteLength < static_cast<size_t>(n) * 4)
        return ev::throwRangeError(std::string(who) + ": dst too small");
    if (forward) {
        broimage::premultiply_alpha_rgba8(src.data, dst.data, n);
    } else {
        broimage::unpremultiply_alpha_rgba8(src.data, dst.data, n);
    }
    return ev::undefined();
}

// resizeRgba8Alpha(dst, src, {srcW, srcH, dstW, dstH, filter}) — alpha-aware
// resize: premultiply, resize, unpremultiply, so transparent regions do not
// bleed their arbitrary RGB into the visible edge.
Value imageResizeRgba8Alpha(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("resizeRgba8Alpha(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "resizeRgba8Alpha", &dst, &src, 1)) return ev::undefined();

    int32_t sw = 0, sh = 0, dw = 0, dh = 0;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "dstW", &dw, 0) || !getPropI32(args[2], "dstH", &dh, 0))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
        return ev::throwRangeError("resizeRgba8Alpha: dims must be positive");

    broimage::Filter f;
    if (!filterOf(args[2], &f, "resizeRgba8Alpha")) return ev::undefined();
    broimage::resize_rgba8_alpha(src.data, sw, sh, dst.data, dw, dh, f);
    return ev::undefined();
}

// letterboxRgba8Alpha(dst, src, {srcW, srcH, dstW, dstH, pad, filter})
//   -> { x, y, w, h }. The pad is written as straight RGBA.
Value imageLetterboxRgba8Alpha(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("letterboxRgba8Alpha(dst, src, params)");
    TypedArrayView dst, src;
    if (!pairU8(args[0], args[1], "letterboxRgba8Alpha", &dst, &src, 1)) return ev::undefined();

    int32_t sw = 0, sh = 0, dw = 0, dh = 0;
    if (!getPropI32(args[2], "srcW", &sw, 0) || !getPropI32(args[2], "srcH", &sh, 0) ||
        !getPropI32(args[2], "dstW", &dw, 0) || !getPropI32(args[2], "dstH", &dh, 0))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0)
        return ev::throwRangeError("letterboxRgba8Alpha: dims must be positive");

    float pad[4];
    if (!getPropFloats(args[2], "pad", pad, 4, kTransparent))
        return ev::throwTypeError("letterboxRgba8Alpha: pad must be [r, g, b, a]");
    broimage::Filter f;
    if (!filterOf(args[2], &f, "letterboxRgba8Alpha")) return ev::undefined();

    int ox = 0, oy = 0, ow = 0, oh = 0;
    broimage::letterbox_rgba8_alpha(src.data, sw, sh, dst.data, dw, dh,
                                    static_cast<uint8_t>(pad[0]), static_cast<uint8_t>(pad[1]),
                                    static_cast<uint8_t>(pad[2]), static_cast<uint8_t>(pad[3]),
                                    f, &ox, &oy, &ow, &oh);
    return makeRect(ox, oy, ow, oh);
}

} // namespace

void installGeometryOnto(Value imageObj) {
    ObjectBuilder b(imageObj);

    b.def("resizeU8", 3, imageResizeU8);
    b.def("resizeF32", 3, imageResizeF32);
    b.def("resizeChwF32", 3, imageResizeChwF32);
    b.def("letterboxU8", 3, imageLetterboxU8);
    b.def("padU8", 3, imagePadU8);
    b.def("cropU8", 3, imageCropU8);
    b.def("centerCropU8", 3, imageCenterCropU8);
    b.def("flipHorizontalU8", 3, [](Value, std::span<const Value> a) {
        return flipU8(a, true);
    });
    b.def("flipVerticalU8", 3, [](Value, std::span<const Value> a) {
        return flipU8(a, false);
    });
    b.def("rotate90U8", 3, imageRotate90U8);

    b.def("premultiplyAlpha", 2, [](Value, std::span<const Value> a) {
        return premultiply(a, true);
    });
    b.def("unpremultiplyAlpha", 2, [](Value, std::span<const Value> a) {
        return premultiply(a, false);
    });
    b.def("resizeRgba8Alpha", 3, imageResizeRgba8Alpha);
    b.def("letterboxRgba8Alpha", 3, imageLetterboxRgba8Alpha);
}

} // namespace broimage::api
