#include "host_image_internal.h"

#include <broimage/geometric.h>
#include <broimage/color.h>
#include <broimage/normalize.h>
#include <broimage/preproc.h>
#include <broimage/kernels.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace broimage::api {

namespace {

// A (dst, src, count) converter: `count` from the positional argument, then
// both views checked against what the kernel reads and writes — `srcPer`
// elements per unit of src and `dstPer` of dst, float32 elements where the
// flag says so (which also requires a Float32Array), bytes otherwise.
bool countedPair(const TypedArrayView& dst, const TypedArrayView& src, Value countV,
                 const char* who, bool dstF32, uint64_t dstPer, bool srcF32, uint64_t srcPer,
                 int32_t* count) {
    if (!countArg(countV, who, "count", count)) return false;
    auto check = [&](const TypedArrayView& v, bool f32, uint64_t per) {
        if (f32 && !requireFloat32(v, who)) return false;
        return requireBytes(v, static_cast<uint64_t>(*count) * per * (f32 ? 4 : 1), who);
    };
    return check(src, srcF32, srcPer) && check(dst, dstF32, dstPer);
}

// ---------------------------------------------------------------------------
// 1. Kernel Operations
// ---------------------------------------------------------------------------

Value imageGradient(Value, std::span<const Value> args) {
    if (args.empty() || !ev::isObject(args[0]))
        return ev::throwTypeError("gradient(stops, n=256): stops must be an array");

    ArgReader reader(args);
    int32_t n = reader.getInt(1, 256);
    if (n < 2) return ev::throwRangeError("gradient: n must be >= 2");

    // args[0] is re-read from the rooted span and each stop is rooted: a
    // "length" read may allocate (property-key interning) and move them.
    Value lenVal = ev::getProperty(args[0], "length");
    if (!ev::isNumber(lenVal)) return ev::throwTypeError("gradient: stops must be an array");
    uint32_t stopCount = static_cast<uint32_t>(ev::toDouble(lenVal));
    if (stopCount < 2) return ev::throwTypeError("gradient: need at least 2 stops");

    std::vector<broimage::GradientStop> stops(stopCount);
    for (uint32_t i = 0; i < stopCount; i++) {
        ev::Persistent stop(ev::getElement(args[0], i));
        if (!ev::isObject(stop.get()))
            return ev::throwTypeError("gradient: stop must be an array");

        Value lv = ev::getProperty(stop.get(), "length");
        uint32_t slen = ev::isNumber(lv) ? static_cast<uint32_t>(ev::toDouble(lv)) : 0;
        if (slen < 4)
            return ev::throwTypeError("gradient: stop must be [t, r, g, b, a?]");

        // getElement may allocate too, so each read re-reads the root.
        double t = ev::toDouble(ev::getElement(stop.get(), 0));
        double r = ev::toDouble(ev::getElement(stop.get(), 1));
        double g = ev::toDouble(ev::getElement(stop.get(), 2));
        double b = ev::toDouble(ev::getElement(stop.get(), 3));
        double a = (slen >= 5) ? ev::toDouble(ev::getElement(stop.get(), 4)) : -1.0;

        stops[i].t = static_cast<float>(t);
        stops[i].r = static_cast<float>(r);
        stops[i].g = static_cast<float>(g);
        stops[i].b = static_cast<float>(b);
        stops[i].a = static_cast<float>(a);
    }

    std::vector<uint8_t> outBytes;
    broimage::gradient(stops.data(), static_cast<int>(stopCount), n, outBytes);

    Value ab = ev::createArrayBuffer(
        std::span<const uint8_t>(outBytes.data(), outBytes.size()));
    return ev::createTypedArrayView(ev::elements::Uint8, ab, 0, static_cast<uint32_t>(outBytes.size()));
}

Value imageAlloc(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("alloc(w, h, channels, dtype='float32')");
    ArgReader reader(args);
    int32_t w = reader.getInt(0, 0);
    int32_t h = reader.getInt(1, 0);
    int32_t channels = reader.getInt(2, 0);
    if (w <= 0 || h <= 0 || channels <= 0)
        return ev::throwRangeError("alloc: dimensions must be positive");

    std::string dtype = reader.getString(3, "float32");
    size_t count = static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(channels);
    if (count > static_cast<size_t>(INT32_MAX))
        return ev::throwRangeError("alloc: w*h*channels is too large");
    ElementKind kind;
    if (dtype == "float32")      { kind = ev::elements::Float32;      }
    else if (dtype == "float64") { kind = ev::elements::Float64;      }
    else if (dtype == "uint8")   { kind = ev::elements::Uint8;        }
    else if (dtype == "uint8c")  { kind = ev::elements::Uint8Clamped; }
    else if (dtype == "int16")   { kind = ev::elements::Int16;        }
    else if (dtype == "int32")   { kind = ev::elements::Int32;        }
    else if (dtype == "uint16")  { kind = ev::elements::Uint16;       }
    else if (dtype == "uint32")  { kind = ev::elements::Uint32;       }
    else return ev::throwTypeError("alloc: unknown dtype '" + dtype + "'");

    return ev::createTypedArray(kind, static_cast<uint32_t>(count));
}

Value imageLookup(Value, std::span<const Value> args) {
    if (args.size() < 4) return ev::throwTypeError("lookup(dst, src, lut, {lo, hi, edge?})");

    TypedArrayView dst, src, lut;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    if (!unpackTypedArray(args[2], "lut", &lut)) return ev::undefined();

    if (dst.bytesPerElement != 1) return ev::throwTypeError("lookup: dst must be Uint8Array/Uint8ClampedArray");
    if (lut.bytesPerElement != 1) return ev::throwTypeError("lookup: lut must be Uint8Array (RGBA8)");
    if (lut.byteLength % 4 != 0) return ev::throwTypeError("lookup: lut length must be a multiple of 4");
    size_t lutN = lut.byteLength / 4;
    if (lutN < 2) return ev::throwRangeError("lookup: lut must have >= 2 entries");

    size_t n = src.byteLength / src.bytesPerElement;
    if (dst.byteLength < n * 4)
        return ev::throwRangeError("lookup: dst too small");

    double lo = 0, hi = 1;
    if (!getPropF64(args[3], "lo", &lo, 0)) return ev::undefined();
    if (!getPropF64(args[3], "hi", &hi, 1)) return ev::undefined();
    std::string edge;
    if (!getPropStr(args[3], "edge", &edge)) return ev::undefined();
    const broimage::LookupEdge be =
        (edge == "wrap") ? broimage::LookupEdge::Wrap : broimage::LookupEdge::Clamp;

    ScalarKind kind{};
    if (!probeScalarKind(args[1], &kind)) return ev::undefined();

    if (!resolveViews({&dst, &src, &lut})) return ev::undefined();
    if (kind.isFloat && src.bytesPerElement == 4) {
        broimage::lookup_f32(
            reinterpret_cast<const float*>(src.data), static_cast<int>(n),
            lut.data, static_cast<int>(lutN),
            dst.data,
            static_cast<float>(lo), static_cast<float>(hi), be);
        return ev::undefined();
    }

    const float loF = static_cast<float>(lo);
    const float invSpan = (hi > lo) ? (1.0f / static_cast<float>(hi - lo)) : 0.0f;
    const float idxMax = static_cast<float>(lutN - 1);
    const bool wrap = (be == broimage::LookupEdge::Wrap);

    const uint8_t* sp = src.data;
    uint8_t* dp = dst.data;
    for (size_t i = 0; i < n; i++) {
        float v = readScalar(sp + i * src.bytesPerElement, src.bytesPerElement, kind.isFloat, kind.isSigned);
        float t = (v - loF) * invSpan;
        float fi = t * idxMax;
        int idx;
        if (wrap) {
            float lf = static_cast<float>(lutN);
            fi = std::fmod(fi, lf);
            if (fi < 0) fi += lf;
            idx = static_cast<int>(fi);
            if (idx >= static_cast<int>(lutN)) idx = static_cast<int>(lutN) - 1;
        } else {
            if (fi < 0) fi = 0;
            if (fi > idxMax) fi = idxMax;
            idx = static_cast<int>(fi);
        }
        const uint8_t* lp = lut.data + idx * 4;
        dp[i * 4 + 0] = lp[0];
        dp[i * 4 + 1] = lp[1];
        dp[i * 4 + 2] = lp[2];
        dp[i * 4 + 3] = lp[3];
    }
    return ev::undefined();
}

Value imageReduce(Value, std::span<const Value> args) {
    if (args.size() < 2) return ev::throwTypeError("reduce(src, op, params?)");

    TypedArrayView src;
    if (!unpackTypedArray(args[0], "src", &src)) return ev::undefined();
    if (!ev::isString(args[1])) return ev::throwTypeError("reduce: op must be a string");
    std::string op = ev::toUtf8(args[1]);

    ScalarKind kind{};
    if (!probeScalarKind(args[0], &kind)) return ev::undefined();

    size_t n = src.byteLength / src.bytesPerElement;
    if (n == 0) return ev::throwRangeError("reduce: src is empty");

    int32_t stride = 1;
    if (args.size() >= 3 && ev::isObject(args[2])) {
        if (!getPropI32(args[2], "stride", &stride, 1)) return ev::undefined();
        if (stride < 1) return ev::throwRangeError("reduce: stride must be >= 1");
    }
    const size_t step = static_cast<size_t>(stride);
    const bool f32Path = (kind.isFloat && src.bytesPerElement == 4);
    auto readAt = [&](size_t i) -> float {
        return readScalar(src.data + i * src.bytesPerElement, src.bytesPerElement, kind.isFloat, kind.isSigned);
    };

    // Each op resolves src after its last option read and before it scans.
    if (op == "minmax") {
        float mn, mx;
        if (!resolveViews({&src})) return ev::undefined();
        if (f32Path) {
            broimage::MinMax mm = broimage::reduce_minmax_f32(
                reinterpret_cast<const float*>(src.data),
                static_cast<int>(n), stride);
            mn = mm.min; mx = mm.max;
        } else {
            mn = readAt(0); mx = mn;
            for (size_t i = step; i < n; i += step) {
                float v = readAt(i);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
        ObjectBuilder obj;
        obj.set("min", static_cast<double>(mn));
        obj.set("max", static_cast<double>(mx));
        return obj.build();
    }
    if (op == "sum" || op == "mean") {
        double r;
        if (!resolveViews({&src})) return ev::undefined();
        if (f32Path) {
            r = (op == "sum")
                ? broimage::reduce_sum_f32(reinterpret_cast<const float*>(src.data), static_cast<int>(n), stride)
                : broimage::reduce_mean_f32(reinterpret_cast<const float*>(src.data), static_cast<int>(n), stride);
        } else {
            double sum = 0;
            size_t count = 0;
            for (size_t i = 0; i < n; i += step) { sum += readAt(i); count++; }
            r = (op == "mean") ? (sum / static_cast<double>(count)) : sum;
        }
        return ev::fromDouble(r);
    }
    if (op == "histogram") {
        if (args.size() < 3 || !ev::isObject(args[2]))
            return ev::throwTypeError("reduce histogram requires {bins, lo, hi}");
        int32_t bins = 256;
        double lo = 0, hi = 1;
        if (!getPropI32(args[2], "bins", &bins, 256)) return ev::undefined();
        if (!getPropF64(args[2], "lo", &lo, 0)) return ev::undefined();
        if (!getPropF64(args[2], "hi", &hi, 1)) return ev::undefined();
        if (bins < 1) return ev::throwRangeError("histogram: bins must be >= 1");
        if (hi <= lo) return ev::throwRangeError("histogram: hi must be > lo");

        std::vector<uint32_t> counts(static_cast<size_t>(bins), 0);
        if (!resolveViews({&src})) return ev::undefined();
        if (f32Path) {
            broimage::reduce_histogram_f32(
                reinterpret_cast<const float*>(src.data), static_cast<int>(n),
                bins, static_cast<float>(lo), static_cast<float>(hi), counts.data(), stride);
        } else {
            const float loF = static_cast<float>(lo);
            const float invSpan = 1.0f / static_cast<float>(hi - lo);
            for (size_t i = 0; i < n; i += step) {
                float v = readAt(i);
                float t = (v - loF) * invSpan;
                int idx = static_cast<int>(t * static_cast<float>(bins));
                if (idx < 0 || idx >= bins) continue;
                counts[idx]++;
            }
        }
        Value ab = ev::createArrayBuffer(
            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(counts.data()), counts.size() * sizeof(uint32_t)));
        return ev::createTypedArrayView(ev::elements::Uint32, ab, 0, static_cast<uint32_t>(counts.size()));
    }
    return ev::throwTypeError("reduce: unknown op '" + op + "'");
}

Value imageMap(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("map(dst, src, opSpec)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    if (dst.bytesPerElement != 4 || src.bytesPerElement != 4)
        return ev::throwTypeError("map: dst and src must be Float32Array");
    int n = static_cast<int>(src.byteLength / 4);
    if (dst.byteLength < static_cast<size_t>(n) * 4)
        return ev::throwRangeError("map: dst too small");

    std::string op;
    if (!getPropStr(args[2], "op", &op)) return ev::undefined();
    if (op.empty()) return ev::throwTypeError("map: opSpec.op required");

    // Every op-specific option is read before src/dst resolve: those reads
    // allocate, and the resolved pointers must not outlive one.
    double a = 1, b = 0, e = 1;
    bool clamp = false;
    float clo = 0, chi = 0;
    if (op == "affine") {
        if (!getPropF64(args[2], "a", &a, 1)) return ev::undefined();
        if (!getPropF64(args[2], "b", &b, 0)) return ev::undefined();
        ev::Persistent cv(ev::getProperty(args[2], "clamp"));
        if (ev::isObject(cv.get())) {
            clamp = true;
            clo = static_cast<float>(ev::toDouble(ev::getElement(cv.get(), 0)));
            chi = static_cast<float>(ev::toDouble(ev::getElement(cv.get(), 1)));
        }
    } else if (op == "pow") {
        if (!getPropF64(args[2], "exp", &e, 1)) return ev::undefined();
    } else if (op != "abs" && op != "log" && op != "sqrt" && op != "exp") {
        return ev::throwTypeError("map: unknown op '" + op + "'");
    }

    if (!resolveViews({&dst, &src})) return ev::undefined();
    const float* sp = reinterpret_cast<const float*>(src.data);
    float* dp = reinterpret_cast<float*>(dst.data);

    if (op == "affine") {
        if (clamp) {
            broimage::map_affine_clamp_f32(sp, dp, n, static_cast<float>(a), static_cast<float>(b), clo, chi);
        } else {
            broimage::map_affine_f32(sp, dp, n, static_cast<float>(a), static_cast<float>(b));
        }
    }
    else if (op == "abs")  broimage::map_abs_f32(sp, dp, n);
    else if (op == "log")  broimage::map_log_f32(sp, dp, n);
    else if (op == "sqrt") broimage::map_sqrt_f32(sp, dp, n);
    else if (op == "exp")  broimage::map_exp_f32(sp, dp, n);
    else                   broimage::map_pow_f32(sp, dp, n, static_cast<float>(e));
    return ev::undefined();
}

Value imageCombine(Value, std::span<const Value> args) {
    if (args.size() < 4) return ev::throwTypeError("combine(dst, a, b, opSpec)");
    TypedArrayView dst, va, vb;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "a",   &va))  return ev::undefined();
    if (!unpackTypedArray(args[2], "b",   &vb))  return ev::undefined();
    if (dst.bytesPerElement != 4 || va.bytesPerElement != 4 || vb.bytesPerElement != 4)
        return ev::throwTypeError("combine: all buffers must be Float32Array");
    int n = static_cast<int>(va.byteLength / 4);
    if (vb.byteLength / 4 != static_cast<size_t>(n))
        return ev::throwRangeError("combine: a and b must have equal length");
    if (dst.byteLength < static_cast<size_t>(n) * 4)
        return ev::throwRangeError("combine: dst too small");

    std::string op;
    if (!getPropStr(args[3], "op", &op)) return ev::undefined();
    if (op.empty()) return ev::throwTypeError("combine: opSpec.op required");

    // Op-specific options first; the views resolve after the last read.
    double t = 0, wa = 1, wb = 1;
    if (op == "lerp") {
        if (!getPropF64(args[3], "t", &t, 0)) return ev::undefined();
    } else if (op == "wsum") {
        if (!getPropF64(args[3], "wa", &wa, 1)) return ev::undefined();
        if (!getPropF64(args[3], "wb", &wb, 1)) return ev::undefined();
    } else if (op != "add" && op != "sub" && op != "mul" && op != "min" && op != "max") {
        return ev::throwTypeError("combine: unknown op '" + op + "'");
    }

    if (!resolveViews({&dst, &va, &vb})) return ev::undefined();
    const float* ap = reinterpret_cast<const float*>(va.data);
    const float* bp = reinterpret_cast<const float*>(vb.data);
    float* dp = reinterpret_cast<float*>(dst.data);

    if      (op == "add")  broimage::combine_add_f32(ap, bp, dp, n);
    else if (op == "sub")  broimage::combine_sub_f32(ap, bp, dp, n);
    else if (op == "mul")  broimage::combine_mul_f32(ap, bp, dp, n);
    else if (op == "min")  broimage::combine_min_f32(ap, bp, dp, n);
    else if (op == "max")  broimage::combine_max_f32(ap, bp, dp, n);
    else if (op == "lerp") broimage::combine_lerp_f32(ap, bp, dp, n, static_cast<float>(t));
    else                   broimage::combine_wsum_f32(ap, bp, dp, n, static_cast<float>(wa), static_cast<float>(wb));
    return ev::undefined();
}

Value imageStencil(Value, std::span<const Value> args) {
    if (args.size() < 4) return ev::throwTypeError("stencil(dst, src, kernel, params)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    if (dst.bytesPerElement != 4 || src.bytesPerElement != 4)
        return ev::throwTypeError("stencil: dst and src must be Float32Array");

    // Rooted: the w/h reads below may allocate and move it.
    ev::Persistent kdataV(ev::getProperty(args[2], "data"));
    int32_t kw = 0, kh = 0;
    if (!getPropI32(args[2], "w", &kw, 0)) return ev::undefined();
    if (!getPropI32(args[2], "h", &kh, 0)) return ev::undefined();
    TypedArrayView kdata;
    if (!unpackTypedArray(kdataV.get(), "kernel.data", &kdata)) return ev::undefined();
    if (kdata.bytesPerElement != 4) return ev::throwTypeError("stencil: kernel.data must be Float32Array");
    if (kw <= 0 || kh <= 0) return ev::throwRangeError("stencil: kernel w/h must be positive");
    if ((kw & 1) == 0 || (kh & 1) == 0)
        return ev::throwRangeError("stencil: kernel w/h must be odd");
    if (kdata.byteLength < static_cast<size_t>(kw) * static_cast<size_t>(kh) * 4)
        return ev::throwRangeError("stencil: kernel.data too small for w*h");

    int32_t srcW = 0, srcH = 0;
    if (!getPropI32(args[3], "srcW", &srcW, 0)) return ev::undefined();
    if (!getPropI32(args[3], "srcH", &srcH, 0)) return ev::undefined();
    if (srcW <= 0 || srcH <= 0)
        return ev::throwRangeError("stencil: srcW/srcH required and positive");
    if (src.byteLength < static_cast<size_t>(srcW) * static_cast<size_t>(srcH) * 4)
        return ev::throwRangeError("stencil: src too small for srcW*srcH");
    if (dst.byteLength < static_cast<size_t>(srcW) * static_cast<size_t>(srcH) * 4)
        return ev::throwRangeError("stencil: dst too small for srcW*srcH");

    std::string edge;
    if (!getPropStr(args[3], "edge", &edge)) return ev::undefined();
    broimage::StencilEdge be = broimage::StencilEdge::Clamp;
    if (edge == "wrap") be = broimage::StencilEdge::Wrap;
    else if (edge == "zero") be = broimage::StencilEdge::Zero;
    else if (!edge.empty() && edge != "clamp")
        return ev::throwTypeError("stencil: edge must be 'clamp'|'wrap'|'zero'");

    double divisor = 1, bias = 0;
    if (!getPropF64(args[3], "divisor", &divisor, 1)) return ev::undefined();
    if (!getPropF64(args[3], "bias",    &bias,    0)) return ev::undefined();

    if (!resolveViews({&dst, &src, &kdata})) return ev::undefined();
    broimage::stencil_f32(
        reinterpret_cast<const float*>(src.data),
        reinterpret_cast<float*>(dst.data),
        srcW, srcH,
        reinterpret_cast<const float*>(kdata.data), kw, kh,
        static_cast<float>(divisor), static_cast<float>(bias), be);
    return ev::undefined();
}

Value imageResample(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("resample(dst, src, params)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    if (dst.bytesPerElement != 4 || src.bytesPerElement != 4)
        return ev::throwTypeError("resample: dst and src must be Float32Array");

    int32_t srcW, srcH, dstW, dstH, channels;
    if (!getPropI32(args[2], "srcW", &srcW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "srcH", &srcH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstW", &dstW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstH", &dstH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "channels", &channels, 1)) return ev::undefined();
    if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0 || channels <= 0)
        return ev::throwRangeError("resample: all dims/channels must be positive");
    size_t needSrc = static_cast<size_t>(srcW) * srcH * channels * 4;
    size_t needDst = static_cast<size_t>(dstW) * dstH * channels * 4;
    if (src.byteLength < needSrc) return ev::throwRangeError("resample: src too small");
    if (dst.byteLength < needDst) return ev::throwRangeError("resample: dst too small");

    std::string filter;
    if (!getPropStr(args[2], "filter", &filter)) return ev::undefined();
    if (filter.empty()) filter = "bilinear";

    broimage::Filter f;
    if      (filter == "nearest")  f = broimage::Filter::Nearest;
    else if (filter == "bilinear") f = broimage::Filter::Bilinear;
    else return ev::throwTypeError("resample: filter must be 'nearest'|'bilinear'");

    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::resample_f32(
        reinterpret_cast<const float*>(src.data), srcW, srcH,
        reinterpret_cast<float*>(dst.data),       dstW, dstH,
        channels, f);
    return ev::undefined();
}

// ---------------------------------------------------------------------------
// 2. Geometric Transforms
// ---------------------------------------------------------------------------

inline broimage::Filter parseFilter(const std::string& filterStr) {
    if (filterStr == "nearest")  return broimage::Filter::Nearest;
    if (filterStr == "bicubic")  return broimage::Filter::Bicubic;
    if (filterStr == "lanczos3") return broimage::Filter::Lanczos3;
    if (filterStr == "area")     return broimage::Filter::Area;
    return broimage::Filter::Bilinear;
}

Value imageResize(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("resize(dst, src, options)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();

    int32_t srcW = 0, srcH = 0, dstW = 0, dstH = 0, channels = 4;
    int32_t srcStride = 0, dstStride = 0;
    std::string filterStr = "bilinear";

    if (!getPropI32(args[2], "srcW", &srcW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "srcH", &srcH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstW", &dstW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstH", &dstH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "channels", &channels, 4)) return ev::undefined();
    if (!getPropI32(args[2], "srcStride", &srcStride, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstStride", &dstStride, 0)) return ev::undefined();
    if (!getPropStr(args[2], "filter", &filterStr, "bilinear")) return ev::undefined();

    if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0 || channels <= 0)
        return ev::throwRangeError("resize: all dimensions and channels must be positive");

    broimage::Filter filter = parseFilter(filterStr);
    const bool f32 = src.kind == ev::elements::Float32 && dst.kind == ev::elements::Float32;
    if (f32) {
        if (!requireImage(src, srcW, srcH, channels, 0, 4, "resize") ||
            !requireImage(dst, dstW, dstH, channels, 0, 4, "resize"))
            return ev::undefined();
    } else if (!requireImage(src, srcW, srcH, channels, srcStride, 1, "resize") ||
               !requireImage(dst, dstW, dstH, channels, dstStride, 1, "resize")) {
        return ev::undefined();
    }
    if (!resolveViews({&dst, &src})) return ev::undefined();

    if (f32) {
        broimage::resize_hwc_f32(reinterpret_cast<const float*>(src.data), srcW, srcH, channels,
                                 reinterpret_cast<float*>(dst.data), dstW, dstH, filter);
    } else {
        broimage::resize_hwc_u8(src.data, srcW, srcH, channels,
                                dst.data, dstW, dstH, filter,
                                srcStride, dstStride);
    }
    return ev::undefined();
}

Value imageCrop(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("crop(dst, src, options)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();

    int32_t srcW = 0, srcH = 0, channels = 4, x = 0, y = 0, w = 0, h = 0;
    int32_t srcStride = 0, dstStride = 0;
    if (!getPropI32(args[2], "srcW", &srcW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "srcH", &srcH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "channels", &channels, 4)) return ev::undefined();
    if (!getPropI32(args[2], "x", &x, 0)) return ev::undefined();
    if (!getPropI32(args[2], "y", &y, 0)) return ev::undefined();
    if (!getPropI32(args[2], "w", &w, 0)) return ev::undefined();
    if (!getPropI32(args[2], "h", &h, 0)) return ev::undefined();
    if (!getPropI32(args[2], "srcStride", &srcStride, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstStride", &dstStride, 0)) return ev::undefined();

    if (srcW <= 0 || srcH <= 0 || w <= 0 || h <= 0 || channels <= 0)
        return ev::throwRangeError("crop: invalid dimensions");
    if (!requireImage(src, srcW, srcH, channels, srcStride, 1, "crop") ||
        !requireImage(dst, w, h, channels, dstStride, 1, "crop"))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::crop_hwc_u8(src.data, srcW, srcH, channels, dst.data, x, y, w, h, srcStride, dstStride);
    return ev::undefined();
}

Value imageCenterCrop(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("centerCrop(dst, src, options)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();

    int32_t srcW = 0, srcH = 0, channels = 4, cropW = 0, cropH = 0;
    int32_t srcStride = 0, dstStride = 0;
    if (!getPropI32(args[2], "srcW", &srcW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "srcH", &srcH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "channels", &channels, 4)) return ev::undefined();
    if (!getPropI32(args[2], "cropW", &cropW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "cropH", &cropH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "srcStride", &srcStride, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstStride", &dstStride, 0)) return ev::undefined();

    if (srcW <= 0 || srcH <= 0 || cropW <= 0 || cropH <= 0 || channels <= 0)
        return ev::throwRangeError("centerCrop: invalid dimensions");
    if (!requireImage(src, srcW, srcH, channels, srcStride, 1, "centerCrop") ||
        !requireImage(dst, cropW, cropH, channels, dstStride, 1, "centerCrop"))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::center_crop_hwc_u8(src.data, srcW, srcH, channels, dst.data, cropW, cropH, srcStride, dstStride);
    return ev::undefined();
}

Value imageFlipHorizontal(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("flipHorizontal(dst, src, options)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();

    int32_t w = 0, h = 0, channels = 4, srcStride = 0, dstStride = 0;
    if (!getPropI32(args[2], "w", &w, 0)) return ev::undefined();
    if (!getPropI32(args[2], "h", &h, 0)) return ev::undefined();
    if (!getPropI32(args[2], "channels", &channels, 4)) return ev::undefined();
    if (!getPropI32(args[2], "srcStride", &srcStride, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstStride", &dstStride, 0)) return ev::undefined();

    if (w <= 0 || h <= 0 || channels <= 0) return ev::throwRangeError("flipHorizontal: invalid dimensions");
    if (!requireImage(src, w, h, channels, srcStride, 1, "flipHorizontal") ||
        !requireImage(dst, w, h, channels, dstStride, 1, "flipHorizontal"))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::flip_horizontal_hwc_u8(src.data, dst.data, w, h, channels, srcStride, dstStride);
    return ev::undefined();
}

Value imageFlipVertical(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("flipVertical(dst, src, options)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();

    int32_t w = 0, h = 0, channels = 4, srcStride = 0, dstStride = 0;
    if (!getPropI32(args[2], "w", &w, 0)) return ev::undefined();
    if (!getPropI32(args[2], "h", &h, 0)) return ev::undefined();
    if (!getPropI32(args[2], "channels", &channels, 4)) return ev::undefined();
    if (!getPropI32(args[2], "srcStride", &srcStride, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstStride", &dstStride, 0)) return ev::undefined();

    if (w <= 0 || h <= 0 || channels <= 0) return ev::throwRangeError("flipVertical: invalid dimensions");
    if (!requireImage(src, w, h, channels, srcStride, 1, "flipVertical") ||
        !requireImage(dst, w, h, channels, dstStride, 1, "flipVertical"))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::flip_vertical_hwc_u8(src.data, dst.data, w, h, channels, srcStride, dstStride);
    return ev::undefined();
}

Value imageRotate90(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("rotate90(dst, src, options)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();

    int32_t srcW = 0, srcH = 0, channels = 4, turns = 1, srcStride = 0, dstStride = 0;
    if (!getPropI32(args[2], "srcW", &srcW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "srcH", &srcH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "channels", &channels, 4)) return ev::undefined();
    if (!getPropI32(args[2], "turns", &turns, 1)) return ev::undefined();
    if (!getPropI32(args[2], "srcStride", &srcStride, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstStride", &dstStride, 0)) return ev::undefined();

    if (srcW <= 0 || srcH <= 0 || channels <= 0) return ev::throwRangeError("rotate90: invalid dimensions");
    const bool odd = (((turns % 4) + 4) % 4) % 2 == 1;
    if (!requireImage(src, srcW, srcH, channels, srcStride, 1, "rotate90") ||
        !requireImage(dst, odd ? srcH : srcW, odd ? srcW : srcH, channels, dstStride, 1, "rotate90"))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::rotate_90_hwc_u8(src.data, srcW, srcH, channels, dst.data, turns, srcStride, dstStride);
    return ev::undefined();
}

Value imagePad(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("pad(dst, src, options)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();

    int32_t srcW = 0, srcH = 0, dstW = 0, dstH = 0, channels = 4;
    int32_t offX = 0, offY = 0, r = 0, g = 0, b = 0, a = 255;
    int32_t srcStride = 0, dstStride = 0;
    if (!getPropI32(args[2], "srcW", &srcW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "srcH", &srcH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstW", &dstW, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstH", &dstH, 0)) return ev::undefined();
    if (!getPropI32(args[2], "channels", &channels, 4)) return ev::undefined();
    if (!getPropI32(args[2], "offX", &offX, 0)) return ev::undefined();
    if (!getPropI32(args[2], "offY", &offY, 0)) return ev::undefined();
    if (!getPropI32(args[2], "padR", &r, 0)) return ev::undefined();
    if (!getPropI32(args[2], "padG", &g, 0)) return ev::undefined();
    if (!getPropI32(args[2], "padB", &b, 0)) return ev::undefined();
    if (!getPropI32(args[2], "padA", &a, 255)) return ev::undefined();
    if (!getPropI32(args[2], "srcStride", &srcStride, 0)) return ev::undefined();
    if (!getPropI32(args[2], "dstStride", &dstStride, 0)) return ev::undefined();

    if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0 || channels <= 0)
        return ev::throwRangeError("pad: invalid dimensions");
    if (!requireImage(src, srcW, srcH, channels, srcStride, 1, "pad") ||
        !requireImage(dst, dstW, dstH, channels, dstStride, 1, "pad"))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::pad_hwc_u8(src.data, srcW, srcH, channels, dst.data, dstW, dstH,
                         offX, offY, static_cast<uint8_t>(r), static_cast<uint8_t>(g),
                         static_cast<uint8_t>(b), static_cast<uint8_t>(a),
                         srcStride, dstStride);
    return ev::undefined();
}

// ---------------------------------------------------------------------------
// 3. Color Conversions
// ---------------------------------------------------------------------------

Value imageRgbaToRgb(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("rgbaToRgb(dst, src, pixelCount)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "rgbaToRgb", false, 3, false, 4, &count)) return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();
    broimage::rgba_to_rgb_u8(src.data, dst.data, count);
    return ev::undefined();
}

Value imageRgbToRgba(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("rgbToRgba(dst, src, pixelCount, alpha?)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "rgbToRgba", false, 4, false, 3, &count)) return ev::undefined();
    int32_t alpha = 255;
    if (args.size() >= 4 && !ev::isUndefined(args[3])) {
        if (!ev::isNumber(args[3])) return ev::throwTypeError("rgbToRgba: alpha must be a number");
        const double a = ev::toDouble(args[3]);
        alpha = a >= 255 ? 255 : a > 0 ? static_cast<int32_t>(a) : 0;
    }
    if (!resolveViews({&dst, &src})) return ev::undefined();
    broimage::rgb_to_rgba_u8(src.data, dst.data, count, static_cast<uint8_t>(alpha));
    return ev::undefined();
}

Value imageRgbaToGray(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("rgbaToGray(dst, src, pixelCount)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "rgbaToGray", false, 1, false, 4, &count)) return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();
    broimage::rgba_to_gray_u8(src.data, dst.data, count);
    return ev::undefined();
}

Value imageRgbToGray(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("rgbToGray(dst, src, pixelCount)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "rgbToGray", false, 1, false, 3, &count)) return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();
    broimage::rgb_to_gray_u8(src.data, dst.data, count);
    return ev::undefined();
}

Value imageSrgbToLinear(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("srgbToLinear(dst, src, count)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    const bool fromBytes = src.bytesPerElement == 1 && dst.bytesPerElement == 4;
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "srgbToLinear", true, 1, !fromBytes, 1, &count))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    if (fromBytes) {
        broimage::srgb_to_linear_u8_to_f32(src.data, reinterpret_cast<float*>(dst.data), count);
    } else {
        broimage::srgb_to_linear_f32(reinterpret_cast<const float*>(src.data),
                                     reinterpret_cast<float*>(dst.data), count);
    }
    return ev::undefined();
}

Value imageLinearToSrgb(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("linearToSrgb(dst, src, count)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    const bool toBytes = src.bytesPerElement == 4 && dst.bytesPerElement == 1;
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "linearToSrgb", !toBytes, 1, true, 1, &count))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    if (toBytes) {
        broimage::linear_f32_to_srgb_u8(reinterpret_cast<const float*>(src.data), dst.data, count);
    } else {
        broimage::linear_to_srgb_f32(reinterpret_cast<const float*>(src.data),
                                     reinterpret_cast<float*>(dst.data), count);
    }
    return ev::undefined();
}

Value imageApplyGamma(Value, std::span<const Value> args) {
    if (args.size() < 4) return ev::throwTypeError("applyGamma(dst, src, count, gamma)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "applyGamma", true, 1, true, 1, &count)) return ev::undefined();
    if (!ev::isNumber(args[3])) return ev::throwTypeError("applyGamma: gamma must be a number");
    double gamma = ev::toDouble(args[3]);
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::apply_gamma_f32(reinterpret_cast<const float*>(src.data),
                              reinterpret_cast<float*>(dst.data),
                              count, static_cast<float>(gamma));
    return ev::undefined();
}

Value imageRgbToHsv(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("rgbToHsv(dst, src, pixelCount)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "rgbToHsv", true, 3, true, 3, &count)) return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::rgb_to_hsv_f32(reinterpret_cast<const float*>(src.data),
                             reinterpret_cast<float*>(dst.data), count);
    return ev::undefined();
}

Value imageHsvToRgb(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("hsvToRgb(dst, src, pixelCount)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "hsvToRgb", true, 3, true, 3, &count)) return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::hsv_to_rgb_f32(reinterpret_cast<const float*>(src.data),
                             reinterpret_cast<float*>(dst.data), count);
    return ev::undefined();
}

Value imageRgbToHsl(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("rgbToHsl(dst, src, pixelCount)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "rgbToHsl", true, 3, true, 3, &count)) return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::rgb_to_hsl_f32(reinterpret_cast<const float*>(src.data),
                             reinterpret_cast<float*>(dst.data), count);
    return ev::undefined();
}

Value imageHslToRgb(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("hslToRgb(dst, src, pixelCount)");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &src)) return ev::undefined();
    int32_t count = 0;
    if (!countedPair(dst, src, args[2], "hslToRgb", true, 3, true, 3, &count)) return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();

    broimage::hsl_to_rgb_f32(reinterpret_cast<const float*>(src.data),
                             reinterpret_cast<float*>(dst.data), count);
    return ev::undefined();
}

// ---------------------------------------------------------------------------
// 4. Normalization and Layout
// ---------------------------------------------------------------------------

Value imageNormalize(Value, std::span<const Value> args) {
    if (args.size() < 8) return ev::throwTypeError("normalize(Y, X, mean, std, N, C, H, W)");
    TypedArrayView yView, xView;
    if (!unpackTypedArray(args[0], "Y", &yView)) return ev::undefined();
    if (!unpackTypedArray(args[1], "X", &xView)) return ev::undefined();

    int32_t n = 0, c = 0, h = 0, w = 0;
    if (!countArg(args[4], "normalize", "N", &n) || !countArg(args[5], "normalize", "C", &c) ||
        !countArg(args[6], "normalize", "H", &h) || !countArg(args[7], "normalize", "W", &w))
        return ev::undefined();

    if (!requireFloat32(yView, "normalize") || !requireFloat32(xView, "normalize"))
        return ev::undefined();
    const uint64_t need = static_cast<uint64_t>(n) * c * h * w * sizeof(float);
    if (!requireBytes(yView, need, "normalize") || !requireBytes(xView, need, "normalize"))
        return ev::undefined();

    // mean/std must each carry C numbers; a short array used to be read past
    // its end (a Float32Array by memcpy, a JS array as NaN).
    std::vector<float> meanBuf(static_cast<size_t>(c));
    std::vector<float> stdBuf(static_cast<size_t>(c));
    if (!readFloatArray(args[2], meanBuf.data(), c, "normalize: mean") ||
        !readFloatArray(args[3], stdBuf.data(), c, "normalize: std"))
        return ev::undefined();

    if (!resolveViews({&yView, &xView})) return ev::undefined();

    broimage::image_normalize_nchw_f32(
        reinterpret_cast<const float*>(xView.data),
        meanBuf.data(), stdBuf.data(),
        n, c, h, w,
        reinterpret_cast<float*>(yView.data));
    return ev::undefined();
}

Value imageU8ToF32(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("u8ToF32(Y, src, options)");
    TypedArrayView yView, srcView;
    if (!unpackTypedArray(args[0], "Y", &yView)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &srcView)) return ev::undefined();

    int32_t n = 1, h = 0, w = 0, c = 3;
    double scale = 1.0 / 255.0, bias = 0.0;
    if (!getPropI32(args[2], "n", &n, 1)) return ev::undefined();
    if (!getPropI32(args[2], "h", &h, 0)) return ev::undefined();
    if (!getPropI32(args[2], "w", &w, 0)) return ev::undefined();
    if (!getPropI32(args[2], "c", &c, 3)) return ev::undefined();
    if (!getPropF64(args[2], "scale", &scale, 1.0 / 255.0)) return ev::undefined();
    if (!getPropF64(args[2], "bias",  &bias,  0.0)) return ev::undefined();

    if (n <= 0 || h <= 0 || w <= 0 || c <= 0)
        return ev::throwRangeError("u8ToF32: invalid dimensions");
    const uint64_t elems = static_cast<uint64_t>(n) * h * w * c;
    if (!requireFloat32(yView, "u8ToF32") || !requireBytes(yView, elems * 4, "u8ToF32") ||
        !requireBytes(srcView, elems, "u8ToF32"))
        return ev::undefined();
    if (!resolveViews({&yView, &srcView})) return ev::undefined();

    broimage::u8_nhwc_to_f32_nchw(
        srcView.data, n, h, w, c,
        static_cast<float>(scale), static_cast<float>(bias),
        reinterpret_cast<float*>(yView.data));
    return ev::undefined();
}

Value imageF32ToU8(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("f32ToU8(Y, src, options)");
    TypedArrayView yView, srcView;
    if (!unpackTypedArray(args[0], "Y", &yView)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &srcView)) return ev::undefined();

    int32_t n = 1, c = 3, h = 0, w = 0;
    double scale = 255.0, bias = 0.0;
    if (!getPropI32(args[2], "n", &n, 1)) return ev::undefined();
    if (!getPropI32(args[2], "c", &c, 3)) return ev::undefined();
    if (!getPropI32(args[2], "h", &h, 0)) return ev::undefined();
    if (!getPropI32(args[2], "w", &w, 0)) return ev::undefined();
    if (!getPropF64(args[2], "scale", &scale, 255.0)) return ev::undefined();
    if (!getPropF64(args[2], "bias",  &bias,  0.0)) return ev::undefined();

    if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
        return ev::throwRangeError("f32ToU8: invalid dimensions");
    const uint64_t elems = static_cast<uint64_t>(n) * c * h * w;
    if (!requireFloat32(srcView, "f32ToU8") || !requireBytes(srcView, elems * 4, "f32ToU8") ||
        !requireBytes(yView, elems, "f32ToU8"))
        return ev::undefined();
    if (!resolveViews({&yView, &srcView})) return ev::undefined();

    broimage::f32_nchw_to_u8_nhwc(
        reinterpret_cast<const float*>(srcView.data), n, c, h, w,
        static_cast<float>(scale), static_cast<float>(bias),
        yView.data);
    return ev::undefined();
}

Value imageNhwcToNchw(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("nhwcToNchw(Y, src, options)");
    TypedArrayView yView, srcView;
    if (!unpackTypedArray(args[0], "Y", &yView)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &srcView)) return ev::undefined();

    int32_t n = 1, h = 0, w = 0, c = 3;
    if (!getPropI32(args[2], "n", &n, 1)) return ev::undefined();
    if (!getPropI32(args[2], "h", &h, 0)) return ev::undefined();
    if (!getPropI32(args[2], "w", &w, 0)) return ev::undefined();
    if (!getPropI32(args[2], "c", &c, 3)) return ev::undefined();

    if (n <= 0 || h <= 0 || w <= 0 || c <= 0)
        return ev::throwRangeError("nhwcToNchw: invalid dimensions");
    const uint64_t need = static_cast<uint64_t>(n) * h * w * c * 4;
    if (!requireFloat32(yView, "nhwcToNchw") || !requireFloat32(srcView, "nhwcToNchw") ||
        !requireBytes(yView, need, "nhwcToNchw") || !requireBytes(srcView, need, "nhwcToNchw"))
        return ev::undefined();
    if (!resolveViews({&yView, &srcView})) return ev::undefined();

    broimage::nhwc_to_nchw_f32(
        reinterpret_cast<const float*>(srcView.data), n, h, w, c,
        reinterpret_cast<float*>(yView.data));
    return ev::undefined();
}

Value imageNchwToNhwc(Value, std::span<const Value> args) {
    if (args.size() < 3) return ev::throwTypeError("nchwToNhwc(Y, src, options)");
    TypedArrayView yView, srcView;
    if (!unpackTypedArray(args[0], "Y", &yView)) return ev::undefined();
    if (!unpackTypedArray(args[1], "src", &srcView)) return ev::undefined();

    int32_t n = 1, c = 3, h = 0, w = 0;
    if (!getPropI32(args[2], "n", &n, 1)) return ev::undefined();
    if (!getPropI32(args[2], "c", &c, 3)) return ev::undefined();
    if (!getPropI32(args[2], "h", &h, 0)) return ev::undefined();
    if (!getPropI32(args[2], "w", &w, 0)) return ev::undefined();

    if (n <= 0 || c <= 0 || h <= 0 || w <= 0)
        return ev::throwRangeError("nchwToNhwc: invalid dimensions");
    const uint64_t need = static_cast<uint64_t>(n) * c * h * w * 4;
    if (!requireFloat32(yView, "nchwToNhwc") || !requireFloat32(srcView, "nchwToNhwc") ||
        !requireBytes(yView, need, "nchwToNhwc") || !requireBytes(srcView, need, "nchwToNhwc"))
        return ev::undefined();
    if (!resolveViews({&yView, &srcView})) return ev::undefined();

    broimage::nchw_to_nhwc_f32(
        reinterpret_cast<const float*>(srcView.data), n, c, h, w,
        reinterpret_cast<float*>(yView.data));
    return ev::undefined();
}

} // namespace

void installOpsOnto(Value imageObj) {
    ObjectBuilder b(imageObj);

    // Kernels
    b.def("gradient", 2, imageGradient);
    b.def("alloc", 4, imageAlloc);
    b.def("lookup", 4, imageLookup);
    b.def("reduce", 3, imageReduce);
    b.def("map", 3, imageMap);
    b.def("combine", 4, imageCombine);
    b.def("stencil", 4, imageStencil);
    b.def("resample", 3, imageResample);

    // Geometric
    b.def("resize", 3, imageResize);
    b.def("crop", 3, imageCrop);
    b.def("centerCrop", 3, imageCenterCrop);
    b.def("flipHorizontal", 3, imageFlipHorizontal);
    b.def("flipVertical", 3, imageFlipVertical);
    b.def("rotate90", 3, imageRotate90);
    b.def("pad", 3, imagePad);

    // Color
    b.def("rgbaToRgb", 3, imageRgbaToRgb);
    b.def("rgbToRgba", 4, imageRgbToRgba);
    b.def("rgbaToGray", 3, imageRgbaToGray);
    b.def("rgbToGray", 3, imageRgbToGray);
    b.def("srgbToLinear", 3, imageSrgbToLinear);
    b.def("linearToSrgb", 3, imageLinearToSrgb);
    b.def("applyGamma", 4, imageApplyGamma);
    b.def("rgbToHsv", 3, imageRgbToHsv);
    b.def("hsvToRgb", 3, imageHsvToRgb);
    b.def("rgbToHsl", 3, imageRgbToHsl);
    b.def("hslToRgb", 3, imageHslToRgb);

    // Normalization & Layout
    b.def("normalize", 8, imageNormalize);
    b.def("u8ToF32", 3, imageU8ToF32);
    b.def("f32ToU8", 3, imageF32ToU8);
    b.def("nhwcToNchw", 3, imageNhwcToNchw);
    b.def("nchwToNhwc", 3, imageNchwToNhwc);
}

} // namespace broimage::api
