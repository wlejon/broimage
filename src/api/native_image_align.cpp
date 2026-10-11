// bro.image.alignSimilarity / similarityMatrix / warpAffineU8: estimate how a
// subject moved between two frames and render a frame with it held still.
// The kernels are include/broimage/align.h.

#include "host_image_internal.h"

#include <broimage/align.h>

#include <cmath>
#include <cstdint>
#include <span>
#include <string>

namespace broimage::api {

namespace {

// Read `n` numbers (as doubles) from obj[key]: an array or a typed array.
bool getPropDoubles(Value obj, const char* key, double* out, int n, const char* who) {
    ev::Persistent v(ev::getProperty(obj, key));
    if (!ev::isObject(v.get())) {
        ev::throwTypeError(std::string(who) + ": " + key + " must be an array of " + std::to_string(n) + " numbers");
        return false;
    }
    Value lenV = ev::getProperty(v.get(), "length");
    if (!ev::isNumber(lenV) || !(ev::toDouble(lenV) >= n)) {
        ev::throwTypeError(std::string(who) + ": " + key + " must be an array of " + std::to_string(n) + " numbers");
        return false;
    }
    for (int i = 0; i < n; ++i) out[i] = ev::toDouble(ev::getElement(v.get(), static_cast<uint32_t>(i)));
    return true;
}

// { scale, rotation, tx, ty } from an object (missing fields: identity).
bool readSimilarity(Value obj, broimage::Similarity* s) {
    if (ev::isUndefined(obj) || ev::isNull(obj)) return true;
    ev::Persistent root(obj);
    return getPropF64(root.get(), "scale", &s->scale, 1.0) && getPropF64(root.get(), "rotation", &s->rotation, 0.0) &&
           getPropF64(root.get(), "tx", &s->tx, 0.0) && getPropF64(root.get(), "ty", &s->ty, 0.0);
}

bool getPropBool(Value obj, const char* key, bool* out, bool defVal) {
    Value v = ev::getProperty(obj, key);
    *out = (ev::isUndefined(v) || ev::isNull(v)) ? defVal : ev::toBool(v);
    return true;
}

Value matrixArray(const double m[6]) {
    ev::Persistent arr(ev::makeArray(0));
    for (uint32_t i = 0; i < 6; ++i) {
        // The number first: it may allocate, and arr.get() must be read after.
        ev::Persistent n(ev::fromDouble(m[i]));
        arr.set(ev::setElement(arr.get(), i, n.get()));
    }
    return arr.get();
}

// alignSimilarity(ref, mov, { width, height, channels, mask?, init?, levels?,
//                 search?, iterations?, similarity?, photometric?,
//                 refStride?, movStride?, cx?, cy? })
Value imageAlignSimilarity(Value, std::span<const Value> args) {
    const char* who = "alignSimilarity";
    if (args.size() < 3 || !ev::isObject(args[2]))
        return ev::throwTypeError("alignSimilarity(ref, mov, { width, height, channels, ... })");
    TypedArrayView ref, mov, mask;
    if (!unpackTypedArray(args[0], "ref", &ref) || !unpackTypedArray(args[1], "mov", &mov))
        return ev::undefined();
    if (ref.bytesPerElement != 1 || mov.bytesPerElement != 1)
        return ev::throwTypeError("alignSimilarity: ref and mov must be Uint8Array / Uint8ClampedArray");
    // Rooted: every option read may allocate and move it (embed.h's pointer contract).
    ev::Persistent optsRoot(args[2]);

    int32_t w = 0, h = 0, ch = 4, rs = 0, ms = 0, levels = 4, search = 4, iters = 40;
    if (!getPropI32(optsRoot.get(),"width", &w, 0) || !getPropI32(optsRoot.get(),"height", &h, 0) ||
        !getPropI32(optsRoot.get(),"channels", &ch, 4) || !getPropI32(optsRoot.get(),"refStride", &rs, 0) ||
        !getPropI32(optsRoot.get(),"movStride", &ms, 0) || !getPropI32(optsRoot.get(),"levels", &levels, 4) ||
        !getPropI32(optsRoot.get(),"search", &search, 4) || !getPropI32(optsRoot.get(),"iterations", &iters, 40))
        return ev::undefined();
    if (w <= 0 || h <= 0 || ch <= 0 || ch > 4)
        return ev::throwRangeError("alignSimilarity: width/height must be positive, channels 1..4");
    if (levels < 0 || levels > 12 || search < 0 || search > 64 || iters < 1 || iters > 1000)
        return ev::throwRangeError("alignSimilarity: levels 0..12, search 0..64, iterations 1..1000");

    broimage::AlignOptions o;
    o.levels = levels;
    o.search = search;
    o.max_iterations = iters;
    getPropBool(optsRoot.get(),"similarity", &o.similarity, true);
    getPropBool(optsRoot.get(),"photometric", &o.photometric, true);

    broimage::Similarity init;
    {
        ev::Persistent initV(ev::getProperty(optsRoot.get(),"init"));
        if (!readSimilarity(initV.get(), &init)) return ev::undefined();
    }

    bool hasMask = false;
    {
        ev::Persistent maskV(ev::getProperty(optsRoot.get(),"mask"));
        if (!ev::isUndefined(maskV.get()) && !ev::isNull(maskV.get())) {
            if (!unpackTypedArray(maskV.get(), "mask", &mask)) return ev::undefined();
            if (mask.bytesPerElement != 1) return ev::throwTypeError("alignSimilarity: mask must be a Uint8Array");
            if (!requireImage(mask, w, h, 1, 0, 1, who)) return ev::undefined();
            hasMask = true;
        }
    }
    if (!requireImage(ref, w, h, ch, rs, 1, who) || !requireImage(mov, w, h, ch, ms, 1, who))
        return ev::undefined();

    if (hasMask) { if (!resolveViews({&ref, &mov, &mask})) return ev::undefined(); }
    else if (!resolveViews({&ref, &mov})) return ev::undefined();

    const broimage::AlignResult r = broimage::align_similarity(
        ref.data, mov.data, w, h, ch, hasMask ? mask.data : nullptr, init, o, rs, ms);

    double m[6];
    broimage::similarity_matrix(r.transform, 0.5 * w, 0.5 * h, m);
    ev::Persistent matrix(matrixArray(m));
    ObjectBuilder b;
    b.set("scale", r.transform.scale);
    b.set("rotation", r.transform.rotation);
    b.set("tx", r.transform.tx);
    b.set("ty", r.transform.ty);
    b.set("gain", r.gain);
    b.set("bias", r.bias);
    b.set("rms", r.rms);
    b.set("ncc", r.ncc);
    b.set("coverage", r.coverage);
    b.set("iterations", static_cast<double>(r.iterations));
    b.set("converged", r.converged);
    b.set("matrix", matrix.get());
    return b.build();
}

// similarityMatrix({ scale, rotation, tx, ty }, cx, cy) -> [6]
Value imageSimilarityMatrix(Value, std::span<const Value> args) {
    broimage::Similarity s;
    if (!readSimilarity(argAt(args, 0), &s)) return ev::undefined();
    double m[6];
    broimage::similarity_matrix(s, numAt(args, 1), numAt(args, 2), m);
    return matrixArray(m);
}

// warpAffineU8(dst, src, { srcW, srcH, dstW, dstH, channels, matrix,
//              filter?, fill?, srcStride?, dstStride? })
Value imageWarpAffineU8(Value, std::span<const Value> args) {
    const char* who = "warpAffineU8";
    if (args.size() < 3 || !ev::isObject(args[2]))
        return ev::throwTypeError("warpAffineU8(dst, src, { srcW, srcH, dstW, dstH, channels, matrix })");
    TypedArrayView dst, src;
    if (!unpackTypedArray(args[0], "dst", &dst) || !unpackTypedArray(args[1], "src", &src))
        return ev::undefined();
    if (dst.bytesPerElement != 1 || src.bytesPerElement != 1)
        return ev::throwTypeError("warpAffineU8: dst/src must be Uint8Array");
    ev::Persistent optsRoot(args[2]);
    int32_t sw = 0, sh = 0, dw = 0, dh = 0, ch = 4, ss = 0, ds = 0;
    if (!getPropI32(optsRoot.get(),"srcW", &sw, 0) || !getPropI32(optsRoot.get(),"srcH", &sh, 0) ||
        !getPropI32(optsRoot.get(),"dstW", &dw, 0) || !getPropI32(optsRoot.get(),"dstH", &dh, 0) ||
        !getPropI32(optsRoot.get(),"channels", &ch, 4) || !getPropI32(optsRoot.get(),"srcStride", &ss, 0) ||
        !getPropI32(optsRoot.get(),"dstStride", &ds, 0))
        return ev::undefined();
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0 || ch <= 0 || ch > 4)
        return ev::throwRangeError("warpAffineU8: dims must be positive, channels 1..4");
    double m[6];
    if (!getPropDoubles(optsRoot.get(),"matrix", m, 6, who)) return ev::undefined();
    for (double v : m)
        if (!std::isfinite(v)) return ev::throwRangeError("warpAffineU8: matrix must be finite");

    std::string fs;
    if (!getPropStr(optsRoot.get(),"filter", &fs, "bilinear")) return ev::undefined();
    broimage::Filter f = broimage::Filter::Bilinear;
    if (fs == "nearest") f = broimage::Filter::Nearest;
    else if (fs == "bicubic") f = broimage::Filter::Bicubic;
    else if (!fs.empty() && fs != "bilinear")
        return ev::throwTypeError("warpAffineU8: filter must be nearest|bilinear|bicubic");

    uint8_t fill[4] = {0, 0, 0, 0};
    bool hasFill = false;
    {
        ev::Persistent fillV(ev::getProperty(optsRoot.get(),"fill"));
        if (!ev::isUndefined(fillV.get()) && !ev::isNull(fillV.get())) {
            double fv[4] = {0, 0, 0, 0};
            if (!getPropDoubles(optsRoot.get(),"fill", fv, ch, who)) return ev::undefined();
            for (int i = 0; i < ch; ++i) fill[i] = static_cast<uint8_t>(std::fmin(std::fmax(fv[i], 0.0), 255.0));
            hasFill = true;
        }
    }
    if (!requireImage(src, sw, sh, ch, ss, 1, who) || !requireImage(dst, dw, dh, ch, ds, 1, who))
        return ev::undefined();
    if (!resolveViews({&dst, &src})) return ev::undefined();
    broimage::warp_affine_hwc_u8(src.data, sw, sh, ch, dst.data, dw, dh, m, f, hasFill ? fill : nullptr, ss, ds);
    return ev::undefined();
}

} // namespace

void installAlignOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("alignSimilarity", 3, imageAlignSimilarity);
    b.def("similarityMatrix", 3, imageSimilarityMatrix);
    b.def("warpAffineU8", 3, imageWarpAffineU8);
}

} // namespace broimage::api
