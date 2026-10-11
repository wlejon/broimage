// bro.image.sharpness: focus measures (variance of the Laplacian, Tenengrad)
// over a region of an 8-bit image. The kernel is include/broimage/sharpness.h.

#include "host_image_internal.h"

#include <broimage/sharpness.h>

#include <cstdint>
#include <span>
#include <string>

namespace broimage::api {

namespace {

// sharpness(src, { width, height, channels?, stride?, rect?: { x, y, w, h }, threads? })
Value imageSharpness(Value, std::span<const Value> args) {
    const char* who = "sharpness";
    if (args.size() < 2 || !ev::isObject(args[1]))
        return ev::throwTypeError("sharpness(src, { width, height, channels, rect? })");
    TypedArrayView src;
    if (!unpackTypedArray(args[0], "src", &src)) return ev::undefined();
    if (src.bytesPerElement != 1)
        return ev::throwTypeError("sharpness: src must be a Uint8Array / Uint8ClampedArray");
    // Rooted: every option read may allocate and move it (embed.h's pointer contract).
    ev::Persistent optsRoot(args[1]);

    int32_t w = 0, h = 0, ch = 4, stride = 0, threads = 0;
    if (!getPropI32(optsRoot.get(), "width", &w, 0) || !getPropI32(optsRoot.get(), "height", &h, 0) ||
        !getPropI32(optsRoot.get(), "channels", &ch, 4) || !getPropI32(optsRoot.get(), "stride", &stride, 0) ||
        !getPropI32(optsRoot.get(), "threads", &threads, 0))
        return ev::undefined();
    if (w <= 0 || h <= 0 || ch < 1 || ch > 4)
        return ev::throwRangeError("sharpness: width/height must be positive, channels 1..4");
    if (threads < 0 || threads > 64) return ev::throwRangeError("sharpness: threads 0..64");

    int32_t rx = 0, ry = 0, rw = 0, rh = 0;
    {
        ev::Persistent rect(ev::getProperty(optsRoot.get(), "rect"));
        if (!ev::isUndefined(rect.get()) && !ev::isNull(rect.get())) {
            if (!ev::isObject(rect.get())) return ev::throwTypeError("sharpness: rect must be { x, y, w, h }");
            if (!getPropI32(rect.get(), "x", &rx, 0) || !getPropI32(rect.get(), "y", &ry, 0) ||
                !getPropI32(rect.get(), "w", &rw, 0) || !getPropI32(rect.get(), "h", &rh, 0))
                return ev::undefined();
            if (rx < 0 || ry < 0 || rw < 0 || rh < 0)
                return ev::throwRangeError("sharpness: rect must not be negative");
            // An empty rect measures nothing (0 would mean "to the edge" below).
            if (rw == 0 || rh == 0 || rx >= w || ry >= h) rw = rh = -1;
        }
    }
    if (!requireImage(src, w, h, ch, stride, 1, who)) return ev::undefined();
    if (!resolveViews({&src})) return ev::undefined();

    broimage::SharpnessResult r;
    if (rw >= 0) r = broimage::sharpness_u8(src.data, w, h, ch, stride, rx, ry, rw, rh, threads);
    ObjectBuilder b;
    b.set("laplacianVar", r.laplacian_variance);
    b.set("tenengrad", r.tenengrad);
    b.set("lumaMean", r.luma_mean);
    b.set("lumaStd", r.luma_std);
    b.set("pixels", static_cast<double>(r.pixels));
    return b.build();
}

} // namespace

void installSharpnessOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("sharpness", 2, imageSharpness);
}

} // namespace broimage::api
