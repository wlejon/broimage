// bro.image.compare: how one 8-bit image differs from another of the same
// size, overall and per cell of a grid. The kernel is
// include/broimage/compare.h.

#include "host_image_internal.h"

#include <broimage/compare.h>

#include <cmath>
#include <cstdint>
#include <span>

namespace broimage::api {

namespace {

Value statsObject(const broimage::CompareStats& s, int channels) {
    ev::Persistent mean(ev::makeArray(0));
    for (int c = 0; c < channels; ++c) {
        // The number first: it may allocate, and mean.get() must be read after.
        ev::Persistent n(ev::fromDouble(s.mean[c]));
        mean.set(ev::setElement(mean.get(), static_cast<uint32_t>(c), n.get()));
    }
    ObjectBuilder b;
    b.set("mean", mean.get());
    b.set("rms", s.rms);
    // (Equal images: Infinity, as the dB of no error is.)
    b.set("psnr", s.psnr);
    b.set("maxAbs", static_cast<double>(s.max_abs));
    b.set("changed", s.changed);
    b.set("ssim", s.ssim);
    return b.build();
}

// compare(a, b, { width, height, channels?, strideA?, strideB?, rows?, cols?, threshold? })
Value imageCompare(Value, std::span<const Value> args) {
    const char* who = "compare";
    if (args.size() < 3 || !ev::isObject(args[2]))
        return ev::throwTypeError("compare(a, b, { width, height, channels, rows?, cols? })");
    TypedArrayView a, b;
    if (!unpackTypedArray(args[0], "a", &a) || !unpackTypedArray(args[1], "b", &b)) return ev::undefined();
    if (a.bytesPerElement != 1 || b.bytesPerElement != 1)
        return ev::throwTypeError("compare: a and b must be Uint8Array / Uint8ClampedArray");
    // Rooted: every option read may allocate and move it (embed.h's pointer contract).
    ev::Persistent optsRoot(args[2]);

    int32_t w = 0, h = 0, ch = 4, sa = 0, sb = 0, rows = 1, cols = 1, threshold = 8;
    if (!getPropI32(optsRoot.get(), "width", &w, 0) || !getPropI32(optsRoot.get(), "height", &h, 0) ||
        !getPropI32(optsRoot.get(), "channels", &ch, 4) || !getPropI32(optsRoot.get(), "strideA", &sa, 0) ||
        !getPropI32(optsRoot.get(), "strideB", &sb, 0) || !getPropI32(optsRoot.get(), "rows", &rows, 1) ||
        !getPropI32(optsRoot.get(), "cols", &cols, 1) || !getPropI32(optsRoot.get(), "threshold", &threshold, 8))
        return ev::undefined();
    if (w <= 0 || h <= 0 || ch < 1 || ch > 4)
        return ev::throwRangeError("compare: width/height must be positive, channels 1..4");
    if (rows < 1 || cols < 1) return ev::throwRangeError("compare: rows and cols must be at least 1");
    if (threshold < 0 || threshold > 255) return ev::throwRangeError("compare: threshold 0..255");
    if (!requireImage(a, w, h, ch, sa, 1, who) || !requireImage(b, w, h, ch, sb, 1, who)) return ev::undefined();
    if (!resolveViews({&a, &b})) return ev::undefined();

    const broimage::CompareResult r = broimage::compare_u8(a.data, b.data, w, h, ch, sa, sb, rows, cols, threshold);
    ev::Persistent all(statsObject(r.all, ch));
    ev::Persistent cells(ev::makeArray(0));
    for (size_t i = 0; i < r.cells.size(); ++i) {
        const broimage::CompareCell& c = r.cells[i];
        ev::Persistent o(statsObject(c.stats, ch));
        const char* keys[4] = {"x", "y", "w", "h"};
        const int vals[4] = {c.x, c.y, c.w, c.h};
        for (int k = 0; k < 4; ++k) {
            ev::Persistent n(ev::fromDouble(vals[k]));
            o.set(ev::setProperty(o.get(), keys[k], n.get()));
        }
        cells.set(ev::setElement(cells.get(), static_cast<uint32_t>(i), o.get()));
    }
    all.set(ev::setProperty(all.get(), "cells", cells.get()));
    return all.get();
}

} // namespace

void installCompareOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("compare", 3, imageCompare);
}

} // namespace broimage::api
