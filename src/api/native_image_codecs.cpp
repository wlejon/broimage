#include "host_image_internal.h"

#include <broimage/encode.h>
#if defined(BROIMAGE_HAS_KTX2)
#include <broimage/ktx2.h>
#endif

#include <climits>
#include <cmath>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace broimage::api {

namespace {

bool bytesOf(Value v, std::span<const uint8_t>& out) {
    if (auto info = ev::typedArrayInfo(v)) {
        if (!info.data) return false;
        out = std::span<const uint8_t>(info.data, info.byteLength);
        return true;
    }
    if (auto ab = ev::arrayBufferInfo(v)) {
        if (!ab.data) return false;
        out = std::span<const uint8_t>(ab.data, ab.byteLength);
        return true;
    }
    return false;
}

Value transcodeKtx2Value(Value, std::span<const Value> args) {
#if defined(BROIMAGE_HAS_KTX2)
    std::span<const uint8_t> bytes;
    if (args.empty() || !bytesOf(args[0], bytes)) {
        return ev::throwTypeError("transcodeKTX2 requires (bytes: a typed array)");
    }
    broimage::Ktx2Format target = broimage::Ktx2Format::RGBA8;
    if (args.size() > 1 && ev::isString(args[1])) {
        const std::string f = ev::toUtf8(args[1]);
        if (f == "bc1") target = broimage::Ktx2Format::BC1;
        else if (f == "bc3") target = broimage::Ktx2Format::BC3;
        else if (f == "bc4") target = broimage::Ktx2Format::BC4;
        else if (f == "bc5") target = broimage::Ktx2Format::BC5;
        else if (f == "bc7") target = broimage::Ktx2Format::BC7;
        else if (f != "rgba8")
            return ev::throwTypeError(("transcodeKTX2: unknown format '" + f + "'").c_str());
    }

    broimage::Ktx2Image img = broimage::transcode_ktx2(bytes.data(), bytes.size(), target);
    if (!img.ok()) return ev::throwTypeError(img.error.c_str());

    const char* formatName = "rgba8";
    switch (img.format) {
        case broimage::Ktx2Format::BC1: formatName = "bc1"; break;
        case broimage::Ktx2Format::BC3: formatName = "bc3"; break;
        case broimage::Ktx2Format::BC4: formatName = "bc4"; break;
        case broimage::Ktx2Format::BC5: formatName = "bc5"; break;
        case broimage::Ktx2Format::BC7: formatName = "bc7"; break;
        default: break;
    }

    ObjectBuilder out;
    out.set("width", ev::fromDouble(img.width));
    out.set("height", ev::fromDouble(img.height));
    out.set("hasAlpha", ev::fromBool(img.hasAlpha));
    out.set("srgb", ev::fromBool(img.srgb));
    out.set("format", ev::fromUtf8(formatName));
    out.set("mips", hostArrayOf(img.mips.size(), [&](size_t i) -> Value {
        const broimage::Ktx2Level& level = img.mips[i];
        ObjectBuilder mip;
        mip.set("width", ev::fromDouble(level.width));
        mip.set("height", ev::fromDouble(level.height));
        mip.set("data", typedArrayFrom(ev::elements::Uint8, level.data.data(),
                                       level.data.size(),
                                       static_cast<uint32_t>(level.data.size())));
        return mip.get();
    }));
    return out.get();
#else
    (void)args;
    return ev::throwTypeError("transcodeKTX2: KTX2 support not compiled in");
#endif
}

// The shared argument shape of the four encoders: pixels, w, h, channels and
// one optional trailing integer (PNG row stride, JPEG quality), starting at
// a[first]. The pixel view is checked against what the encoder will read —
// h rows of the stride (or w*channels) bytes — and left unresolved; the
// caller resolves it after its last allocating call.
struct EncodeArgs {
    TypedArrayView px;
    int32_t w = 0, h = 0, c = 0, extra = 0;
};

bool readEncodeArgs(std::span<const Value> a, size_t first, const char* who, bool extraIsStride,
                    int32_t extraDefault, EncodeArgs* out) {
    if (!unpackTypedArray(a[first], "pixels", &out->px)) return false;
    if (out->px.bytesPerElement != 1) {
        ev::throwTypeError(std::string(who) + ": pixels must be a Uint8Array");
        return false;
    }
    if (!countArg(a[first + 1], who, "width", &out->w) ||
        !countArg(a[first + 2], who, "height", &out->h) ||
        !countArg(a[first + 3], who, "channels", &out->c))
        return false;
    if (out->c > 4) {
        ev::throwRangeError(std::string(who) + ": channels must be 1 to 4");
        return false;
    }
    out->extra = extraDefault;
    if (a.size() > first + 4 && !ev::isUndefined(a[first + 4])) {
        if (!ev::isNumber(a[first + 4])) {
            ev::throwTypeError(std::string(who) + (extraIsStride ? ": strideBytes" : ": quality") +
                               " must be a number");
            return false;
        }
        const double d = std::trunc(ev::toDouble(a[first + 4]));
        if (!(d >= 0 && d <= static_cast<double>(INT32_MAX))) {
            ev::throwRangeError(std::string(who) + (extraIsStride ? ": strideBytes" : ": quality") +
                                " is out of range");
            return false;
        }
        out->extra = static_cast<int32_t>(d);
    }
    return requireImage(out->px, out->w, out->h, out->c, extraIsStride ? out->extra : 0, 1, who);
}

Value encodePngFileValue(Value, std::span<const Value> a) {
    if (a.size() < 5) return ev::throwTypeError("encodePngFile(path, pixels, w, h, channels, strideBytes?)");
    if (!ev::isString(a[0])) return ev::throwTypeError("encodePngFile: path must be a string");
    EncodeArgs e;
    if (!readEncodeArgs(a, 1, "encodePngFile", true, 0, &e)) return ev::undefined();
    std::string path = resolvePath(ev::toUtf8(a[0]));
    if (!resolveViews({&e.px})) return ev::undefined();
    bool ok = broimage::encode_png_file(path, e.px.data, e.w, e.h, e.c, e.extra);
    return ev::fromBool(ok);
}

Value encodePngValue(Value, std::span<const Value> a) {
    if (a.size() < 4) return ev::throwTypeError("encodePng(pixels, w, h, channels, strideBytes?)");
    EncodeArgs e;
    if (!readEncodeArgs(a, 0, "encodePng", true, 0, &e)) return ev::undefined();
    if (!resolveViews({&e.px})) return ev::undefined();
    std::vector<uint8_t> out;
    if (!broimage::encode_png_memory(out, e.px.data, e.w, e.h, e.c, e.extra)) return ev::null();
    return typedArrayFrom(ev::elements::Uint8, out.data(), out.size(), static_cast<uint32_t>(out.size()));
}

Value encodeJpegFileValue(Value, std::span<const Value> a) {
    if (a.size() < 5) return ev::throwTypeError("encodeJpegFile(path, pixels, w, h, channels, quality?)");
    if (!ev::isString(a[0])) return ev::throwTypeError("encodeJpegFile: path must be a string");
    EncodeArgs e;
    if (!readEncodeArgs(a, 1, "encodeJpegFile", false, 90, &e)) return ev::undefined();
    std::string path = resolvePath(ev::toUtf8(a[0]));
    if (!resolveViews({&e.px})) return ev::undefined();
    bool ok = broimage::encode_jpeg_file(path, e.px.data, e.w, e.h, e.c, e.extra);
    return ev::fromBool(ok);
}

Value encodeJpegValue(Value, std::span<const Value> a) {
    if (a.size() < 4) return ev::throwTypeError("encodeJpeg(pixels, w, h, channels, quality?)");
    EncodeArgs e;
    if (!readEncodeArgs(a, 0, "encodeJpeg", false, 90, &e)) return ev::undefined();
    if (!resolveViews({&e.px})) return ev::undefined();
    std::vector<uint8_t> out;
    if (!broimage::encode_jpeg_memory(out, e.px.data, e.w, e.h, e.c, e.extra)) return ev::null();
    return typedArrayFrom(ev::elements::Uint8, out.data(), out.size(), static_cast<uint32_t>(out.size()));
}

} // namespace

void installCodecsOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("transcodeKTX2", 2, transcodeKtx2Value);
    b.def("encodePngFile", 5, encodePngFileValue);
    b.def("encodePng", 4, encodePngValue);
    b.def("encodeJpegFile", 5, encodeJpegFileValue);
    b.def("encodeJpeg", 4, encodeJpegValue);
}

} // namespace broimage::api
