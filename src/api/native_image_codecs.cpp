#include "host_image_internal.h"

#include <broimage/encode.h>
#if defined(BROIMAGE_HAS_KTX2)
#include <broimage/ktx2.h>
#endif

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

Value encodePngFileValue(Value, std::span<const Value> a) {
    if (a.size() < 5) return ev::throwTypeError("encodePngFile(path, pixels, w, h, channels, strideBytes?)");
    std::string path = resolvePath(ev::toUtf8(a[0]));
    auto info = ev::typedArrayInfo(a[1]);
    if (!info.data) return ev::throwTypeError("encodePngFile: pixels must be a TypedArray");
    int32_t w = static_cast<int32_t>(ev::toDouble(a[2]));
    int32_t h = static_cast<int32_t>(ev::toDouble(a[3]));
    int32_t c = static_cast<int32_t>(ev::toDouble(a[4]));
    int32_t stride = 0;
    if (a.size() >= 6 && !ev::isUndefined(a[5])) {
        stride = static_cast<int32_t>(ev::toDouble(a[5]));
    }
    bool ok = broimage::encode_png_file(path, info.data, w, h, c, stride);
    return ev::fromBool(ok);
}

Value encodePngValue(Value, std::span<const Value> a) {
    if (a.size() < 4) return ev::throwTypeError("encodePng(pixels, w, h, channels, strideBytes?)");
    auto info = ev::typedArrayInfo(a[0]);
    if (!info.data) return ev::throwTypeError("encodePng: pixels must be a TypedArray");
    int32_t w = static_cast<int32_t>(ev::toDouble(a[1]));
    int32_t h = static_cast<int32_t>(ev::toDouble(a[2]));
    int32_t c = static_cast<int32_t>(ev::toDouble(a[3]));
    int32_t stride = 0;
    if (a.size() >= 5 && !ev::isUndefined(a[4])) {
        stride = static_cast<int32_t>(ev::toDouble(a[4]));
    }
    std::vector<uint8_t> out;
    if (!broimage::encode_png_memory(out, info.data, w, h, c, stride)) return ev::null();
    return typedArrayFrom(ev::elements::Uint8, out.data(), out.size(), static_cast<uint32_t>(out.size()));
}

Value encodeJpegFileValue(Value, std::span<const Value> a) {
    if (a.size() < 5) return ev::throwTypeError("encodeJpegFile(path, pixels, w, h, channels, quality?)");
    std::string path = resolvePath(ev::toUtf8(a[0]));
    auto info = ev::typedArrayInfo(a[1]);
    if (!info.data) return ev::throwTypeError("encodeJpegFile: pixels must be a TypedArray");
    int32_t w = static_cast<int32_t>(ev::toDouble(a[2]));
    int32_t h = static_cast<int32_t>(ev::toDouble(a[3]));
    int32_t c = static_cast<int32_t>(ev::toDouble(a[4]));
    int32_t quality = 90;
    if (a.size() >= 6 && !ev::isUndefined(a[5])) {
        quality = static_cast<int32_t>(ev::toDouble(a[5]));
    }
    bool ok = broimage::encode_jpeg_file(path, info.data, w, h, c, quality);
    return ev::fromBool(ok);
}

Value encodeJpegValue(Value, std::span<const Value> a) {
    if (a.size() < 4) return ev::throwTypeError("encodeJpeg(pixels, w, h, channels, quality?)");
    auto info = ev::typedArrayInfo(a[0]);
    if (!info.data) return ev::throwTypeError("encodeJpeg: pixels must be a TypedArray");
    int32_t w = static_cast<int32_t>(ev::toDouble(a[1]));
    int32_t h = static_cast<int32_t>(ev::toDouble(a[2]));
    int32_t c = static_cast<int32_t>(ev::toDouble(a[3]));
    int32_t quality = 90;
    if (a.size() >= 5 && !ev::isUndefined(a[4])) {
        quality = static_cast<int32_t>(ev::toDouble(a[4]));
    }
    std::vector<uint8_t> out;
    if (!broimage::encode_jpeg_memory(out, info.data, w, h, c, quality)) return ev::null();
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
