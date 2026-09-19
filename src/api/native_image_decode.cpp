// Decode / probe / EXIF entry points.
//
// Ported from the QuickJS-era src/js/image_bindings.cpp (img_decodeU16,
// img_decodeF32, img_decodeOriented, img_probeDimensions,
// img_readExifOrientation, img_applyExifOrientation). The bronze port kept
// only the encoders, so `bro.image` lost every decode path that was not the
// plain 8-bit one brokit supplies.
//
// Paths are used as given, exactly as encodePngFile does — a host that wants
// app-relative assets resolves before calling.

#include "host_image_internal.h"

#include <broimage/decode.h>

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace broimage::api {

namespace {

bool bytesArg(Value v, const uint8_t*& data, size_t& size) {
    ev::TypedArrayInfo info = ev::typedArrayInfo(v);
    if (!info.data) return false;
    data = info.data;
    size = info.byteLength;
    return true;
}

// { width, height, channels, pixels } — the shape every decode entry point
// returned before the transition.
Value decodeResult(int w, int h, int channels, ElementKind kind,
                   const void* data, size_t bytes, size_t elementCount) {
    ObjectBuilder obj;
    obj.set("width", static_cast<double>(w));
    obj.set("height", static_cast<double>(h));
    obj.set("channels", static_cast<double>(channels));
    ev::Persistent px(typedArrayFrom(kind, data, bytes,
                                     static_cast<uint32_t>(elementCount)));
    obj.set("pixels", px.get());
    return obj.build();
}

// decodeU16(pathOrBytes) -> { width, height, channels, pixels: Uint16Array }
// or null. 16-bit PNG depth maps keep their precision here.
Value imageDecodeU16(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("decodeU16(pathOrBytes)");
    broimage::ImageU16 out;
    std::string err;
    bool ok = false;
    if (ev::isString(args[0])) {
        ok = broimage::decode_file_u16(ev::toUtf8(args[0]), out, &err);
    } else {
        const uint8_t* p = nullptr;
        size_t n = 0;
        if (!bytesArg(args[0], p, n)) {
            return ev::throwTypeError("decodeU16: expected path string or bytes");
        }
        ok = broimage::decode_memory_u16(p, n, out, &err);
    }
    if (!ok) return ev::null();
    return decodeResult(out.width, out.height, out.channels, ev::elements::Uint16,
                        out.pixels.data(), out.pixels.size() * sizeof(uint16_t),
                        out.pixels.size());
}

// decodeF32(pathOrBytes) -> { width, height, channels, pixels: Float32Array }
// or null. The HDR / Radiance path.
Value imageDecodeF32(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("decodeF32(pathOrBytes)");
    broimage::ImageF32 out;
    std::string err;
    bool ok = false;
    if (ev::isString(args[0])) {
        ok = broimage::decode_file_f32(ev::toUtf8(args[0]), out, &err);
    } else {
        const uint8_t* p = nullptr;
        size_t n = 0;
        if (!bytesArg(args[0], p, n)) {
            return ev::throwTypeError("decodeF32: expected path string or bytes");
        }
        ok = broimage::decode_memory_f32(p, n, out, &err);
    }
    if (!ok) return ev::null();
    return decodeResult(out.width, out.height, out.channels, ev::elements::Float32,
                        out.pixels.data(), out.pixels.size() * sizeof(float),
                        out.pixels.size());
}

// decodeOriented(pathOrBytes) -> { width, height, channels, pixels: Uint8Array }
// RGBA8 with the EXIF orientation already applied, so phone JPEGs load upright.
Value imageDecodeOriented(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("decodeOriented(pathOrBytes)");
    broimage::Image out;
    std::string err;
    if (ev::isString(args[0])) {
        broimage::decode_file_oriented(ev::toUtf8(args[0]), out, &err);
    } else {
        const uint8_t* p = nullptr;
        size_t n = 0;
        if (!bytesArg(args[0], p, n)) {
            return ev::throwTypeError("decodeOriented: expected path string or bytes");
        }
        broimage::decode_memory_oriented(p, n, out, &err);
    }
    // Oriented decode has the same 1x1 fallback as decode_file, so a failure
    // still yields a usable image rather than null.
    return decodeResult(out.width, out.height, out.channels, ev::elements::Uint8,
                        out.pixels.data(), out.pixels.size(), out.pixels.size());
}

// probeDimensions(bytes) -> { width, height, channels } or null — reads the
// header only, without decoding any pixels.
Value imageProbeDimensions(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("probeDimensions(bytes)");
    const uint8_t* p = nullptr;
    size_t n = 0;
    if (!bytesArg(args[0], p, n)) {
        return ev::throwTypeError("probeDimensions: expected bytes");
    }
    int w = 0, h = 0, c = 0;
    if (!broimage::probe_dimensions_memory(p, n, &w, &h, &c)) return ev::null();
    ObjectBuilder obj;
    obj.set("width", static_cast<double>(w));
    obj.set("height", static_cast<double>(h));
    obj.set("channels", static_cast<double>(c));
    return obj.build();
}

// readExifOrientation(pathOrBytes) -> the raw EXIF orientation code (1..8;
// 1 = upright, 0 = none found).
Value imageReadExifOrientation(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("readExifOrientation(pathOrBytes)");
    broimage::ExifOrientation o{};
    if (ev::isString(args[0])) {
        o = broimage::read_exif_orientation_file(ev::toUtf8(args[0]));
    } else {
        const uint8_t* p = nullptr;
        size_t n = 0;
        if (!bytesArg(args[0], p, n)) {
            return ev::throwTypeError("readExifOrientation: expected path string or bytes");
        }
        o = broimage::read_exif_orientation(p, n);
    }
    return ev::fromDouble(static_cast<double>(static_cast<int>(o)));
}

// applyExifOrientation(pixels, w, h, orient) -> a new
// { width, height, channels, pixels } with the rotation/flip applied. The
// transposing orientations swap width and height, which is why this returns a
// fresh buffer instead of writing in place.
Value imageApplyExifOrientation(Value, std::span<const Value> args) {
    if (args.size() < 4) {
        return ev::throwTypeError("applyExifOrientation(pixels, w, h, orient)");
    }
    TypedArrayView px;
    if (!unpackTypedArray(args[0], "pixels", &px)) return ev::undefined();
    if (px.bytesPerElement != 1) {
        return ev::throwTypeError("applyExifOrientation: pixels must be a Uint8Array (RGBA8)");
    }
    const int32_t w = static_cast<int32_t>(ev::toDouble(args[1]));
    const int32_t h = static_cast<int32_t>(ev::toDouble(args[2]));
    const int32_t orient = static_cast<int32_t>(ev::toDouble(args[3]));
    if (w <= 0 || h <= 0 ||
        px.byteLength < static_cast<size_t>(w) * static_cast<size_t>(h) * 4) {
        return ev::throwRangeError("applyExifOrientation: pixels too small for w*h*4");
    }

    broimage::Image img;
    img.width = w;
    img.height = h;
    img.channels = 4;
    img.pixels.assign(px.data, px.data + static_cast<size_t>(w) * h * 4);
    broimage::apply_exif_orientation(img, static_cast<broimage::ExifOrientation>(orient));
    return decodeResult(img.width, img.height, img.channels, ev::elements::Uint8,
                        img.pixels.data(), img.pixels.size(), img.pixels.size());
}

} // namespace

void installDecodeOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("decodeU16", 1, imageDecodeU16);
    b.def("decodeF32", 1, imageDecodeF32);
    b.def("decodeOriented", 1, imageDecodeOriented);
    b.def("probeDimensions", 1, imageProbeDimensions);
    b.def("readExifOrientation", 1, imageReadExifOrientation);
    b.def("applyExifOrientation", 4, imageApplyExifOrientation);
}

} // namespace broimage::api
