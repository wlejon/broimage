// HEIF (AVIF / HEIC) and decoder capability entry points:
//
//   canDecode(type)          -> '' | 'probably'   (like bro.media.canDecode)
//   decodeSupport(type)      -> { supported, reason }
//   readHeifInfo(src)        -> the container's description, or null
//   decodeThumbnail(src, minSide?) -> { width, height, channels, pixels } or null
//
// `src` is a path or bytes, as decodeOriented takes.

#include "host_image_internal.h"

#include "file_bytes.h"

#include <broimage/heif.h>

#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace broimage::api {

namespace {

// The bytes of a path or a typed array; false (with a TypeError thrown) when
// the argument is neither. A path that cannot be read gives no bytes.
bool sourceBytes(Value v, const char* who, std::vector<uint8_t>& owned, const uint8_t*& p, size_t& n) {
    if (ev::isString(v)) {
        detail::read_file_bytes(readPath(ev::toUtf8(v)), owned);
        p = owned.data();
        n = owned.size();
        return true;
    }
    ev::TypedArrayInfo info = ev::typedArrayInfo(v);
    if (!info.data) {
        ev::throwTypeError(std::string(who) + ": expected path string or bytes");
        return false;
    }
    p = info.data;
    n = info.byteLength;
    return true;
}

Value imageCanDecode(Value, std::span<const Value> args) {
    if (args.empty() || !ev::isString(args[0])) return ev::throwTypeError("canDecode(type)");
    return ev::fromUtf8(broimage::can_decode(ev::toUtf8(args[0])) ? "probably" : "");
}

Value imageDecodeSupport(Value, std::span<const Value> args) {
    if (args.empty() || !ev::isString(args[0])) return ev::throwTypeError("decodeSupport(type)");
    std::string why;
    const bool ok = broimage::can_decode(ev::toUtf8(args[0]), &why);
    ObjectBuilder obj;
    obj.set("supported", ok);
    obj.set("reason", why);
    return obj.build();
}

Value imageReadHeifInfo(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("readHeifInfo(pathOrBytes)");
    std::vector<uint8_t> owned;
    const uint8_t* p = nullptr;
    size_t n = 0;
    if (!sourceBytes(args[0], "readHeifInfo", owned, p, n)) return ev::undefined();
    broimage::HeifInfo info;
    if (!broimage::read_heif_info(p, n, info)) return ev::null();
    ObjectBuilder obj;
    obj.set("mime", info.mime);
    obj.set("codec", info.codec);
    obj.set("majorBrand", info.major_brand);
    obj.set("width", double(info.width));
    obj.set("height", double(info.height));
    obj.set("bitDepth", double(info.bit_depth));
    obj.set("hasAlpha", info.has_alpha);
    obj.set("grid", info.grid);
    if (info.grid) {
        obj.set("gridRows", double(info.grid_rows));
        obj.set("gridColumns", double(info.grid_columns));
    }
    obj.set("orientation", double(int(info.orientation)));
    obj.set("hasExif", info.has_exif);
    obj.set("hasIcc", info.has_icc);
    obj.set("colorPrimaries", double(info.primaries));
    obj.set("transferCharacteristics", double(info.transfer));
    obj.set("matrixCoefficients", double(info.matrix));
    obj.set("fullRange", info.full_range);
    obj.set("hdr", info.hdr);
    obj.set("sequence", info.sequence);
    const auto thumbs = info.thumbnails;
    obj.set("thumbnails", hostArrayOf(thumbs.size(), [&](size_t i) -> Value {
        ObjectBuilder t;
        t.set("width", double(thumbs[i].width));
        t.set("height", double(thumbs[i].height));
        t.set("codec", thumbs[i].codec);
        return t.get();
    }));
    return obj.build();
}

Value imageDecodeThumbnail(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("decodeThumbnail(pathOrBytes, minSide?)");
    int minSide = 0;
    if (args.size() > 1 && ev::isNumber(args[1])) {
        const double d = ev::toDouble(args[1]);
        if (std::isfinite(d) && d > 0) minSide = d > 65535 ? 65535 : int(d);
    }
    std::vector<uint8_t> owned;
    const uint8_t* p = nullptr;
    size_t n = 0;
    if (!sourceBytes(args[0], "decodeThumbnail", owned, p, n)) return ev::undefined();
    broimage::Image img;
    if (!broimage::decode_heif_thumbnail(p, n, minSide, img)) return ev::null();
    ObjectBuilder obj;
    obj.set("width", double(img.width));
    obj.set("height", double(img.height));
    obj.set("channels", 4.0);
    ev::Persistent px(typedArrayFrom(ev::elements::Uint8, img.pixels.data(), img.pixels.size(),
                                     static_cast<uint32_t>(img.pixels.size())));
    obj.set("pixels", px.get());
    return obj.build();
}

} // namespace

void installHeifOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("canDecode", 1, imageCanDecode);
    b.def("decodeSupport", 1, imageDecodeSupport);
    b.def("readHeifInfo", 1, imageReadHeifInfo);
    b.def("decodeThumbnail", 2, imageDecodeThumbnail);
}

} // namespace broimage::api
