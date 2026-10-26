// Animation frames (codec.h): decodeFrames, the whole animation at once under
// a cap, and openFrames, a decoder stepped a frame at a time so a long
// animation never has to be held as RGBA. Both read every format decode does
// (a still image is one frame) — an animated GIF, and in bro an animated WebP.

#include "host_image_internal.h"

#include <broimage/codec.h>

#include "../file_bytes.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace broimage::api {

namespace {

// The bytes behind `src` (a path or a typed array), copied: a decoder keeps
// them, and a typed array's storage moves with the heap.
broimage::SharedBytes sourceBytes(Value src, const char* fn, bool& threw) {
    threw = false;
    if (ev::isString(src)) {
        auto bytes = std::make_shared<std::vector<uint8_t>>();
        if (!broimage::detail::read_file_bytes(readPath(ev::toUtf8(src)), *bytes)) return nullptr;
        return bytes;
    }
    ev::TypedArrayInfo info = ev::typedArrayInfo(src);
    if (!info.data) {
        threw = true;
        ev::throwTypeError(std::string(fn) + ": expected a path string or bytes");
        return nullptr;
    }
    return std::make_shared<const std::vector<uint8_t>>(info.data, info.data + info.byteLength);
}

double numberOption(Value opts, const char* key) {
    if (!ev::isObject(opts)) return 0;
    Value v = ev::getProperty(opts, key);
    return ev::isNumber(v) ? ev::toDouble(v) : 0;
}

Value delaysArray(const std::vector<int>& delays) {
    ev::Persistent arr(ev::makeArray(0));
    for (std::size_t i = 0; i < delays.size(); ++i) {
        // The number first: it may allocate, and arr.get() must be read after.
        ev::Persistent v(ev::fromDouble(delays[i]));
        arr.set(ev::setElement(arr.get(), uint32_t(i), v.get()));
    }
    return arr.get();
}

Value frameObject(const broimage::AnimationFrame& f, int index) {
    ObjectBuilder o;
    ev::Persistent px(typedArrayFrom(ev::elements::Uint8, f.rgba.data(), f.rgba.size(), uint32_t(f.rgba.size())));
    o.set("pixels", px.get());
    o.set("delay", double(f.delay_ms));
    o.set("index", double(index));
    return o.build();
}

// decodeFrames(src, { maxFrames, maxBytes }?) -> { width, height, frameCount,
// loopCount, truncated, frames: [{ pixels, delay, index }] } or null.
Value imageDecodeFrames(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("decodeFrames(pathOrBytes, options?)");
    ev::Persistent src(args[0]);
    ev::Persistent opts(args.size() > 1 ? args[1] : ev::undefined());
    const double maxFrames = numberOption(opts.get(), "maxFrames");
    const double maxBytes = numberOption(opts.get(), "maxBytes");
    bool threw = false;
    broimage::SharedBytes bytes = sourceBytes(src.get(), "decodeFrames", threw);
    if (threw) return ev::undefined();
    if (!bytes) return ev::null();
    broimage::DecodedFrames df;
    if (!broimage::decode_frames(bytes->data(), bytes->size(), df, maxFrames > 0 ? int(maxFrames) : 0,
                                 maxBytes > 0 ? std::size_t(maxBytes) : 0))
        return ev::null();
    ObjectBuilder out;
    out.set("width", double(df.width));
    out.set("height", double(df.height));
    out.set("frameCount", double(df.frame_count));
    out.set("loopCount", double(df.loop_count));
    out.set("truncated", df.truncated);
    ev::Persistent frames(ev::makeArray(0));
    for (std::size_t i = 0; i < df.frames.size(); ++i) {
        ev::Persistent f(frameObject(df.frames[i], int(i)));
        frames.set(ev::setElement(frames.get(), uint32_t(i), f.get()));
        df.frames[i].rgba = {};  // copied into the heap: let it go
    }
    out.set("frames", frames.get());
    return out.build();
}

struct FramesState {
    std::unique_ptr<broimage::FrameDecoder> dec;
};

// openFrames(src) -> { width, height, frameCount, loopCount, delays, index,
// next(), reset(), close() } or null.
Value imageOpenFrames(Value, std::span<const Value> args) {
    if (args.empty()) return ev::throwTypeError("openFrames(pathOrBytes)");
    bool threw = false;
    broimage::SharedBytes bytes = sourceBytes(args[0], "openFrames", threw);
    if (threw) return ev::undefined();
    if (!bytes) return ev::null();
    auto st = std::make_shared<FramesState>();
    st->dec = broimage::open_frames(std::move(bytes));
    if (!st->dec) return ev::null();

    ObjectBuilder o;
    o.set("width", double(st->dec->width()));
    o.set("height", double(st->dec->height()));
    o.set("frameCount", double(st->dec->frame_count()));
    o.set("loopCount", double(st->dec->loop_count()));
    ev::Persistent delays(delaysArray(st->dec->delays_ms()));
    o.set("delays", delays.get());
    o.accessor("index", [st](Value, std::span<const Value>) {
        return ev::fromDouble(st->dec ? st->dec->index() : 0);
    });
    o.def("next", 0, [st](Value, std::span<const Value>) -> Value {
        if (!st->dec) return ev::null();
        broimage::AnimationFrame f;
        if (!st->dec->next(f)) return ev::null();
        return frameObject(f, st->dec->index() - 1);
    });
    o.def("reset", 0, [st](Value, std::span<const Value>) {
        return ev::fromBool(st->dec && st->dec->rewind());
    });
    o.def("close", 0, [st](Value, std::span<const Value>) {
        st->dec.reset();
        return ev::undefined();
    });
    return o.build();
}

} // namespace

void installFramesOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("decodeFrames", 2, imageDecodeFrames);
    b.def("openFrames", 1, imageOpenFrames);
}

} // namespace broimage::api
