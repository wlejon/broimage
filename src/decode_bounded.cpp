// Bounded and animated decode (decode.h): the size of what a decode would
// produce is read from the headers and checked before anything is decoded.

#include "broimage/decode.h"
#include "broimage/codec.h"

#include "decode_gif.h"
#include "decode_ico.h"
#include "decode_tiff.h"

#include <stb_image.h>

#include <cstring>
#include <limits>
#include <memory>
#include <string>

namespace broimage {

namespace {

void fail(std::string* error, const char* why, Animation& out) {
    out = Animation{};
    if (error) *error = why;
}

} // namespace

int gif_frame_count(const uint8_t* data, std::size_t size) {
    detail::GifScan s;
    if (!data || !detail::gif_scan(data, size, s)) return 0;
    return s.frames;
}

bool decode_memory_bounded(const uint8_t* data, std::size_t size, const DecodeLimits& limits,
                           Animation& out, std::string* error) {
    out = Animation{};
    if (!data || size == 0 || size > std::size_t(std::numeric_limits<int>::max())) {
        fail(error, "no data, or more than a decoder takes", out);
        return false;
    }
    auto within = [&](uint64_t w, uint64_t h, uint64_t frames) {
        if ((limits.max_width > 0 && w > uint64_t(limits.max_width)) ||
            (limits.max_height > 0 && h > uint64_t(limits.max_height))) {
            fail(error, "image dimensions exceed the limit", out);
            return false;
        }
        const uint64_t bytes = w * h * 4;
        if (limits.max_bytes > 0 && (bytes > limits.max_bytes || frames > limits.max_bytes / bytes)) {
            fail(error, "decoded size exceeds the limit", out);
            return false;
        }
        return true;
    };

    // Animations — a GIF, or a registered format with frames (bro: WebP) —
    // a frame at a time: the frame count from the headers is the bound, and
    // the decoder holds one canvas, not every frame.
    const Codec* reg = find_registered_codec(data, size);
    std::unique_ptr<FrameDecoder> frames;
    std::string why;
    if (reg && reg->open_frames)
        frames = reg->open_frames(std::make_shared<const std::vector<uint8_t>>(data, data + size), &why);
    else if (detail::is_gif(data, size))
        frames = detail::gif_open(data, size, nullptr, &why);
    if (frames) {
        const int count = frames->frame_count();
        if (count <= 0) {
            fail(error, "an animation without frames", out);
            return false;
        }
        if (!within(uint64_t(frames->width()), uint64_t(frames->height()), uint64_t(count))) return false;
        out.width = frames->width();
        out.height = frames->height();
        out.frames.reserve(std::size_t(count));
        AnimationFrame f;
        while (int(out.frames.size()) < count && frames->next(f, &why)) out.frames.push_back(std::move(f));
        if (out.frames.empty()) {
            fail(error, why.empty() ? "the animation decoded no frame" : why.c_str(), out);
            return false;
        }
        return true;
    }
    if (detail::is_gif(data, size) || (reg && reg->open_frames)) {
        fail(error, why.empty() ? "the animation could not be read" : why.c_str(), out);
        return false;
    }

    if (reg && reg->probe) {
        int w = 0, h = 0, c = 0;
        if (reg->probe(data, size, w, h, c) && w > 0 && h > 0) {
            if (!within(uint64_t(w), uint64_t(h), 1)) return false;
            Image img;
            if (reg->decode(data, size, img, &why) && img.width == w && img.height == h) {
                out.width = w;
                out.height = h;
                out.frames.resize(1);
                out.frames[0].rgba = std::move(img.pixels);
                return true;
            }
        }
    }

    if (detail::is_tiff(data, size)) {
        detail::TiffLayout l;
        if (!detail::tiff_layout(data, size, l, &why)) {
            fail(error, why.c_str(), out);
            return false;
        }
        if (!within(l.width, l.height, 1)) return false;
        out.frames.resize(1);
        if (!detail::tiff_decode(data, size, l, out.frames[0].rgba, &why)) {
            fail(error, why.c_str(), out);
            return false;
        }
        out.width = int(l.width);
        out.height = int(l.height);
        return true;
    }

    if (int iw = 0, ih = 0; detail::ico_probe(data, size, iw, ih)) {
        if (!within(uint64_t(iw), uint64_t(ih), 1)) return false;
        out.frames.resize(1);
        if (!detail::ico_decode(data, size, iw, ih, out.frames[0].rgba)) {
            fail(error, "ICO image could not be decoded", out);
            return false;
        }
        out.width = iw;
        out.height = ih;
        return true;
    }

    int w = 0, h = 0, c = 0;
    if (!stbi_info_from_memory(data, int(size), &w, &h, &c) || w <= 0 || h <= 0) {
        fail(error, stbi_failure_reason() ? stbi_failure_reason() : "unrecognized image", out);
        return false;
    }
    if (!within(uint64_t(w), uint64_t(h), 1)) return false;
    const std::size_t frame_bytes = std::size_t(w) * std::size_t(h) * 4;
    int dw = 0, dh = 0, dc = 0;
    stbi_uc* px = stbi_load_from_memory(data, int(size), &dw, &dh, &dc, 4);
    if (!px || dw != w || dh != h) {
        if (px) stbi_image_free(px);
        fail(error, stbi_failure_reason() ? stbi_failure_reason() : "decode failed", out);
        return false;
    }
    out.width = w;
    out.height = h;
    out.frames.resize(1);
    out.frames[0].rgba.assign(px, px + frame_bytes);
    stbi_image_free(px);
    return true;
}

} // namespace broimage
