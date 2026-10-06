// Bounded and animated decode (decode.h): the size of what a decode would
// produce is read from the headers and checked before anything is decoded.

#include "broimage/decode.h"

#include <stb_image.h>

#include <cstring>
#include <limits>
#include <string>

namespace broimage {

namespace {

bool is_gif(const uint8_t* data, std::size_t size) {
    return size >= 6 && std::memcmp(data, "GIF8", 4) == 0 && (data[4] == '7' || data[4] == '9') && data[5] == 'a';
}

void fail(std::string* error, const char* why, Animation& out) {
    out = Animation{};
    if (error) *error = why;
}

} // namespace

int gif_frame_count(const uint8_t* data, std::size_t size) {
    if (!data || !is_gif(data, size) || size < 13) return 0;
    std::size_t p = 13;  // header (6) + logical screen descriptor (7)
    const uint8_t screen_flags = data[10];
    if (screen_flags & 0x80) p += 3u * (1u << ((screen_flags & 7) + 1));  // global colour table
    // Sub-blocks: a size byte and that many bytes, until a zero size.
    auto skip_sub_blocks = [&](std::size_t& q) {
        while (q < size) {
            const uint8_t n = data[q++];
            if (n == 0) return true;
            q += n;
        }
        return false;
    };
    int frames = 0;
    while (p < size) {
        const uint8_t block = data[p++];
        if (block == 0x3B) break;  // trailer
        if (block == 0x21) {       // extension: label, then sub-blocks
            if (p >= size) break;
            ++p;
            if (!skip_sub_blocks(p)) break;
        } else if (block == 0x2C) {  // image descriptor
            if (p + 9 > size) break;
            const uint8_t flags = data[p + 8];
            p += 9;
            if (flags & 0x80) p += 3u * (1u << ((flags & 7) + 1));  // local colour table
            if (p >= size) break;
            ++p;  // LZW minimum code size
            if (!skip_sub_blocks(p)) break;
            ++frames;
        } else {
            break;  // not a GIF block: what follows is not read either
        }
    }
    return frames;
}

bool decode_memory_bounded(const uint8_t* data, std::size_t size, const DecodeLimits& limits,
                           Animation& out, std::string* error) {
    out = Animation{};
    if (!data || size == 0 || size > std::size_t(std::numeric_limits<int>::max())) {
        fail(error, "no data, or more than a decoder takes", out);
        return false;
    }
    int w = 0, h = 0, c = 0;
    if (!stbi_info_from_memory(data, int(size), &w, &h, &c) || w <= 0 || h <= 0) {
        fail(error, stbi_failure_reason() ? stbi_failure_reason() : "unrecognized image", out);
        return false;
    }
    if ((limits.max_width > 0 && w > limits.max_width) || (limits.max_height > 0 && h > limits.max_height)) {
        fail(error, "image dimensions exceed the limit", out);
        return false;
    }
    const bool gif = is_gif(data, size);
    const int frames = gif ? gif_frame_count(data, size) : 1;
    if (frames <= 0) {
        fail(error, "a GIF without frames", out);
        return false;
    }
    const std::size_t frame_bytes = std::size_t(w) * std::size_t(h) * 4;
    if (limits.max_bytes > 0 && (frame_bytes > limits.max_bytes || std::size_t(frames) > limits.max_bytes / frame_bytes)) {
        fail(error, "decoded size exceeds the limit", out);
        return false;
    }

    if (gif && frames > 1) {
        int* delays = nullptr;
        int gw = 0, gh = 0, gz = 0, comp = 0;
        stbi_uc* px = stbi_load_gif_from_memory(data, int(size), &delays, &gw, &gh, &gz, &comp, 4);
        if (!px || gw != w || gh != h || gz <= 0 || gz > frames) {
            // stb decoding more frames than the block walk counted would mean
            // the bound was not the bound: refuse rather than trust it.
            if (px) stbi_image_free(px);
            if (delays) stbi_image_free(delays);
            fail(error, stbi_failure_reason() ? stbi_failure_reason() : "GIF decode failed", out);
            return false;
        }
        out.width = w;
        out.height = h;
        out.frames.resize(std::size_t(gz));
        for (int i = 0; i < gz; ++i) {
            const stbi_uc* f = px + std::size_t(i) * frame_bytes;
            out.frames[std::size_t(i)].rgba.assign(f, f + frame_bytes);
            out.frames[std::size_t(i)].delay_ms = delays ? delays[i] : 0;
        }
        stbi_image_free(px);
        if (delays) stbi_image_free(delays);
        return true;
    }

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
