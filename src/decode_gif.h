#pragma once
// Internal: GIF, decoded a frame at a time (codec.h's FrameDecoder).
//
// The decoder keeps the composed canvas, the indices of the frame being
// drawn, and — for a frame whose disposal is "restore to previous" — a copy
// of the region it covers; nothing of the frames already shown. Disposal 0/1
// leaves a frame in place, 2 clears its rectangle to transparent (as
// browsers do, rather than to the logical screen's background colour), 3
// restores what was under it. Local colour tables, the transparent index,
// interlaced frames, frames that hang off the canvas and a logical screen of
// 0 x 0 (sized to its frames) are all read. A frame whose data ends before
// its block terminator is an error.

#include "broimage/codec.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace broimage::detail {

bool is_gif(const uint8_t* data, std::size_t size);

// The block structure, walked without decoding: canvas size, frames, loop
// count and per-frame delays.
struct GifScan {
    int width = 0;
    int height = 0;
    int frames = 0;
    int loop_count = 1;  // FrameDecoder::loop_count
    std::vector<int> delays_ms;
};
bool gif_scan(const uint8_t* data, std::size_t size, GifScan& out);

// A frame decoder over `data`, which must outlive it unless `keep` owns it.
std::unique_ptr<FrameDecoder> gif_open(const uint8_t* data, std::size_t size, SharedBytes keep,
                                       std::string* why);

} // namespace broimage::detail
