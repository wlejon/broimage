#pragma once

#include <cstdint>
#include <vector>

namespace broimage {

// An RGBA8 host-side image. Used as the decode/encode currency: anything
// loaded from a file or about to be written to one travels through this.
// Geometric and color ops operate on raw pointer + dims rather than this
// struct so they compose with caller-supplied buffers (a tile in a larger
// atlas, a slice of a video frame).
struct Image {
    int width  = 0;
    int height = 0;
    int channels = 4;                // 1 (gray), 3 (RGB), 4 (RGBA). Decode always emits 4.
    std::vector<uint8_t> pixels;     // size == width * height * channels

    bool empty() const { return pixels.empty(); }
};

} // namespace broimage
