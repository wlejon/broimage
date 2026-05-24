#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace broimage {

// PNG: lossless RGBA. Stride defaults to width*channels (tightly packed).
bool encode_png_file(const std::string& path,
                     const uint8_t* pixels, int width, int height, int channels,
                     int stride_bytes = 0);

bool encode_png_memory(std::vector<uint8_t>& out,
                       const uint8_t* pixels, int width, int height, int channels,
                       int stride_bytes = 0);

// JPEG: lossy. `quality` in [1, 100]. RGB or grayscale (channels == 1 or 3).
// Pass 4-channel RGBA via `encode_jpeg_file` only if you've already
// flattened the alpha; stb writes RGBA-as-RGB but the result is
// surprising for non-opaque inputs.
bool encode_jpeg_file(const std::string& path,
                      const uint8_t* pixels, int width, int height, int channels,
                      int quality = 90);

bool encode_jpeg_memory(std::vector<uint8_t>& out,
                        const uint8_t* pixels, int width, int height, int channels,
                        int quality = 90);

} // namespace broimage
