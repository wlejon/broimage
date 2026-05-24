#pragma once

#include "broimage/buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace broimage {

// Decode an image file (PNG, JPEG, BMP, TGA, GIF, PSD, PIC, PNM via stb_image)
// into an RGBA8 buffer. Returns true on success. On failure `out` is set to a
// 1x1 white pixel (matches the bro Image fallback) and `error` (if non-null)
// receives stb's failure reason.
bool decode_file(const std::string& path, Image& out, std::string* error = nullptr);

// Decode an image from an in-memory byte buffer (e.g. a fetched response body).
bool decode_memory(const uint8_t* data, std::size_t size, Image& out,
                   std::string* error = nullptr);

} // namespace broimage
