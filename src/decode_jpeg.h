#pragma once
// Internal: JPEG through libjpeg-turbo, at full size or DCT-scaled.
//
// Built without libjpeg-turbo (BROIMAGE_WITH_JPEG_TURBO=OFF) jpeg_turbo()
// is false and jpeg_decode() always fails, so JPEG goes to stb_image.

#include "broimage/buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace broimage::detail {

// Whether this build decodes JPEG with libjpeg-turbo.
bool jpeg_turbo();

bool is_jpeg(const uint8_t* data, std::size_t size);

// The frame's stored size (before any EXIF orientation), from its SOF.
bool jpeg_probe(const uint8_t* data, std::size_t size, int& width, int& height, int& channels);

// What decoding at `scale`/8 (1..8) gives: libjpeg rounds each side up.
inline int jpeg_scaled_size(int full, int scale) { return int((int64_t(full) * scale + 7) / 8); }

// Decode at `scale`/8 of the stored size into RGBA8, as stored (no EXIF
// orientation). Gray expands to RGB; CMYK and YCCK (Adobe's inverted ink)
// become RGB as R = C*K/255, as stb_image reads them.
bool jpeg_decode(const uint8_t* data, std::size_t size, int scale, Image& out, std::string* why);

} // namespace broimage::detail
