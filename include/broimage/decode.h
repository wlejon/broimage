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

// ----- EXIF orientation ------------------------------------------------------
//
// EXIF orientation tag values (1..8). Camera phones routinely set this — the
// raw pixels are in sensor order and the displayed image only matches
// expectations after the indicated rotate/flip is applied. stb_image does not
// honor it, so loading a phone photo without `decode_file_oriented` will
// render sideways most of the time.
enum class ExifOrientation {
    Normal       = 1, // no transform
    FlipH        = 2, // mirror left<->right
    Rotate180    = 3,
    FlipV        = 4, // mirror top<->bottom
    Transpose    = 5, // flip across top-left/bottom-right diagonal
    Rotate90CW   = 6,
    Transverse   = 7, // flip across top-right/bottom-left diagonal
    Rotate90CCW  = 8,
};

// Inspect a JPEG byte buffer for an EXIF Orientation tag. Returns Normal if
// the buffer is not JPEG, has no EXIF block, or the tag is missing/invalid.
// Cheap — only scans the APP1 segment, doesn't touch pixels.
ExifOrientation read_exif_orientation(const uint8_t* data, std::size_t size);

// As above, by file path. Reads only the first few KB.
ExifOrientation read_exif_orientation_file(const std::string& path);

// Decode + auto-orient. Same fallback semantics as decode_file/decode_memory
// on failure. Non-JPEG inputs decode as usual (orientation tag is JPEG-only
// in practice; TIFF has it too but stb doesn't decode TIFF).
bool decode_file_oriented(const std::string& path, Image& out,
                          std::string* error = nullptr);
bool decode_memory_oriented(const uint8_t* data, std::size_t size, Image& out,
                            std::string* error = nullptr);

// Apply an EXIF orientation transform to an in-memory RGBA8 image in place.
// No-op for Normal. Exposed so callers that already have a decoded buffer
// can apply the orientation themselves.
void apply_exif_orientation(Image& img, ExifOrientation orient);

} // namespace broimage
