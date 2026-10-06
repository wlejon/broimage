#pragma once

#include "broimage/buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace broimage {

// Decode an image file (PNG, JPEG, BMP, TGA, GIF, PSD, PIC, PNM via stb_image)
// into an RGBA8 buffer. Returns true on success. On failure `out` is set to a
// 1x1 white pixel (matches the bro Image fallback) and `error` (if non-null)
// receives stb's failure reason.
bool decode_file(const std::string& path, Image& out, std::string* error = nullptr);

// Decode an image from an in-memory byte buffer (e.g. a fetched response body).
bool decode_memory(const uint8_t* data, std::size_t size, Image& out,
                   std::string* error = nullptr);

// Read just the dimensions of an encoded image without decoding pixels — cheap
// header probe (a few hundred bytes) for sizing image elements or laying out
// before the full decode lands. `channels` reports the encoded channel count
// (1 / 3 / 4); `decode_memory` will always emit RGBA regardless. Returns false
// (and leaves outputs untouched) on an unrecognized or truncated header.
bool probe_dimensions_memory(const uint8_t* data, std::size_t size,
                             int* width, int* height, int* channels);

// ----- Bounded and animated decode ------------------------------------------
//
// For untrusted input whose size the caller must cap before paying for it (a
// terminal decoding what a program sent): the dimensions, the frame count
// and so the decoded size are read from the headers first, and nothing is
// decoded when they exceed the limits. Every frame of an animated GIF is
// decoded (composed onto the canvas, as stb_image does); any other format
// yields its one frame. Besides stb_image's formats this reads baseline
// uncompressed TIFF (8-bit RGB, RGBA, gray, gray + alpha; strips, either
// byte order), which is what chafa sends as an iTerm2 inline image.

struct DecodeLimits {
    int max_width  = 0;          // 0: no limit
    int max_height = 0;
    std::size_t max_bytes = 0;   // RGBA8 of all frames together; 0: no limit
};

struct AnimationFrame {
    std::vector<uint8_t> rgba;   // width * height * 4
    int delay_ms = 0;            // how long it shows (GIF: as stored; 0 = unspecified)
};

struct Animation {
    int width  = 0;
    int height = 0;
    std::vector<AnimationFrame> frames;
};

// The number of frames a GIF holds (its image descriptors), from its block
// structure without decoding anything; 0 when `data` is not a well-formed GIF.
int gif_frame_count(const uint8_t* data, std::size_t size);

// Decode `data` within `limits`. False (and `out` empty) on a decode
// failure or when the image would exceed a limit; `error` says which.
bool decode_memory_bounded(const uint8_t* data, std::size_t size, const DecodeLimits& limits,
                           Animation& out, std::string* error = nullptr);

// ----- High-bit-depth / HDR --------------------------------------------------
//
// 16-bit and float decode for formats that carry more than 8 bits per channel
// — most importantly 16-bit PNGs (depth maps, masks for diffusion) and Radiance
// .hdr / .pic files. stb_image supports both: stbi_load_16 / stbi_loadf. The
// channel count is forced to 4 to match decode_file's RGBA convention.
//
// On failure the output `pixels`/`width`/`height` are cleared and the error
// reason (if non-null) is set; there is no 1-pixel fallback for these — the
// 8-bit fallback exists to match a bro JS contract that doesn't apply here.

struct ImageU16 {
    int width  = 0;
    int height = 0;
    int channels = 4;
    std::vector<uint16_t> pixels;
    bool empty() const { return pixels.empty(); }
};

struct ImageF32 {
    int width  = 0;
    int height = 0;
    int channels = 4;
    std::vector<float> pixels;
    bool empty() const { return pixels.empty(); }
};

bool decode_file_u16(const std::string& path, ImageU16& out,
                     std::string* error = nullptr);
bool decode_memory_u16(const uint8_t* data, std::size_t size, ImageU16& out,
                       std::string* error = nullptr);

bool decode_file_f32(const std::string& path, ImageF32& out,
                     std::string* error = nullptr);
bool decode_memory_f32(const uint8_t* data, std::size_t size, ImageF32& out,
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
