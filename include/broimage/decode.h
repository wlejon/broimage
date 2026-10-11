#pragma once

#include "broimage/buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace broimage {

// Decode an image file into an RGBA8 buffer: JPEG via libjpeg-turbo (stb_image
// in a build without it); PNG, BMP, TGA, PSD, PIC, PNM and HDR via stb_image;
// ICO/CUR, TIFF, GIF (the first frame) and HEIF
// (AVIF through a host's AV1 decoder, HEIC through the OS: heif.h) by
// broimage's own decoders; a Canon CR3 as its embedded full-size JPEG (the
// RAW data itself is not decoded); and whatever the host registered (codec.h: bro
// adds WebP and SVG), tried first. TIFF reads page 0: uncompressed, LZW (+ horizontal
// predictor), Deflate, PackBits and JPEG compression; 1/2/4/8/16-bit
// samples; gray, RGB(A), palette, CMYK and JPEG YCbCr; strips or tiles;
// planar configuration 1 or 2 (src/decode_tiff.h has the details). Returns
// true on success. On failure `out` is emptied (0 x 0) and `error` (if
// non-null) receives the reason.
//
// The pixels are as stored: EXIF orientation is applied only by the
// *_oriented variants below.
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
// decoded when they exceed the limits. Every frame of an animated GIF (or a
// registered animated format, codec.h) is decoded, composed onto the canvas,
// a frame at a time; any other format yields its one frame. Every format decode_memory reads is read here,
// TIFF included (what chafa sends as an iTerm2 inline image); a TIFF's
// size comes from its first IFD before any strip or tile is decompressed.

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

// The EXIF Orientation tag of a JPEG (APP1), TIFF (its IFD0), WebP (EXIF
// chunk) or PNG (eXIf chunk), read by read_exif (exif.h). Returns Normal
// when there is no EXIF block or the tag is missing/invalid. Doesn't touch
// pixels.
ExifOrientation read_exif_orientation(const uint8_t* data, std::size_t size);

// As above, by file path: a JPEG's head, any other file whole.
ExifOrientation read_exif_orientation_file(const std::string& path);

// Decode + auto-orient: decode_file/decode_memory, then the orientation
// read_exif_orientation finds, in any container it reads. Same failure
// semantics as decode_file/decode_memory.
bool decode_file_oriented(const std::string& path, Image& out,
                          std::string* error = nullptr);
bool decode_memory_oriented(const uint8_t* data, std::size_t size, Image& out,
                            std::string* error = nullptr);

// Apply an EXIF orientation transform to an in-memory RGBA8 image in place.
// No-op for Normal. Exposed so callers that already have a decoded buffer
// can apply the orientation themselves.
void apply_exif_orientation(Image& img, ExifOrientation orient);

// ----- Scaled decode ---------------------------------------------------------
//
// An image decoded for a box: a thumbnail, or a photo shown on a screen
// smaller than it. The result is upright (EXIF orientation applied, as
// decode_memory_oriented does), keeps the aspect ratio, and is the size the
// box asks for, never larger than the image itself.
//
// It costs what that size needs rather than what the file holds. A JPEG
// decodes through libjpeg-turbo's scaled IDCT at the smallest of 1/8, 2/8,
// ... 8/8 whose result still covers the target, then an area resize takes it
// to the target: a 256 px thumbnail of a 24 MP photo decodes at 1/8. When the
// target is no bigger than a JPEG's EXIF thumbnail (decode_exif_thumbnail)
// that is used instead, and a HEIF's thumbnail item likewise. A Canon CR3 is
// never RAW-decoded: the smallest of its embedded JPEGs that covers the target
// (the 160 x 120 THMB, the 1620 x 1080 PRVW, the full-size one) is, scaled the
// same way. Any other image is decoded whole and resized.

enum class ScaleFit {
    Contain,  // the largest size that fits inside the box (a thumbnail, a photo on a screen)
    Cover,    // the smallest size that covers the box: at least width x height
};

struct ScaledDecodeOptions {
    // The box, in upright pixels. 0 leaves that side unconstrained; both 0 is
    // the image's own size.
    int width = 0;
    int height = 0;
    ScaleFit fit = ScaleFit::Contain;
    // Whether an embedded thumbnail (a JPEG's EXIF one, a HEIF's thumbnail
    // item) may stand in when it is at least the target size.
    bool embedded_thumbnail = true;
};

enum class ScaledDecodePath {
    Full,           // decoded whole (any format, or a JPEG asked for its full size)
    Dct,            // a JPEG decoded at dct_scale/8 by libjpeg-turbo
    ExifThumbnail,  // the JPEG's EXIF thumbnail
    HeifThumbnail,  // the HEIF's thumbnail item
    RawPreview,     // one of a RAW file's embedded JPEGs (CR3: THMB, PRVW or the full-size one)
};

struct ScaledDecodeInfo {
    int source_width = 0;   // the image's own size, upright
    int source_height = 0;
    int decoded_width = 0;  // what the decoder produced, upright, before the resize
    int decoded_height = 0;
    ScaledDecodePath path = ScaledDecodePath::Full;
    int dct_scale = 8;      // the M of M/8 (Dct only; 8 otherwise)
};

// False (and `out` empty, `error` set) when the image does not decode.
bool decode_memory_scaled(const uint8_t* data, std::size_t size, const ScaledDecodeOptions& options,
                          Image& out, std::string* error = nullptr, ScaledDecodeInfo* info = nullptr);
bool decode_file_scaled(const std::string& path, const ScaledDecodeOptions& options, Image& out,
                        std::string* error = nullptr, ScaledDecodeInfo* info = nullptr);

// The thumbnail a JPEG's EXIF block carries (IFD1's JPEGInterchangeFormat:
// 160 x 120 from most cameras), decoded, upright by the main image's
// orientation, and cropped to the main image's aspect ratio: a camera pads a
// 3:2 photo's thumbnail to 4:3 with black bars, which are cut off. A Canon
// CR3's THMB box (also 160 x 120) likewise. False when there is none or it
// does not decode.
bool decode_exif_thumbnail(const uint8_t* data, std::size_t size, Image& out,
                           std::string* error = nullptr);
// As above from a file, reading only its head (the first 1 MB at most).
bool decode_file_exif_thumbnail(const std::string& path, Image& out, std::string* error = nullptr);

} // namespace broimage
