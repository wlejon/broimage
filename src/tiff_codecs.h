#pragma once
// Internal: the compressions a TIFF chunk (a strip or a tile) is stored in.
// Each decompressor writes exactly `expected` bytes into `dst` and returns
// false when the stream is malformed or ends before `expected` bytes came
// out; bytes the stream holds past `expected` are ignored. Nothing here
// allocates from a size the stream claims.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace broimage::detail {

// Compression tag values.
enum TiffCompression : uint32_t {
    kTiffNone = 1, kTiffLzw = 5, kTiffOldJpeg = 6, kTiffJpeg = 7, kTiffAdobeDeflate = 8,
    kTiffPackBits = 32773, kTiffDeflate = 32946,
};

// LZW as TIFF 6.0 specifies it: MSB-first codes of 9..12 bits, the width
// growing one code early. The pre-6.0 bit-reversed variant is refused.
bool tiff_lzw(const uint8_t* src, std::size_t n, uint8_t* dst, std::size_t expected);

// Apple PackBits run-length coding.
bool tiff_packbits(const uint8_t* src, std::size_t n, uint8_t* dst, std::size_t expected);

// zlib-wrapped Deflate (compression 8 and 32946), through stb_image's inflate.
bool tiff_inflate(const uint8_t* src, std::size_t n, uint8_t* dst, std::size_t expected);

// JPEG (compression 7): one chunk is a JPEG stream whose tables may live in
// the IFD's JPEGTables instead (an abbreviated stream); the two are spliced
// into one interchange stream and decoded by stb_image. `rgb` marks a
// chunk stored as RGB rather than YCbCr (photometric 2), which an Adobe
// marker tells the decoder. `channels` (1 or 3) is what comes out; the
// decoded size is in `w` / `h`.
bool tiff_jpeg(const uint8_t* tables, std::size_t tables_n, const uint8_t* src, std::size_t n, bool rgb,
               int channels, std::vector<uint8_t>& out, int& w, int& h);

} // namespace broimage::detail
