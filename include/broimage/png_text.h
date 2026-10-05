#pragma once

// PNG textual metadata: tEXt, zTXt and iTXt chunks.
//
// These carry key/value tags beside the pixels; the Freedesktop thumbnail
// spec stores Thumb::URI / Thumb::MTime / Thumb::Size in them, and most tools
// write a "Software" tag. stb_image ignores them on decode and stb_image_write
// cannot emit them, so this is the one place in the stack that reads and
// writes them.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace broimage {

struct PngTextEntry {
    // 1-79 bytes of printable Latin-1 (PNG spec 11.3.4.2).
    std::string keyword;
    // tEXt / zTXt: the stored Latin-1 bytes. iTXt: UTF-8.
    std::string text;
};

struct PngInfo {
    int width = 0;
    int height = 0;
    int bit_depth = 0;   // IHDR bit depth (1, 2, 4, 8, 16)
    int color_type = 0;  // IHDR colour type (0 gray, 2 RGB, 3 palette, 4 gray+alpha, 6 RGBA)
    std::vector<PngTextEntry> text; // every text chunk, in file order
};

// Reads the header and every text chunk of an in-memory PNG without decoding
// pixels. zTXt and compressed iTXt are inflated. A text chunk with a bad CRC
// or a malformed body is skipped; a truncated tail ends the walk with what was
// read so far. Returns false (with `error`) only when the signature or IHDR is
// missing or invalid.
bool read_png_info(const uint8_t* data, std::size_t size, PngInfo& out,
                   std::string* error = nullptr);

// encode_png_memory plus one text chunk per entry, placed right after IHDR in
// order. Text that is 7-bit ASCII is written as tEXt, anything else as an
// uncompressed iTXt (UTF-8). Returns false on an invalid keyword (empty,
// longer than 79 bytes, or a byte outside printable Latin-1) or when encoding
// fails.
bool encode_png_memory_with_text(std::vector<uint8_t>& out,
                                 const uint8_t* pixels, int width, int height, int channels,
                                 const std::vector<PngTextEntry>& text,
                                 int stride_bytes = 0);

} // namespace broimage
