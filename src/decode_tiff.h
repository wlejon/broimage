#pragma once
// Internal: the TIFF image decoder. Reads the first page (IFD0):
//
//   compression   none, LZW (with horizontal predictor 2), Deflate (8 and
//                 32946), PackBits, JPEG (7, with or without JPEGTables;
//                 8-bit YCbCr, RGB or gray, chunky)
//   samples       1, 2, 4, 8 or 16 bits, unsigned integer
//   photometric   WhiteIsZero, BlackIsZero, RGB, palette, CMYK (InkSet 1),
//                 YCbCr (only as JPEG)
//   alpha         the first extra sample: associated (straightened here) or
//                 unassociated; an RGB(A)/gray with more samples than its
//                 colour channels and no ExtraSamples tag reads it as alpha
//   layout        strips or tiles, planar configuration 1 or 2, FillOrder 2,
//                 either byte order
//
// The layout is read first (tiff_layout) so a caller can check the size
// before decoding (tiff_decode). Used by decode_memory / decode_memory_u16 /
// probe_dimensions_memory / decode_memory_bounded.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace broimage::detail {

struct TiffLayout {
    uint32_t width = 0, height = 0;
    uint32_t bits = 1;          // per sample, every sample alike
    uint32_t samples = 1;       // per pixel
    uint32_t photometric = 0;
    uint32_t compression = 1;
    uint32_t predictor = 1;
    uint32_t planar = 1;
    uint32_t fill_order = 1;
    uint32_t color = 1;         // colour samples: 1 gray / palette, 3 RGB / YCbCr, 4 CMYK
    int alpha = -1;             // sample index of alpha, or -1
    bool premultiplied = false; // associated alpha
    bool tiled = false;
    uint32_t chunk_w = 0, chunk_h = 0;  // tile size, or width x RowsPerStrip
    std::vector<uint32_t> offsets, counts;
    std::vector<uint16_t> colormap;     // palette: 3 << bits entries, R then G then B
    const uint8_t* jpeg_tables = nullptr;
    std::size_t jpeg_tables_n = 0;

    // What a decode reports as the encoded channel count (1 / 2 / 3 / 4).
    int channels() const;
};

bool is_tiff(const uint8_t* data, std::size_t size);

// Page 0's layout. False (and `why` set) when `data` is not a TIFF this
// decoder reads, or the image is beyond its size caps.
bool tiff_layout(const uint8_t* data, std::size_t size, TiffLayout& out, std::string* why = nullptr);

// RGBA, straight alpha, at 8 or 16 bits per channel.
bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& layout, std::vector<uint8_t>& rgba,
                 std::string* why = nullptr);
bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& layout, std::vector<uint16_t>& rgba,
                 std::string* why = nullptr);

} // namespace broimage::detail
