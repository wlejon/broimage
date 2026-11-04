#pragma once
// Internal: the TIFF image decoder. Reads the first page (IFD0):
//
//   compression   none, LZW (with horizontal predictor 2 or float predictor 3),
//                 Deflate (8 and 32946), PackBits, JPEG (7, with or without
//                 JPEGTables; 8-bit YCbCr, RGB or gray, chunky)
//   samples       1, 2, 4, 8, 16 or 32-bit unsigned integer (SampleFormat 1);
//                 8, 16 or 32-bit signed integer (SampleFormat 2);
//                 32 or 64-bit IEEE floating point (SampleFormat 3)
//   photometric   WhiteIsZero, BlackIsZero, RGB, palette, CMYK (InkSet 1),
//                 YCbCr (only as JPEG)
//   alpha         the first extra sample: associated (straightened here) or
//                 unassociated; an RGB(A)/gray with more samples than its
//                 colour channels and no ExtraSamples tag reads it as alpha
//   layout        strips or tiles, planar configuration 1 or 2, FillOrder 2,
//                 either byte order, classic TIFF (42) or BigTIFF (43)
//
// The layout is read first (tiff_layout) so a caller can check the size
// before decoding (tiff_decode). Used by decode_memory / decode_memory_u16 /
// decode_memory_f32 / probe_dimensions_memory / decode_memory_bounded.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace broimage::detail {

struct TiffLayout {
    uint32_t width = 0, height = 0;
    uint32_t bits = 1;          // per sample, every sample alike
    uint32_t samples = 1;       // per pixel
    uint32_t sample_format = 1; // 1 = uint, 2 = sint, 3 = float
    uint32_t photometric = 0;
    uint32_t compression = 1;
    uint32_t predictor = 1;     // 1 = none, 2 = horizontal diff, 3 = float byte-lane diff
    uint32_t planar = 1;
    uint32_t fill_order = 1;
    uint32_t color = 1;         // colour samples: 1 gray / palette, 3 RGB / YCbCr, 4 CMYK
    int alpha = -1;             // sample index of alpha, or -1
    bool premultiplied = false; // associated alpha
    bool tiled = false;
    bool bigtiff = false;
    uint32_t chunk_w = 0, chunk_h = 0;  // tile size, or width x RowsPerStrip
    std::vector<uint64_t> offsets, counts;
    std::vector<uint16_t> colormap;     // palette: 3 << bits entries, R then G then B
    const uint8_t* jpeg_tables = nullptr;
    std::size_t jpeg_tables_n = 0;

    // What a decode reports as the encoded channel count (1 / 2 / 3 / 4).
    int channels() const;
};

bool is_tiff(const uint8_t* data, std::size_t size);

// Decompress a single strip/tile chunk into `dst` (which has size `expected`)
// and apply FillOrder=2 bit reversal if needed.
bool tiff_decompress_chunk(uint32_t compression, uint32_t fill_order,
                           const uint8_t* src, std::size_t cnt,
                           uint8_t* dst, std::size_t expected);

// Adobe TIFF Tech Note 3 floating-point horizontal predictor (Predictor = 3)
// undo for one row of `row_pixels` pixels, `cs` channels per pixel, `bps`
// bytes per sample ({2, 4, 8}). Un-shuffles the MSB-first byte lanes into
// `out_le` byte order.
void tiff_undo_predictor3(uint8_t* rp, std::size_t row_pixels, uint32_t cs,
                          uint32_t bps, bool out_le, std::vector<uint8_t>& scratch);

// Page 0's layout. False (and `why` set) when `data` is not a TIFF this
// decoder reads, or the image is beyond its size caps.
bool tiff_layout(const uint8_t* data, std::size_t size, TiffLayout& out, std::string* why = nullptr);

// RGBA, straight alpha, at 8-bit, 16-bit, or 32-bit float per channel.
bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& layout, std::vector<uint8_t>& rgba,
                 std::string* why = nullptr);
bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& layout, std::vector<uint16_t>& rgba,
                 std::string* why = nullptr);
bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& layout, std::vector<float>& rgba,
                 std::string* why = nullptr);

} // namespace broimage::detail
