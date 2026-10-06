#pragma once
// Internal: baseline TIFF, uncompressed, 8 bits per sample, chunky RGB(A)
// or grayscale(+alpha) in any strip layout and either byte order. The
// layout is read first (tiff_header) so a caller can check the size before
// decoding (tiff_decode). Used by decode_memory_bounded.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace broimage::detail {

struct TiffLayout {
    uint32_t width = 0, height = 0;
    uint32_t samples = 1;      // per pixel
    uint32_t photometric = 2;  // 0 WhiteIsZero, 1 BlackIsZero, 2 RGB
    uint32_t extra = 0;        // ExtraSamples: 1 associated alpha, 2 unassociated
    std::vector<uint32_t> offsets, counts;  // strips
};

bool is_tiff(const uint8_t* data, std::size_t size);
// False when it is not a TIFF this decoder reads.
bool tiff_header(const uint8_t* data, std::size_t size, TiffLayout& out);
// RGBA8, straight alpha.
bool tiff_decode(const uint8_t* data, std::size_t size, const TiffLayout& layout, std::vector<uint8_t>& rgba);

} // namespace broimage::detail
