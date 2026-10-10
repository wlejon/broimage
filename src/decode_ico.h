#pragma once
// Internal: Windows icon / cursor containers (.ico / .cur). An ICO is a
// directory of images, each either a whole PNG file or a headerless BMP (a
// DIB: BITMAPINFOHEADER, optional palette, the colour bitmap, then a 1-bit
// AND transparency mask, the stored height counting both). The best entry is
// the largest one, ties going to the deepest colour; that one is decoded.
// Used by decode_memory / probe_dimensions_memory when stb does not know the
// bytes.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace broimage::detail {

bool is_ico(const uint8_t* data, std::size_t size);
// Size of the entry ico_decode would decode. False when it is not an ICO
// this decoder reads.
bool ico_probe(const uint8_t* data, std::size_t size, int& width, int& height);
// RGBA8, straight alpha. A PNG entry is decoded by stb.
bool ico_decode(const uint8_t* data, std::size_t size, int& width, int& height,
                std::vector<uint8_t>& rgba);

} // namespace broimage::detail
