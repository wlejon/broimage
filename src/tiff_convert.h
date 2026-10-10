#pragma once
// Internal: TIFF samples to RGBA. The decoder (decode_tiff) unpacks every
// chunk into one interleaved buffer of `layout.samples` raw values per pixel
// (each 0 .. 2^bits - 1, 8-bit storage for depths up to 8, 16-bit for 16);
// this turns that into straight-alpha RGBA at 8 or 16 bits per channel by
// the layout's photometric interpretation.

#include "decode_tiff.h"

#include <cstdint>

namespace broimage::detail {

void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, uint8_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, uint16_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, uint8_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, uint16_t* rgba);

} // namespace broimage::detail
