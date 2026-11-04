#pragma once
// Internal: TIFF samples to RGBA. The decoder (decode_tiff) unpacks every
// chunk into one interleaved buffer of `layout.samples` raw values per pixel;
// this turns that into straight-alpha RGBA at 8-bit, 16-bit, or float32 per
// channel by the layout's photometric interpretation.

#include "decode_tiff.h"

#include <cstdint>

namespace broimage::detail {

void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, uint8_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, uint16_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const uint8_t* samples, float* rgba);

void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, uint8_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, uint16_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const uint16_t* samples, float* rgba);

void tiff_to_rgba(const TiffLayout& l, const float* samples, uint8_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const float* samples, uint16_t* rgba);
void tiff_to_rgba(const TiffLayout& l, const float* samples, float* rgba);

} // namespace broimage::detail
