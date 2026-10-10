// HEIF decoding inside broimage (heif.h): what decode.cpp, exif.cpp and
// decode_bounded.cpp call for a HEIF file.

#pragma once

#include "broimage/buffer.h"
#include "broimage/decode.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace broimage::detail {

bool heif_probe(const uint8_t* data, std::size_t size, int& width, int& height, int& channels);

// The primary image as stored: after clap, before irot / imir.
bool heif_decode(const uint8_t* data, std::size_t size, Image& out, std::string* why);

// What irot / imir amount to (Normal for a file that is not HEIF).
ExifOrientation heif_orientation(const uint8_t* data, std::size_t size);

// The TIFF block of the file's Exif item.
bool heif_find_exif(const uint8_t* data, std::size_t size, const uint8_t*& tiff, std::size_t& tiff_size);

}  // namespace broimage::detail
