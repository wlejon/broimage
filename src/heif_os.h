// The operating system's HEIF decoder, for HEVC-coded images (HEIC): WIC on
// Windows, ImageIO on macOS, none elsewhere. broimage carries no HEVC
// decoder of its own.

#pragma once

#include "broimage/buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace broimage::detail {

// Whether the OS decodes HEIC here (probed once, then cached). When it does
// not, `why` says what is missing.
bool os_heif_available(std::string* why = nullptr);

// The primary image of a whole HEIF file as straight RGBA8. `oriented` says
// whether the OS already applied the file's irot / imir (the caller wants
// the stored pixels and undoes it).
bool os_heif_decode(const uint8_t* data, std::size_t size, Image& out, bool& oriented,
                    std::string* why = nullptr);

}  // namespace broimage::detail
