// HEIC where the operating system has no decoder broimage uses (Linux and
// the rest): broimage carries no HEVC decoder, so HEIC does not decode.
// The container still probes, and its Exif and orientation still read.

#include "heif_os.h"

namespace broimage::detail {

namespace {

const char* kWhy = "HEIC is not supported on this platform: HEVC images are decoded by the operating system "
                   "(Windows, macOS), and there is no HEVC decoder here";

}  // namespace

bool os_heif_available(std::string* why) {
    if (why) *why = kWhy;
    return false;
}

bool os_heif_decode(const uint8_t*, std::size_t, Image& out, bool& oriented, std::string* why) {
    out = Image{};
    oriented = false;
    if (why) *why = kWhy;
    return false;
}

}  // namespace broimage::detail
