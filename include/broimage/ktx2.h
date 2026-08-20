#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace broimage {

// KTX2 texture transcode (basis_universal). A .ktx2 file carries a
// GPU-agnostic payload (ETC1S/BasisLZ or UASTC, optionally Zstd
// supercompressed); the transcoder turns it into a format a GPU accepts.
// RGBA8 is the always-works target — every GL path uploads it — and the BC
// formats keep the memory savings on desktop GL where the extension is a
// given.

enum class Ktx2Format : uint8_t {
    RGBA8,  // uncompressed, 4 bytes/pixel; universally uploadable
    BC1,    // opaque, 8 bytes per 4x4 block (GL: COMPRESSED_RGB_S3TC_DXT1)
    BC3,    // with alpha, 16 bytes per 4x4 block (DXT5)
    BC4,    // single channel, 8 bytes per 4x4 block
    BC5,    // two channel, 16 bytes per 4x4 block
    BC7,    // high quality RGBA, 16 bytes per 4x4 block
};

struct Ktx2Level {
    int width = 0;   // this mip's pixel size
    int height = 0;
    std::vector<uint8_t> data;  // pixels (RGBA8) or blocks (BC*)
};

struct Ktx2Image {
    int width = 0;   // level 0 size
    int height = 0;
    int layers = 1;
    int faces = 1;
    bool hasAlpha = false;
    bool srgb = false;
    Ktx2Format format = Ktx2Format::RGBA8;  // what `mips` actually holds
    std::vector<Ktx2Level> mips;            // layer 0 / face 0 mip chain
    std::string error;
    bool ok() const { return error.empty(); }
};

// Transcode a KTX2 file in memory to `target`. A BC target silently falls
// back to RGBA8 when the payload cannot serve it (BC1 asked of an alpha
// image keeps its alpha rather than dropping it); check `format` on the
// result for what was produced. On failure `error` is set and everything
// else is empty.
Ktx2Image transcode_ktx2(const uint8_t* data, std::size_t size,
                         Ktx2Format target = Ktx2Format::RGBA8);

}  // namespace broimage
