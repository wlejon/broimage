#include "broimage/ktx2.h"

#include "basisu_transcoder.h"

#include <mutex>

namespace broimage {

namespace {

// The transcoder's global tables build once per process.
void ensureTranscoderInit() {
    static std::once_flag once;
    std::call_once(once, [] { basist::basisu_transcoder_init(); });
}

basist::transcoder_texture_format basisFormatOf(Ktx2Format f) {
    switch (f) {
        case Ktx2Format::BC1: return basist::transcoder_texture_format::cTFBC1_RGB;
        case Ktx2Format::BC3: return basist::transcoder_texture_format::cTFBC3_RGBA;
        case Ktx2Format::BC4: return basist::transcoder_texture_format::cTFBC4_R;
        case Ktx2Format::BC5: return basist::transcoder_texture_format::cTFBC5_RG;
        case Ktx2Format::BC7: return basist::transcoder_texture_format::cTFBC7_RGBA;
        default: return basist::transcoder_texture_format::cTFRGBA32;
    }
}

size_t blockBytesOf(Ktx2Format f) {
    switch (f) {
        case Ktx2Format::BC1:
        case Ktx2Format::BC4: return 8;
        default: return 16;  // BC3/BC5/BC7
    }
}

}  // namespace

Ktx2Image transcode_ktx2(const uint8_t* data, std::size_t size, Ktx2Format target) {
    ensureTranscoderInit();

    Ktx2Image out;
    if (!data || size == 0) {
        out.error = "ktx2: empty buffer";
        return out;
    }

    basist::ktx2_transcoder transcoder;
    if (!transcoder.init(data, static_cast<uint32_t>(size))) {
        out.error = "ktx2: not a KTX2 file (or unsupported payload)";
        return out;
    }
    if (!transcoder.start_transcoding()) {
        out.error = "ktx2: start_transcoding failed (corrupt or unsupported "
                    "supercompression)";
        return out;
    }

    out.width = static_cast<int>(transcoder.get_width());
    out.height = static_cast<int>(transcoder.get_height());
    out.layers = static_cast<int>(transcoder.get_layers() ? transcoder.get_layers() : 1);
    out.faces = static_cast<int>(transcoder.get_faces());
    out.hasAlpha = transcoder.get_has_alpha() != 0;
    out.srgb = transcoder.is_srgb();

    // BC1 asked of an image that carries alpha would silently drop it —
    // answer RGBA8 instead and let the caller see `format`.
    Ktx2Format format = target;
    if (format == Ktx2Format::BC1 && out.hasAlpha) format = Ktx2Format::RGBA8;
    out.format = format;

    const basist::transcoder_texture_format basisFormat = basisFormatOf(format);
    const uint32_t levelCount = transcoder.get_levels() ? transcoder.get_levels() : 1;

    for (uint32_t level = 0; level < levelCount; ++level) {
        basist::ktx2_image_level_info info;
        if (!transcoder.get_image_level_info(info, level, /*layer=*/0, /*face=*/0)) {
            out.error = "ktx2: level info unavailable";
            out.mips.clear();
            return out;
        }

        Ktx2Level mip;
        mip.width = static_cast<int>(info.m_orig_width);
        mip.height = static_cast<int>(info.m_orig_height);

        uint32_t unitCount;  // pixels for RGBA32, 4x4 blocks for BC
        if (format == Ktx2Format::RGBA8) {
            unitCount = info.m_orig_width * info.m_orig_height;
            mip.data.resize(static_cast<size_t>(unitCount) * 4);
        } else {
            const uint32_t bw = (info.m_orig_width + 3) / 4;
            const uint32_t bh = (info.m_orig_height + 3) / 4;
            unitCount = bw * bh;
            mip.data.resize(static_cast<size_t>(unitCount) * blockBytesOf(format));
        }

        if (!transcoder.transcode_image_level(level, /*layer=*/0, /*face=*/0,
                                              mip.data.data(), unitCount, basisFormat)) {
            out.error = "ktx2: transcode failed at level " + std::to_string(level);
            out.mips.clear();
            return out;
        }
        out.mips.push_back(std::move(mip));
    }

    return out;
}

}  // namespace broimage
