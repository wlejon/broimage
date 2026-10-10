#pragma once

// HEIF images: AVIF (AV1) and HEIC (HEVC).
//
// broimage reads the HEIF container itself (ISO/IEC 23008-12, with the MIAF
// and AVIF profiles): the primary item, grid-derived images (the tiles a
// phone stores a photo as), the alpha auxiliary image, the clean aperture
// (`clap`), rotation and mirroring (`irot` / `imir`), colour (`colr`: nclx
// and ICC), Exif items and thumbnail items. Every decode entry point reads
// it — decode_memory and its variants, probe_dimensions_memory (from `ispe`,
// without decoding), read_exif / read_exif_orientation, open_frames and the
// bounded decode.
//
// The coded pixels go to a codec backend:
//
//   AV1 (AVIF)  a decoder the host registers with set_av1_decoder (bro: dav1d).
//               broimage converts its YUV planes to RGBA8 itself: 8/10/12-bit,
//               4:2:0 / 4:2:2 / 4:4:4 / monochrome, the item's nclx matrix
//               and range (or the bitstream's), BT.601/709/2020, full or
//               limited, bilinear chroma upsampling. Without a decoder an
//               AVIF fails to decode with a reason (its size still probes).
//   HEVC (HEIC) the operating system's decoder, never one of ours: WIC on
//               Windows (needs Microsoft's "HEVC Video Extensions" from the
//               Store), ImageIO on macOS. Linux has none: HEIC fails with a
//               reason saying so. The whole file goes to the OS; the
//               container reader still answers size, Exif and orientation.
//
// Orientation. HEIF rotates and mirrors with item properties, not the Exif
// Orientation tag (which a HEIF reader must ignore). broimage follows that:
// decode_memory returns the pixels as stored — after `clap`, before `irot`
// and `imir` — and read_exif_orientation (and read_exif's `orientation`)
// answers the EXIF orientation value those two properties amount to, so
// decode_memory_oriented, `<img>` and `image-orientation` treat a HEIF like
// any other photo.
//
// High dynamic range. PQ (SMPTE ST 2084) and HLG (ARIB STD-B67) content is
// tone-mapped to SDR: decoded to linear light (HLG through its BT.2100
// OOTF at a 1000-nit display), SDR reference white placed at 203 nits
// (ITU-R BT.2408), converted from the item's primaries (BT.2020) to
// BT.709, then highlights rolled off on the largest channel (so hue holds):
// identity up to 75% of SDR white, an extended-Reinhard shoulder above it
// that reaches 1.0 at the content's peak (`clli` max CLL, else 1000 nits);
// then sRGB-encoded. SDR white lands near 240 of 255. SDR content is not
// colour-managed: its values are taken as sRGB, as every other format
// broimage reads (ICC profiles are reported, not applied). On macOS a HEIC
// is drawn into sRGB by ColorSync (so a P3 photo is converted there).
//
// imir: the HEIF 2022 revision's `mode` — 0 mirrors top to bottom, 1 left
// to right (WIC, libavif and Chrome read it so).
//
// Image sequences (`avis`, `msf1`) give their still: the primary item every
// writer adds, or the track's first sample when there is none. They do not
// animate.

#include "broimage/buffer.h"
#include "broimage/decode.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace broimage {

// ---- The AV1 backend -----------------------------------------------------------

// One decoded AV1 frame, as the decoder holds it: borrowed, valid during
// the sink call only.
struct YuvPlanes {
    int width = 0;           // of the luma plane
    int height = 0;
    int bit_depth = 8;       // 8, 10 or 12; above 8 a sample is a uint16_t
    int chroma_shift_x = 1;  // 4:2:0 = 1,1   4:2:2 = 1,0   4:4:4 = 0,0
    int chroma_shift_y = 1;
    bool monochrome = false; // no chroma planes (4:0:0)
    const void* planes[3] = {nullptr, nullptr, nullptr};  // Y, U (Cb), V (Cr)
    std::ptrdiff_t strides[3] = {0, 0, 0};                // bytes
    // The sequence header's colour description, used where the item has no
    // nclx `colr` (H.273 code points; 2 = unspecified).
    int color_primaries = 2;
    int transfer = 2;
    int matrix = 2;
    bool full_range = false;
    // AV1 chroma_sample_position: 0 unknown, 1 vertical, 2 colocated.
    int chroma_position = 0;
};

// Receives the frame; return false to fail the decode.
using YuvSink = bool (*)(void* ctx, const YuvPlanes& frame);

// Decode one AV1 image (a temporal unit of OBUs, the sequence header first)
// and hand its frame to `sink`. False (and `why`) when it does not decode.
// Called from any thread.
using Av1Decoder = bool (*)(const uint8_t* obus, std::size_t size, YuvSink sink, void* ctx,
                            std::string* why);

// Register the AV1 decoder (once; thread-safe). Null unregisters.
void set_av1_decoder(Av1Decoder decoder);
bool has_av1_decoder();

// ---- Inspection ------------------------------------------------------------------

// A HEIF file: `ftyp` with a HEIF image brand (mif1, msf1, avif, avis,
// heic, heix, ...).
bool is_heif(const uint8_t* data, std::size_t size);

struct HeifInfo {
    std::string major_brand;
    std::vector<std::string> compatible_brands;
    // "image/avif", "image/heic" or "image/heif" (another codec).
    std::string mime;
    // The primary image's coding ("av01", "hvc1", "jpeg", ...; for a grid,
    // its tiles').
    std::string codec;
    // As decode_memory returns it: after `clap`, before `irot` / `imir`.
    int width = 0;
    int height = 0;
    int bit_depth = 8;         // pixi, else the codec configuration
    bool has_alpha = false;
    bool grid = false;
    int grid_rows = 0;
    int grid_columns = 0;
    ExifOrientation orientation = ExifOrientation::Normal;  // from irot / imir
    bool has_exif = false;
    bool has_icc = false;
    int primaries = 2, transfer = 2, matrix = 2;  // nclx colr; 2 when absent
    bool full_range = false;
    bool hdr = false;          // PQ or HLG
    bool sequence = false;     // an image sequence (a track)
    // Thumbnail items of the primary image, their stored sizes.
    struct Thumb {
        int width = 0;
        int height = 0;
        std::string codec;
    };
    std::vector<Thumb> thumbnails;
};

// Read the container (no decoding). False (and `why`) when `data` is not a
// HEIF file or its structure does not parse.
bool read_heif_info(const uint8_t* data, std::size_t size, HeifInfo& out, std::string* why = nullptr);

// A fast preview: the embedded thumbnail item of the primary image — the
// smallest whose shorter side is at least `min_side` (the largest when none
// is), upright (its own irot / imir applied) and straight RGBA8. False when
// the file has no thumbnail item or it does not decode.
bool decode_heif_thumbnail(const uint8_t* data, std::size_t size, int min_side, Image& out,
                           std::string* why = nullptr);

// ---- Capability ------------------------------------------------------------------

// Whether this build, on this machine, decodes images of `type`: a MIME type
// ("image/avif", "image/heic", ...) or a file extension ("avif", ".heic").
// AVIF needs a registered AV1 decoder; HEIC/HEIF the OS decoder, probed once
// at run time. When it does not, `why` says what is missing (on Windows,
// the HEVC Video Extensions from the Microsoft Store). Unknown types are
// false with an empty `why`.
bool can_decode(std::string_view type, std::string* why = nullptr);

}  // namespace broimage
