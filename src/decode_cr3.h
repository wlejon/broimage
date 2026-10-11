#pragma once
// Internal: Canon CR3 (ISO base media file format, brand "crx"): not the RAW
// data, the JPEGs and metadata Canon stores beside it.
//
//   moov / uuid 85c0b687-820f-11e0-8111-f4ce462b6a48
//        CMT1 .. CMT4   TIFF blocks: IFD0, the Exif IFD, makernotes, GPS
//        THMB           a 160 x 120 JPEG
//   uuid eaf42b5e-1c98-4b88-b9fb-b7dc406e4d16 / PRVW
//                       a 1620 x 1080 JPEG
//   moov / trak[0]      its one sample is a full-size JPEG (in mdat)
//
// The JPEGs are as the sensor sees, the orientation is CMT1's Orientation.

#include <cstddef>
#include <cstdint>

namespace broimage::detail {

struct Cr3Span {
    const uint8_t* data = nullptr;
    std::size_t size = 0;
    explicit operator bool() const { return data != nullptr; }
};

struct Cr3Parts {
    Cr3Span cmt[4];     // CMT1..CMT4, each starting with its TIFF header
    Cr3Span thumbnail;  // THMB's JPEG
    Cr3Span preview;    // PRVW's JPEG
    Cr3Span full;       // trak[0]'s JPEG
    // trak[0]'s size from its tkhd, known when only the head of the file was
    // read and `full` (which lies in mdat, after the RAW data) is not there.
    int full_width = 0;
    int full_height = 0;
};

bool cr3_sniff(const uint8_t* data, std::size_t size);

// Locate what is there; any of it may be missing (a span left empty). False
// when `data` is not a CR3.
bool cr3_parts(const uint8_t* data, std::size_t size, Cr3Parts& out);

} // namespace broimage::detail
