#pragma once

// EXIF metadata: the camera, exposure, date and location a photo carries.
//
// EXIF is a TIFF directory structure (IFD0, with the Exif and GPS sub-IFDs
// hanging off it) embedded in the image file: a JPEG's APP1 "Exif" segment,
// a WebP's RIFF "EXIF" chunk, a PNG's eXIf chunk, a HEIF's Exif item, or,
// for a TIFF, the file's own first IFD. read_exif finds it in any of those
// and reads the fields below; it never touches pixels. For a HEIF (AVIF,
// HEIC) `orientation` is what its irot / imir properties amount to — HEIF
// readers ignore the Exif tag — 1 when it has neither.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace broimage {

struct ExifInfo {
    // Strings are empty when the tag is absent.
    std::string make;          // Make (0x010F)
    std::string model;         // Model (0x0110)
    std::string lens_make;     // LensMake (0xA433)
    std::string lens_model;    // LensModel (0xA434)

    // ExposureTime (0x829A) in seconds, and as a photographer writes it:
    // "1/250" below a quarter second (or whenever 1/t is a whole number),
    // otherwise decimal seconds ("0.3", "2", "2.5"). Empty when absent.
    std::optional<double> exposure_time;
    std::string exposure_time_text;

    std::optional<double> f_number;           // FNumber (0x829D)
    // PhotographicSensitivity (0x8827), else ISOSpeed (0x8833).
    std::optional<uint32_t> iso;
    std::optional<double> focal_length;       // FocalLength (0x920A), millimetres
    std::optional<double> focal_length_35mm;  // FocalLengthIn35mmFilm (0xA405), when non-zero

    // DateTimeOriginal (0x9003) as ISO 8601: "2024-05-06T14:30:00", with
    // SubSecTimeOriginal (0x9291) fractional seconds ("...00.25") and
    // OffsetTimeOriginal (0x9011) appended when present ("...+02:00"). EXIF
    // times are local; without the offset the zone is unknown. Empty when
    // absent or malformed.
    std::string date_taken;
    std::string offset_time;   // OffsetTimeOriginal as stored ("+02:00"), or empty
    std::string sub_sec_time;  // SubSecTimeOriginal as stored ("25"), or empty

    // GPS, in signed decimal degrees (south and west negative) and metres
    // (below sea level negative). Absent unless the coordinate and its
    // reference are both present.
    std::optional<double> latitude;
    std::optional<double> longitude;
    std::optional<double> altitude;

    // Orientation (0x0112), 1..8 (see ExifOrientation in decode.h).
    std::optional<int> orientation;
};

// Locate the EXIF block (the TIFF header it starts with) in a JPEG, TIFF,
// WebP or PNG. False when there is none.
bool find_exif(const uint8_t* data, std::size_t size, const uint8_t*& tiff, std::size_t& tiff_size);

// The JPEG thumbnail of an EXIF block's IFD1 (JPEGInterchangeFormat and
// JPEGInterchangeFormatLength), as a span of `data`. Pixels are not touched;
// decode_exif_thumbnail (decode.h) decodes it upright. False when there is
// none or it does not lie within `data`.
bool find_exif_thumbnail(const uint8_t* data, std::size_t size, const uint8_t*& jpeg, std::size_t& jpeg_size);

// Read the fields above. False (and `out` default) when `data` carries no
// EXIF block whose header and IFD0 parse; a block whose IFD0 parses yields
// whatever fields it has, possibly none.
bool read_exif(const uint8_t* data, std::size_t size, ExifInfo& out);

// As above from a file: a JPEG's first 256 KB, any other file whole (a
// TIFF's directory, a WebP's EXIF chunk and a PNG's eXIf may come after the
// pixels).
bool read_exif_file(const std::string& path, ExifInfo& out);

} // namespace broimage
