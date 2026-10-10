#pragma once
// Internal: the TIFF container. A TIFF (and every EXIF block, which is a TIFF
// header and IFDs without pixels) is a byte-order mark, the magic 42, and a
// chain of image file directories (IFDs): 12-byte entries of tag, type,
// count and either the value itself (when it fits in four bytes) or the
// offset of the values. This reads the directory structure only, bounds
// checked against the buffer; the pixel decoder (decode_tiff) and the EXIF
// reader (exif) both sit on it. Classic TIFF only: BigTIFF (magic 43) is not
// read.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace broimage::detail {

// TIFF field types.
enum TiffType : uint16_t {
    kTiffByte = 1, kTiffAscii = 2, kTiffShort = 3, kTiffLong = 4, kTiffRational = 5,
    kTiffSByte = 6, kTiffUndefined = 7, kTiffSShort = 8, kTiffSLong = 9, kTiffSRational = 10,
    kTiffFloat = 11, kTiffDouble = 12, kTiffIfd = 13,
};

// Bytes per value of a type; 0 for an unknown type.
uint32_t tiff_type_size(uint16_t type);

struct TiffEntry {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint32_t count = 0;
    std::size_t at = 0;  // where the values start; at + count * size is in bounds
};

class TiffReader {
public:
    // False unless `data` starts with a classic TIFF header ("II*\0" / "MM\0*").
    bool open(const uint8_t* data, std::size_t size);

    const uint8_t* data() const { return b_; }
    std::size_t size() const { return n_; }
    bool little_endian() const { return le_; }
    uint32_t first_ifd() const { return ifd0_; }

    uint16_t u16(std::size_t o) const;
    uint32_t u32(std::size_t o) const;

    // The entries of the IFD at `offset`. An entry whose values lie outside
    // the buffer, or whose type is unknown, is left out rather than failing
    // the directory. False when the directory itself is out of bounds.
    bool read_ifd(uint32_t offset, std::vector<TiffEntry>& out, uint32_t* next = nullptr) const;

    // Value `i` of an integer entry (BYTE, SHORT, LONG, their signed forms,
    // UNDEFINED and IFD); false for another type or i >= count.
    bool uint_at(const TiffEntry& e, uint32_t i, uint32_t& v) const;
    bool sint_at(const TiffEntry& e, uint32_t i, int64_t& v) const;
    // Value `i` as a real number: RATIONAL / SRATIONAL (false when the
    // denominator is 0), FLOAT / DOUBLE, or any integer type.
    bool real_at(const TiffEntry& e, uint32_t i, double& v) const;
    // ASCII (or UNDEFINED / BYTE) up to the first NUL, trailing spaces trimmed.
    std::string text(const TiffEntry& e) const;

private:
    const uint8_t* b_ = nullptr;
    std::size_t n_ = 0;
    bool le_ = true;
    uint32_t ifd0_ = 0;
};

// The entry with `tag`, or nullptr.
const TiffEntry* tiff_find(const std::vector<TiffEntry>& entries, uint16_t tag);

} // namespace broimage::detail
