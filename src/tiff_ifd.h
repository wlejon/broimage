#pragma once
// Internal: the TIFF container. A TIFF (and every EXIF block, which is a TIFF
// header and IFDs without pixels) is a byte-order mark, the magic 42 (classic
// TIFF) or 43 (BigTIFF), and a chain of image file directories (IFDs):
// 12-byte (classic) or 20-byte (BigTIFF) entries of tag, type, count and
// either the value itself (when it fits in 4 / 8 bytes) or the offset of the
// values. This reads the directory structure only, bounds checked against the
// buffer or file; the pixel decoder (decode_tiff), the GeoTIFF raster decoder
// (decode_geotiff), and the EXIF reader (exif) all sit on it.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace broimage::detail {

// TIFF field types.
enum TiffType : uint16_t {
    kTiffByte = 1, kTiffAscii = 2, kTiffShort = 3, kTiffLong = 4, kTiffRational = 5,
    kTiffSByte = 6, kTiffUndefined = 7, kTiffSShort = 8, kTiffSLong = 9, kTiffSRational = 10,
    kTiffFloat = 11, kTiffDouble = 12, kTiffIfd = 13,
    kTiffLong8 = 16, kTiffSLong8 = 17, kTiffIfd8 = 18,
};

// Bytes per value of a type; 0 for an unknown type.
uint32_t tiff_type_size(uint16_t type);

struct TiffEntry {
    uint16_t tag = 0;
    uint16_t type = 0;
    uint64_t count = 0;
    uint64_t at = 0;           // where the values start; at + count * size is in bounds
    uint8_t inline_bytes[8]{}; // used in stream mode when value fits inline
    bool is_inline = false;    // true in stream mode when value fits inline
};

class TiffReader {
public:
    using ReadFn = std::function<bool(uint64_t offset, void* dst, std::size_t len)>;

    // False unless `data` starts with a classic TIFF ("II*\0" / "MM\0*") or
    // BigTIFF ("II+\0\x08\0\0\0" / "MM\0+\0\x08\0\0") header.
    bool open(const uint8_t* data, std::size_t size);

    // Open a file-backed or callback-backed stream of `size` bytes without
    // reading the whole file into memory.
    bool open_stream(uint64_t size, ReadFn read_fn);

    const uint8_t* data() const { return b_; }
    uint64_t size() const { return n_; }
    bool little_endian() const { return le_; }
    bool is_bigtiff() const { return big_; }
    uint64_t first_ifd() const { return ifd0_; }

    bool read_bytes(uint64_t o, void* dst, std::size_t len) const;
    uint16_t u16(uint64_t o) const;
    uint32_t u32(uint64_t o) const;
    uint64_t u64(uint64_t o) const;

    // The entries of the IFD at `offset`. An entry whose values lie outside
    // the buffer/file, or whose type is unknown, is left out rather than
    // failing the directory. False when the directory itself is out of bounds.
    bool read_ifd(uint64_t offset, std::vector<TiffEntry>& out, uint64_t* next = nullptr) const;
    bool read_ifd(uint64_t offset, std::vector<TiffEntry>& out, uint32_t* next) const;

    // Value `i` of an integer entry (BYTE, SHORT, LONG, LONG8, their signed
    // forms, UNDEFINED, ASCII and IFD/IFD8); false for another type or i >= count.
    bool uint64_at(const TiffEntry& e, uint64_t i, uint64_t& v) const;
    bool uint_at(const TiffEntry& e, uint64_t i, uint32_t& v) const;
    bool int_at(const TiffEntry& e, uint64_t i, int64_t& v) const;
    bool sint_at(const TiffEntry& e, uint64_t i, int64_t& v) const { return int_at(e, i, v); }

    // Value `i` as a real number: RATIONAL / SRATIONAL (false when the
    // denominator is 0), FLOAT / DOUBLE, or any integer type.
    bool double_at(const TiffEntry& e, uint64_t i, double& v) const;
    bool real_at(const TiffEntry& e, uint64_t i, double& v) const { return double_at(e, i, v); }

    // ASCII (or UNDEFINED / BYTE) up to the first NUL, trailing spaces trimmed.
    bool ascii(const TiffEntry& e, std::string& out) const;
    std::string text(const TiffEntry& e) const;

private:
    const uint8_t* entry_bytes(const TiffEntry& e, uint64_t byte_offset, std::size_t need) const;
    bool parse_header(const uint8_t* hdr, std::size_t hdr_n);

    const uint8_t* b_ = nullptr;
    uint64_t n_ = 0;
    ReadFn read_fn_;
    bool le_ = true;
    bool big_ = false;
    uint64_t ifd0_ = 0;

    mutable uint64_t cache_off_ = ~uint64_t(0);
    mutable std::vector<uint8_t> cache_buf_;
};

// The entry with `tag`, or nullptr.
const TiffEntry* tiff_find(const std::vector<TiffEntry>& entries, uint16_t tag);

} // namespace broimage::detail
