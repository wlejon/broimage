// The TIFF container (tiff_ifd.h): header, IFDs, typed values.

#include "tiff_ifd.h"

#include <cstring>

namespace broimage::detail {

uint32_t tiff_type_size(uint16_t type) {
    switch (type) {
        case kTiffByte: case kTiffAscii: case kTiffSByte: case kTiffUndefined: return 1;
        case kTiffShort: case kTiffSShort: return 2;
        case kTiffLong: case kTiffSLong: case kTiffFloat: case kTiffIfd: return 4;
        case kTiffRational: case kTiffSRational: case kTiffDouble: return 8;
        default: return 0;
    }
}

bool TiffReader::open(const uint8_t* data, std::size_t size) {
    b_ = nullptr;
    n_ = 0;
    if (!data || size < 8) return false;
    if (std::memcmp(data, "II*\0", 4) == 0) le_ = true;
    else if (std::memcmp(data, "MM\0*", 4) == 0) le_ = false;
    else return false;
    b_ = data;
    n_ = size;
    ifd0_ = u32(4);
    return true;
}

uint16_t TiffReader::u16(std::size_t o) const {
    if (o > n_ || n_ - o < 2) return 0;
    return le_ ? uint16_t(b_[o] | b_[o + 1] << 8) : uint16_t(b_[o] << 8 | b_[o + 1]);
}

uint32_t TiffReader::u32(std::size_t o) const {
    if (o > n_ || n_ - o < 4) return 0;
    return le_ ? uint32_t(u16(o)) | uint32_t(u16(o + 2)) << 16 : uint32_t(u16(o)) << 16 | uint32_t(u16(o + 2));
}

bool TiffReader::read_ifd(uint32_t offset, std::vector<TiffEntry>& out, uint32_t* next) const {
    out.clear();
    if (next) *next = 0;
    if (!b_ || offset < 8 || uint64_t(offset) + 2 > n_) return false;
    const uint32_t count = u16(offset);
    if (uint64_t(offset) + 2 + uint64_t(count) * 12 > n_) return false;
    out.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const std::size_t at = std::size_t(offset) + 2 + std::size_t(i) * 12;
        TiffEntry e;
        e.tag = u16(at);
        e.type = u16(at + 2);
        e.count = u32(at + 4);
        const uint32_t unit = tiff_type_size(e.type);
        if (unit == 0) continue;
        const uint64_t bytes = uint64_t(e.count) * unit;
        e.at = bytes <= 4 ? at + 8 : u32(at + 8);
        if (uint64_t(e.at) + bytes > n_) continue;
        out.push_back(e);
    }
    if (next) {
        const std::size_t link = std::size_t(offset) + 2 + std::size_t(count) * 12;
        *next = u32(link);
    }
    return true;
}

bool TiffReader::uint_at(const TiffEntry& e, uint32_t i, uint32_t& v) const {
    if (i >= e.count) return false;
    switch (e.type) {
        case kTiffByte: case kTiffSByte: case kTiffUndefined: case kTiffAscii: v = b_[e.at + i]; return true;
        case kTiffShort: case kTiffSShort: v = u16(e.at + std::size_t(i) * 2); return true;
        case kTiffLong: case kTiffSLong: case kTiffIfd: v = u32(e.at + std::size_t(i) * 4); return true;
        default: return false;
    }
}

bool TiffReader::sint_at(const TiffEntry& e, uint32_t i, int64_t& v) const {
    uint32_t u = 0;
    if (!uint_at(e, i, u)) return false;
    switch (e.type) {
        case kTiffSByte: v = int8_t(u); break;
        case kTiffSShort: v = int16_t(u); break;
        case kTiffSLong: v = int32_t(u); break;
        default: v = u; break;
    }
    return true;
}

bool TiffReader::real_at(const TiffEntry& e, uint32_t i, double& v) const {
    if (i >= e.count) return false;
    const std::size_t o = e.at + std::size_t(i) * tiff_type_size(e.type);
    switch (e.type) {
        case kTiffRational: {
            const uint32_t num = u32(o), den = u32(o + 4);
            if (den == 0) return false;
            v = double(num) / double(den);
            return true;
        }
        case kTiffSRational: {
            const int32_t num = int32_t(u32(o)), den = int32_t(u32(o + 4));
            if (den == 0) return false;
            v = double(num) / double(den);
            return true;
        }
        case kTiffFloat: {
            const uint32_t bits = u32(o);
            float f;
            std::memcpy(&f, &bits, 4);
            v = f;
            return true;
        }
        case kTiffDouble: {
            const uint64_t lo = u32(o + (le_ ? 0 : 4)), hi = u32(o + (le_ ? 4 : 0));
            const uint64_t bits = hi << 32 | lo;
            std::memcpy(&v, &bits, 8);
            return true;
        }
        default: {
            int64_t s = 0;
            if (!sint_at(e, i, s)) return false;
            v = double(s);
            return true;
        }
    }
}

std::string TiffReader::text(const TiffEntry& e) const {
    if (tiff_type_size(e.type) != 1) return {};
    std::size_t len = 0;
    while (len < e.count && b_[e.at + len] != 0) ++len;
    while (len > 0 && b_[e.at + len - 1] == ' ') --len;
    return std::string(reinterpret_cast<const char*>(b_ + e.at), len);
}

const TiffEntry* tiff_find(const std::vector<TiffEntry>& entries, uint16_t tag) {
    for (const TiffEntry& e : entries)
        if (e.tag == tag) return &e;
    return nullptr;
}

} // namespace broimage::detail
