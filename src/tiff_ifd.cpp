// The TIFF container (tiff_ifd.h): header (classic TIFF 42 and BigTIFF 43),
// IFDs, typed values, and optional random-access stream reading.

#include "tiff_ifd.h"

#include <algorithm>
#include <cstring>

namespace broimage::detail {

namespace {

uint16_t u16_at(const uint8_t* p, bool le) {
    return le ? uint16_t(p[0] | (uint16_t(p[1]) << 8))
              : uint16_t((uint16_t(p[0]) << 8) | p[1]);
}

uint32_t u32_at(const uint8_t* p, bool le) {
    return le ? (uint32_t(u16_at(p, true)) | (uint32_t(u16_at(p + 2, true)) << 16))
              : ((uint32_t(u16_at(p, false)) << 16) | uint32_t(u16_at(p + 2, false)));
}

uint64_t u64_at(const uint8_t* p, bool le) {
    return le ? (uint64_t(u32_at(p, true)) | (uint64_t(u32_at(p + 4, true)) << 32))
              : ((uint64_t(u32_at(p, false)) << 32) | uint64_t(u32_at(p + 4, false)));
}

} // namespace

uint32_t tiff_type_size(uint16_t type) {
    switch (type) {
        case kTiffByte: case kTiffAscii: case kTiffSByte: case kTiffUndefined: return 1;
        case kTiffShort: case kTiffSShort: return 2;
        case kTiffLong: case kTiffSLong: case kTiffFloat: case kTiffIfd: return 4;
        case kTiffRational: case kTiffSRational: case kTiffDouble:
        case kTiffLong8: case kTiffSLong8: case kTiffIfd8: return 8;
        default: return 0;
    }
}

bool TiffReader::parse_header(const uint8_t* hdr, std::size_t hdr_n) {
    if (hdr_n < 8) return false;
    if (hdr[0] == 'I' && hdr[1] == 'I') le_ = true;
    else if (hdr[0] == 'M' && hdr[1] == 'M') le_ = false;
    else return false;

    const uint16_t magic = u16_at(hdr + 2, le_);
    if (magic == 42) {
        big_ = false;
        ifd0_ = u32_at(hdr + 4, le_);
        return true;
    }
    if (magic == 43) {
        if (hdr_n < 16) return false;
        if (u16_at(hdr + 4, le_) != 8 || u16_at(hdr + 6, le_) != 0) return false;
        big_ = true;
        ifd0_ = u64_at(hdr + 8, le_);
        return true;
    }
    return false;
}

bool TiffReader::open(const uint8_t* data, std::size_t size) {
    b_ = nullptr;
    n_ = 0;
    read_fn_ = nullptr;
    cache_off_ = ~uint64_t(0);
    cache_buf_.clear();
    if (!data || !parse_header(data, size)) return false;
    b_ = data;
    n_ = size;
    return true;
}

bool TiffReader::open_stream(uint64_t size, ReadFn read_fn) {
    b_ = nullptr;
    n_ = 0;
    read_fn_ = nullptr;
    cache_off_ = ~uint64_t(0);
    cache_buf_.clear();
    if (!read_fn || size < 8) return false;
    uint8_t hdr[16]{};
    const std::size_t hdr_len = std::min<uint64_t>(size, 16);
    if (!read_fn(0, hdr, hdr_len) || !parse_header(hdr, hdr_len)) return false;
    n_ = size;
    read_fn_ = std::move(read_fn);
    return true;
}

bool TiffReader::read_bytes(uint64_t o, void* dst, std::size_t len) const {
    if (o > n_ || n_ - o < len) return false;
    if (b_) {
        std::memcpy(dst, b_ + std::size_t(o), len);
        return true;
    }
    if (read_fn_) return read_fn_(o, dst, len);
    return false;
}

uint16_t TiffReader::u16(uint64_t o) const {
    uint8_t buf[2];
    if (!read_bytes(o, buf, 2)) return 0;
    return u16_at(buf, le_);
}

uint32_t TiffReader::u32(uint64_t o) const {
    uint8_t buf[4];
    if (!read_bytes(o, buf, 4)) return 0;
    return u32_at(buf, le_);
}

uint64_t TiffReader::u64(uint64_t o) const {
    uint8_t buf[8];
    if (!read_bytes(o, buf, 8)) return 0;
    return u64_at(buf, le_);
}

bool TiffReader::read_ifd(uint64_t offset, std::vector<TiffEntry>& out, uint32_t* next) const {
    uint64_t next64 = 0;
    if (!read_ifd(offset, out, next ? &next64 : nullptr)) {
        if (next) *next = 0;
        return false;
    }
    if (next) *next = next64 <= 0xFFFFFFFFull ? uint32_t(next64) : 0;
    return true;
}

bool TiffReader::read_ifd(uint64_t offset, std::vector<TiffEntry>& out, uint64_t* next) const {
    out.clear();
    if (next) *next = 0;
    if ((!b_ && !read_fn_) || offset < (big_ ? 16u : 8u)) return false;

    const uint64_t hdr_size = big_ ? 8 : 2;
    const uint64_t entry_size = big_ ? 20 : 12;
    const uint64_t inline_max = big_ ? 8 : 4;
    if (offset > n_ || n_ - offset < hdr_size) return false;

    const uint64_t count = big_ ? u64(offset) : u16(offset);
    if (count > 65536 || count * entry_size > n_ - offset - hdr_size) return false;

    const std::size_t table_bytes = std::size_t(count * entry_size);
    std::vector<uint8_t> table_buf;
    const uint8_t* table = nullptr;
    if (b_) {
        table = b_ + std::size_t(offset + hdr_size);
    } else {
        table_buf.resize(table_bytes);
        if (table_bytes > 0 && !read_bytes(offset + hdr_size, table_buf.data(), table_bytes)) return false;
        table = table_buf.data();
    }

    out.reserve(std::size_t(count));
    for (uint64_t i = 0; i < count; ++i) {
        const uint8_t* ep = table + std::size_t(i * entry_size);
        TiffEntry e;
        e.tag = u16_at(ep, le_);
        e.type = u16_at(ep + 2, le_);
        e.count = big_ ? u64_at(ep + 4, le_) : u32_at(ep + 4, le_);
        const uint32_t unit = tiff_type_size(e.type);
        if (unit == 0 || e.count > (~uint64_t(0)) / unit) continue;
        const uint64_t bytes = e.count * unit;
        const uint64_t val_pos = offset + hdr_size + i * entry_size + (big_ ? 12 : 8);
        const uint8_t* val_ptr = ep + (big_ ? 12 : 8);
        if (bytes <= inline_max) {
            e.at = val_pos;
            e.is_inline = true;
            std::memcpy(e.inline_bytes, val_ptr, std::size_t(inline_max));
        } else {
            e.at = big_ ? u64_at(val_ptr, le_) : u32_at(val_ptr, le_);
            e.is_inline = false;
        }
        if (e.at > n_ || bytes > n_ - e.at) continue;
        out.push_back(e);
    }
    if (next) {
        const uint64_t link = offset + hdr_size + count * entry_size;
        *next = big_ ? u64(link) : u32(link);
    }
    return true;
}

const uint8_t* TiffReader::entry_bytes(const TiffEntry& e, uint64_t byte_offset, std::size_t need) const {
    const uint32_t unit = tiff_type_size(e.type);
    const uint64_t total = e.count * unit;
    if (byte_offset > total || total - byte_offset < need) return nullptr;
    if (b_) return b_ + std::size_t(e.at + byte_offset);
    if (e.is_inline) return e.inline_bytes + std::size_t(byte_offset);
    if (!read_fn_) return nullptr;

    const uint64_t abs_off = e.at + byte_offset;
    if (abs_off >= cache_off_ && cache_off_ != ~uint64_t(0)) {
        const uint64_t rel = abs_off - cache_off_;
        if (rel <= cache_buf_.size() && cache_buf_.size() - rel >= need)
            return cache_buf_.data() + std::size_t(rel);
    }
    const uint64_t entry_end = e.at + total;
    const uint64_t fetch = std::min<uint64_t>(entry_end - abs_off, std::max<uint64_t>(need, 65536));
    cache_buf_.resize(std::size_t(fetch));
    if (!read_fn_(abs_off, cache_buf_.data(), std::size_t(fetch))) {
        cache_buf_.clear();
        cache_off_ = ~uint64_t(0);
        return nullptr;
    }
    cache_off_ = abs_off;
    return cache_buf_.data();
}

bool TiffReader::uint64_at(const TiffEntry& e, uint64_t i, uint64_t& v) const {
    if (i >= e.count) return false;
    const uint32_t unit = tiff_type_size(e.type);
    const uint8_t* p = entry_bytes(e, i * unit, unit);
    if (!p) return false;
    switch (e.type) {
        case kTiffByte: case kTiffSByte: case kTiffUndefined: case kTiffAscii:
            v = p[0];
            return true;
        case kTiffShort: case kTiffSShort:
            v = u16_at(p, le_);
            return true;
        case kTiffLong: case kTiffSLong: case kTiffIfd:
            v = u32_at(p, le_);
            return true;
        case kTiffLong8: case kTiffSLong8: case kTiffIfd8:
            v = u64_at(p, le_);
            return true;
        default:
            return false;
    }
}

bool TiffReader::uint_at(const TiffEntry& e, uint64_t i, uint32_t& v) const {
    uint64_t u = 0;
    if (!uint64_at(e, i, u) || u > 0xFFFFFFFFull) return false;
    v = uint32_t(u);
    return true;
}

bool TiffReader::int_at(const TiffEntry& e, uint64_t i, int64_t& v) const {
    uint64_t u = 0;
    if (!uint64_at(e, i, u)) return false;
    switch (e.type) {
        case kTiffSByte: v = int8_t(uint8_t(u)); break;
        case kTiffSShort: v = int16_t(uint16_t(u)); break;
        case kTiffSLong: v = int32_t(uint32_t(u)); break;
        case kTiffSLong8: v = int64_t(u); break;
        default: v = int64_t(u); break;
    }
    return true;
}

bool TiffReader::double_at(const TiffEntry& e, uint64_t i, double& v) const {
    if (i >= e.count) return false;
    const uint32_t unit = tiff_type_size(e.type);
    const uint8_t* p = entry_bytes(e, i * unit, unit);
    if (!p) return false;
    switch (e.type) {
        case kTiffRational: {
            const uint32_t num = u32_at(p, le_), den = u32_at(p + 4, le_);
            if (den == 0) return false;
            v = double(num) / double(den);
            return true;
        }
        case kTiffSRational: {
            const int32_t num = int32_t(u32_at(p, le_)), den = int32_t(u32_at(p + 4, le_));
            if (den == 0) return false;
            v = double(num) / double(den);
            return true;
        }
        case kTiffFloat: {
            const uint32_t bits = u32_at(p, le_);
            float f;
            std::memcpy(&f, &bits, 4);
            v = f;
            return true;
        }
        case kTiffDouble: {
            const uint64_t bits = u64_at(p, le_);
            std::memcpy(&v, &bits, 8);
            return true;
        }
        case kTiffLong8: case kTiffIfd8: {
            v = double(u64_at(p, le_));
            return true;
        }
        default: {
            int64_t s = 0;
            if (!int_at(e, i, s)) return false;
            v = double(s);
            return true;
        }
    }
}

bool TiffReader::ascii(const TiffEntry& e, std::string& out) const {
    out.clear();
    if (tiff_type_size(e.type) != 1) return false;
    if (e.count == 0) return true;
    const std::size_t max_len = std::size_t(std::min<uint64_t>(e.count, 65536));
    const uint8_t* p = entry_bytes(e, 0, max_len);
    if (!p) return false;
    std::size_t len = 0;
    while (len < max_len && p[len] != 0) ++len;
    while (len > 0 && p[len - 1] == ' ') --len;
    out.assign(reinterpret_cast<const char*>(p), len);
    return true;
}

std::string TiffReader::text(const TiffEntry& e) const {
    std::string out;
    ascii(e, out);
    return out;
}

const TiffEntry* tiff_find(const std::vector<TiffEntry>& entries, uint16_t tag) {
    for (const TiffEntry& e : entries)
        if (e.tag == tag) return &e;
    return nullptr;
}

} // namespace broimage::detail
