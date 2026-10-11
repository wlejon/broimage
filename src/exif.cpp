// EXIF metadata (exif.h): the block is found in its container, then read
// through the TIFF directory reader the pixel decoder uses.

#include "broimage/exif.h"

#include "broimage/decode.h"

#include "file_bytes.h"
#include "tiff_ifd.h"
#include "decode_heif.h"
#include "heif_parse.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace broimage {

namespace {

using detail::TiffEntry;
using detail::TiffReader;
using detail::tiff_find;

bool has_exif_prefix(const uint8_t* p, std::size_t n) {
    return n >= 6 && std::memcmp(p, "Exif\0\0", 6) == 0;
}

// The payload with an optional "Exif\0\0" prefix dropped.
bool tiff_payload(const uint8_t* p, std::size_t n, const uint8_t*& tiff, std::size_t& tiff_size) {
    if (has_exif_prefix(p, n)) {
        p += 6;
        n -= 6;
    }
    if (n < 8) return false;
    tiff = p;
    tiff_size = n;
    return true;
}

bool find_in_jpeg(const uint8_t* data, std::size_t size, const uint8_t*& tiff, std::size_t& tiff_size) {
    std::size_t i = 2;
    while (i + 4 <= size) {
        if (data[i] != 0xFF) return false;
        while (i + 1 < size && data[i + 1] == 0xFF) ++i;  // fill bytes
        if (i + 4 > size) break;
        const uint8_t marker = data[i + 1];
        if (marker == 0xDA || marker == 0xD9) return false;  // scan data or the end: no more metadata
        if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0x01) {
            i += 2;
            continue;
        }
        const std::size_t len = std::size_t(data[i + 2]) << 8 | data[i + 3];
        if (len < 2 || i + 2 + len > size) return false;
        if (marker == 0xE1 && has_exif_prefix(data + i + 4, len - 2))
            return tiff_payload(data + i + 4, len - 2, tiff, tiff_size);
        i += 2 + len;
    }
    return false;
}

uint32_t le32(const uint8_t* p) { return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24; }
uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | uint32_t(p[3]); }

bool find_in_webp(const uint8_t* data, std::size_t size, const uint8_t*& tiff, std::size_t& tiff_size) {
    std::size_t end = std::size_t(le32(data + 4)) + 8;
    if (end > size || end < 12) end = size;
    std::size_t p = 12;
    while (p + 8 <= end) {
        const std::size_t len = le32(data + p + 4);
        if (len > end - p - 8) return false;
        if (std::memcmp(data + p, "EXIF", 4) == 0) return tiff_payload(data + p + 8, len, tiff, tiff_size);
        p += 8 + len + (len & 1);
    }
    return false;
}

bool find_in_png(const uint8_t* data, std::size_t size, const uint8_t*& tiff, std::size_t& tiff_size) {
    std::size_t p = 8;
    while (p + 12 <= size) {
        const std::size_t len = be32(data + p);
        if (len > size - p - 12) return false;
        const uint8_t* type = data + p + 4;
        if (std::memcmp(type, "eXIf", 4) == 0) return tiff_payload(data + p + 8, len, tiff, tiff_size);
        if (std::memcmp(type, "IEND", 4) == 0) return false;
        p += 12 + len;
    }
    return false;
}

std::string ascii(const TiffReader& r, const std::vector<TiffEntry>& ifd, uint16_t tag) {
    const TiffEntry* e = tiff_find(ifd, tag);
    return e ? r.text(*e) : std::string();
}

std::optional<double> real(const TiffReader& r, const std::vector<TiffEntry>& ifd, uint16_t tag) {
    double v = 0;
    const TiffEntry* e = tiff_find(ifd, tag);
    if (e && r.real_at(*e, 0, v) && std::isfinite(v)) return v;
    return std::nullopt;
}

std::optional<uint32_t> integer(const TiffReader& r, const std::vector<TiffEntry>& ifd, uint16_t tag) {
    uint32_t v = 0;
    const TiffEntry* e = tiff_find(ifd, tag);
    if (e && r.uint_at(*e, 0, v)) return v;
    return std::nullopt;
}

// A sub-IFD the pointer tag names; empty when there is none or it is bad.
std::vector<TiffEntry> sub_ifd(const TiffReader& r, const std::vector<TiffEntry>& ifd, uint16_t tag) {
    std::vector<TiffEntry> out;
    if (auto off = integer(r,ifd, tag); off && *off)
        if (!r.read_ifd(*off, out)) out.clear();
    return out;
}

std::string decimal(double v) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f", v);
    std::string s = buf;
    if (s.size() > 2 && s.compare(s.size() - 2, 2, ".0") == 0) s.resize(s.size() - 2);
    return s;
}

std::string exposure_text(double t) {
    if (t < 1) {
        const double inv = 1 / t;
        const double whole = std::round(inv);
        if (t < 0.25 || std::fabs(inv - whole) < 0.01 * inv) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "1/%.0f", whole);
            return buf;
        }
    }
    return decimal(t);
}

// "YYYY:MM:DD HH:MM:SS" to "YYYY-MM-DDTHH:MM:SS"; empty when malformed or
// the all-zero placeholder cameras write when the clock is unset.
std::string iso_date(const std::string& s) {
    if (s.size() < 19) return {};
    for (int i = 0; i < 19; ++i) {
        const char c = s[std::size_t(i)];
        bool ok;
        if (i == 4 || i == 7) ok = c == ':' || c == '-';
        else if (i == 10) ok = c == ' ' || c == 'T';
        else if (i == 13 || i == 16) ok = c == ':';
        else ok = c >= '0' && c <= '9';
        if (!ok) return {};
    }
    if (s.compare(0, 4, "0000") == 0) return {};
    std::string out = s.substr(0, 19);
    out[4] = out[7] = '-';
    out[10] = 'T';
    return out;
}

bool valid_offset(const std::string& s) {
    return s.size() == 6 && (s[0] == '+' || s[0] == '-') && std::isdigit(uint8_t(s[1])) &&
           std::isdigit(uint8_t(s[2])) && s[3] == ':' && std::isdigit(uint8_t(s[4])) && std::isdigit(uint8_t(s[5]));
}

std::string trim_subsec(std::string s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\0')) s.pop_back();
    return s;
}

bool valid_subsec(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

std::optional<double> gps_coord(const TiffReader& r, const std::vector<TiffEntry>& gps, uint16_t ref_tag,
                                uint16_t tag, char negative) {
    const TiffEntry* e = tiff_find(gps, tag);
    const std::string ref = ascii(r, gps, ref_tag);
    if (!e || ref.empty() || e->count < 1) return std::nullopt;
    double v = 0, scale = 1;
    for (uint32_t i = 0; i < 3 && i < e->count; ++i, scale *= 60) {
        double part = 0;
        if (!r.real_at(*e, i, part)) return std::nullopt;
        v += part / scale;
    }
    if (!std::isfinite(v)) return std::nullopt;
    return ref[0] == negative ? -v : v;
}

} // namespace

bool find_exif(const uint8_t* data, std::size_t size, const uint8_t*& tiff, std::size_t& tiff_size) {
    tiff = nullptr;
    tiff_size = 0;
    if (!data || size < 8) return false;
    if (data[0] == 0xFF && data[1] == 0xD8) return find_in_jpeg(data, size, tiff, tiff_size);
    if (std::memcmp(data, "II*\0", 4) == 0 || std::memcmp(data, "MM\0*", 4) == 0) {
        tiff = data;
        tiff_size = size;
        return true;
    }
    if (size >= 12 && std::memcmp(data, "RIFF", 4) == 0 && std::memcmp(data + 8, "WEBP", 4) == 0)
        return find_in_webp(data, size, tiff, tiff_size);
    if (std::memcmp(data, "\x89PNG\r\n\x1a\n", 8) == 0) return find_in_png(data, size, tiff, tiff_size);
    if (detail::heif_sniff(data, size)) return detail::heif_find_exif(data, size, tiff, tiff_size);
    return false;
}

bool read_exif(const uint8_t* data, std::size_t size, ExifInfo& out) {
    out = ExifInfo{};
    const uint8_t* block = nullptr;
    std::size_t n = 0;
    if (!find_exif(data, size, block, n)) return false;
    TiffReader r;
    std::vector<TiffEntry> ifd0;
    if (!r.open(block, n) || !r.read_ifd(r.first_ifd(), ifd0)) return false;

    out.make = ascii(r, ifd0, 0x010F);
    out.model = ascii(r, ifd0, 0x0110);
    if (auto o = integer(r,ifd0, 0x0112); o && *o >= 1 && *o <= 8) out.orientation = int(*o);
    // HEIF orients with irot / imir; the Exif tag in it is to be ignored.
    if (detail::heif_sniff(data, size)) out.orientation = int(detail::heif_orientation(data, size));

    const std::vector<TiffEntry> ex = sub_ifd(r, ifd0, 0x8769);
    if (auto t = real(r, ex, 0x829A); t && *t > 0) {
        out.exposure_time = *t;
        out.exposure_time_text = exposure_text(*t);
    }
    if (auto f = real(r, ex, 0x829D); f && *f > 0) out.f_number = *f;
    if (auto iso = integer(r,ex, 0x8827); iso && *iso) out.iso = *iso;
    else if (auto speed = integer(r,ex, 0x8833); speed && *speed) out.iso = *speed;
    if (auto fl = real(r, ex, 0x920A); fl && *fl > 0) out.focal_length = *fl;
    if (auto fl35 = integer(r,ex, 0xA405); fl35 && *fl35) out.focal_length_35mm = double(*fl35);
    out.lens_make = ascii(r, ex, 0xA433);
    out.lens_model = ascii(r, ex, 0xA434);
    out.date_taken = iso_date(ascii(r, ex, 0x9003));
    if (std::string sub = trim_subsec(ascii(r, ex, 0x9291)); valid_subsec(sub)) {
        out.sub_sec_time = sub;
        if (!out.date_taken.empty()) out.date_taken += '.' + sub;
    }
    if (const std::string off = ascii(r, ex, 0x9011); valid_offset(off)) {
        out.offset_time = off;
        if (!out.date_taken.empty()) out.date_taken += off;
    }

    const std::vector<TiffEntry> gps = sub_ifd(r, ifd0, 0x8825);
    out.latitude = gps_coord(r, gps, 1, 2, 'S');
    out.longitude = gps_coord(r, gps, 3, 4, 'W');
    if (auto alt = real(r, gps, 6)) {
        const auto below = integer(r,gps, 5);
        out.altitude = below && *below == 1 ? -*alt : *alt;
    }
    return true;
}

bool read_exif_file(const std::string& path, ExifInfo& out) {
    out = ExifInfo{};
    std::vector<uint8_t> bytes;
    if (!detail::read_file_bytes(path, bytes, 4)) return false;
    // A JPEG's APP segments come before the scan: its head is enough.
    const bool jpeg = bytes.size() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8;
    if (!detail::read_file_bytes(path, bytes, jpeg ? std::size_t(256) * 1024 : ~std::size_t(0))) return false;
    return read_exif(bytes.data(), bytes.size(), out);
}

ExifOrientation read_exif_orientation(const uint8_t* data, std::size_t size) {
    // A HEIF's orientation is its irot / imir, Exif item or not.
    if (detail::heif_sniff(data, size)) return detail::heif_orientation(data, size);
    ExifInfo info;
    if (!read_exif(data, size, info) || !info.orientation) return ExifOrientation::Normal;
    return static_cast<ExifOrientation>(*info.orientation);
}

ExifOrientation read_exif_orientation_file(const std::string& path) {
    std::vector<uint8_t> bytes;
    if (!detail::read_file_bytes(path, bytes, 4)) return ExifOrientation::Normal;
    const bool jpeg = bytes.size() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8;
    if (!detail::read_file_bytes(path, bytes, jpeg ? std::size_t(256) * 1024 : ~std::size_t(0)))
        return ExifOrientation::Normal;
    return read_exif_orientation(bytes.data(), bytes.size());
}

} // namespace broimage
