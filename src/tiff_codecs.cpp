// TIFF chunk decompressors (tiff_codecs.h).

#include "tiff_codecs.h"

#include <stb_image.h>

#include <cstring>
#include <limits>

namespace broimage::detail {

bool tiff_lzw(const uint8_t* src, std::size_t n, uint8_t* dst, std::size_t expected) {
    if (!src || n < 1) return expected == 0;
    // A pre-6.0 stream starts with a Clear code written LSB-first: 0x00 0x01.
    if (n >= 2 && src[0] == 0 && (src[1] & 1)) return false;
    constexpr uint32_t kClear = 256, kEoi = 257, kMax = 4096;
    uint16_t prefix[kMax];
    uint8_t suffix[kMax], first[kMax];
    uint16_t length[kMax];
    for (uint32_t i = 0; i < 256; ++i) {
        prefix[i] = 0xFFFF;
        suffix[i] = first[i] = uint8_t(i);
        length[i] = 1;
    }
    uint32_t next = 258, width = 9;
    uint32_t old = kMax;  // none
    std::size_t out = 0;
    uint64_t bitbuf = 0;
    uint32_t bits = 0;
    std::size_t pos = 0;
    // Writes the string `code` names at dst[out]; clipped at `expected`.
    auto emit = [&](uint32_t code) {
        const std::size_t len = length[code];
        std::size_t end = out + len;
        uint32_t c = code;
        // Walk back from the last byte; skip the part past `expected`.
        for (std::size_t i = end; i > out; --i) {
            if (i - 1 < expected) dst[i - 1] = suffix[c];
            c = prefix[c];
        }
        out = end < expected ? end : expected;
    };
    while (out < expected) {
        while (bits < width) {
            if (pos >= n) return false;  // the stream ran out before the chunk did
            bitbuf = bitbuf << 8 | src[pos++];
            bits += 8;
        }
        const uint32_t code = uint32_t(bitbuf >> (bits - width)) & ((1u << width) - 1);
        bits -= width;
        if (code == kEoi) break;
        if (code == kClear) {
            next = 258;
            width = 9;
            old = kMax;
            continue;
        }
        if (old == kMax) {
            if (code >= 256) return false;
            emit(code);
            old = code;
            continue;
        }
        uint8_t head;
        if (code < next) {
            head = first[code];
            emit(code);
        } else if (code == next) {
            head = first[old];
        } else {
            return false;
        }
        if (next < kMax) {
            prefix[next] = uint16_t(old);
            suffix[next] = head;
            first[next] = first[old];
            length[next] = uint16_t(length[old] + 1);
            if (code == next) emit(next);
            ++next;
            // Early change: the width grows when the next entry would need it,
            // one code before it is used.
            if (next + 1 >= (1u << width) && width < 12) ++width;
        } else if (code == next) {
            return false;  // a full table must be cleared before it grows
        }
        old = code;
    }
    return out >= expected;
}

bool tiff_packbits(const uint8_t* src, std::size_t n, uint8_t* dst, std::size_t expected) {
    std::size_t out = 0, pos = 0;
    while (out < expected) {
        if (pos >= n) return false;
        const int8_t h = int8_t(src[pos++]);
        if (h >= 0) {
            std::size_t len = std::size_t(h) + 1;
            if (pos + len > n) return false;
            if (len > expected - out) len = expected - out;
            std::memcpy(dst + out, src + pos, len);
            out += len;
            pos += std::size_t(h) + 1;
        } else if (h != -128) {
            if (pos >= n) return false;
            std::size_t len = std::size_t(1 - h);
            if (len > expected - out) len = expected - out;
            std::memset(dst + out, src[pos++], len);
            out += len;
        }
    }
    return true;
}

bool tiff_inflate(const uint8_t* src, std::size_t n, uint8_t* dst, std::size_t expected) {
    constexpr std::size_t kIntMax = std::size_t(std::numeric_limits<int>::max());
    if (n > kIntMax || expected > kIntMax) return false;
    // stb's buffer variant fails rather than writing past `olen`; a stream
    // that inflates to more than the chunk holds is decoded into a scratch
    // buffer one byte larger so the excess is detected and dropped.
    const int got = stbi_zlib_decode_buffer(reinterpret_cast<char*>(dst), int(expected),
                                            reinterpret_cast<const char*>(src), int(n));
    if (got >= 0) return std::size_t(got) == expected;
    int len = 0;
    char* all = stbi_zlib_decode_malloc_guesssize(reinterpret_cast<const char*>(src), int(n),
                                                  int(expected) + 1, &len);
    if (!all) return false;
    const bool ok = std::size_t(len) >= expected;
    if (ok) std::memcpy(dst, all, expected);
    stbi_image_free(all);
    return ok;
}

bool tiff_jpeg(const uint8_t* tables, std::size_t tables_n, const uint8_t* src, std::size_t n, bool rgb,
               int channels, std::vector<uint8_t>& out, int& w, int& h) {
    if (!src || n < 4 || src[0] != 0xFF || src[1] != 0xD8) return false;
    // Adobe APP14 with transform 0: the components are not YCbCr.
    static const uint8_t kAdobeRgb[] = {0xFF, 0xEE, 0x00, 0x0E, 'A', 'd', 'o', 'b', 'e',
                                        0x00, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00};
    std::vector<uint8_t> s;
    s.reserve(tables_n + n + sizeof kAdobeRgb);
    s.insert(s.end(), {0xFF, 0xD8});
    if (rgb) s.insert(s.end(), kAdobeRgb, kAdobeRgb + sizeof kAdobeRgb);
    // JPEGTables: SOI, DQT / DHT segments, EOI. Its segments go between the
    // chunk's SOI and the rest of the chunk.
    if (tables && tables_n >= 4 && tables[0] == 0xFF && tables[1] == 0xD8) {
        std::size_t end = tables_n;
        if (tables[end - 2] == 0xFF && tables[end - 1] == 0xD9) end -= 2;
        s.insert(s.end(), tables + 2, tables + end);
    }
    std::size_t i = 2;
    if (rgb) {
        // The photometric tag is the authority: a JFIF or Adobe marker in the
        // chunk that says otherwise is dropped.
        while (i + 4 <= n && src[i] == 0xFF) {
            const uint8_t m = src[i + 1];
            if (m == 0xDA || m < 0xC0) break;
            const std::size_t len = std::size_t(src[i + 2]) << 8 | src[i + 3];
            if (len < 2 || i + 2 + len > n) break;
            if (m != 0xE0 && m != 0xEE) s.insert(s.end(), src + i, src + i + 2 + len);
            i += 2 + len;
        }
    }
    s.insert(s.end(), src + i, src + n);
    if (s.size() > std::size_t(std::numeric_limits<int>::max())) return false;
    int c = 0;
    stbi_uc* px = stbi_load_from_memory(s.data(), int(s.size()), &w, &h, &c, channels);
    if (!px) return false;
    out.assign(px, px + std::size_t(w) * std::size_t(h) * std::size_t(channels));
    stbi_image_free(px);
    return true;
}

} // namespace broimage::detail
