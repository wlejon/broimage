#include "broimage/png_text.h"

#include "broimage/encode.h"

#include <stb_image.h>

#include <array>
#include <cstdlib>
#include <cstring>

namespace broimage {

namespace {

constexpr uint8_t kSignature[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
// Inflated text larger than this is treated as malformed (a zip bomb in a tag).
constexpr int kMaxInflatedText = 16 * 1024 * 1024;

const std::array<uint32_t, 256>& crc_table() {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t n = 0; n < 256; ++n) {
            uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    return table;
}

uint32_t crc32(const uint8_t* p, std::size_t n, uint32_t crc = 0) {
    const auto& t = crc_table();
    crc ^= 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) crc = t[(crc ^ p[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

void put_be32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(uint8_t(x >> 24));
    v.push_back(uint8_t(x >> 16));
    v.push_back(uint8_t(x >> 8));
    v.push_back(uint8_t(x));
}

bool valid_keyword(const std::string& k) {
    if (k.empty() || k.size() > 79) return false;
    if (k.front() == ' ' || k.back() == ' ') return false;
    for (unsigned char c : k) {
        if (!((c >= 32 && c <= 126) || c >= 161)) return false;
    }
    return true;
}

bool inflate_zlib(const uint8_t* p, std::size_t n, std::string& out) {
    if (n == 0 || n > static_cast<std::size_t>(INT32_MAX)) return false;
    int len = 0;
    char* buf = stbi_zlib_decode_malloc(reinterpret_cast<const char*>(p), static_cast<int>(n), &len);
    if (!buf) return false;
    bool ok = len >= 0 && len <= kMaxInflatedText;
    if (ok) out.assign(buf, static_cast<std::size_t>(len));
    std::free(buf);
    return ok;
}

// Keyword up to the first NUL; returns the offset just past it, or 0 when malformed.
std::size_t split_keyword(const uint8_t* body, std::size_t n, std::string& keyword) {
    const void* nul = std::memchr(body, 0, n);
    if (!nul) return 0;
    std::size_t klen = static_cast<std::size_t>(static_cast<const uint8_t*>(nul) - body);
    if (klen == 0 || klen > 79) return 0;
    keyword.assign(reinterpret_cast<const char*>(body), klen);
    return klen + 1;
}

bool parse_text(const char* type, const uint8_t* body, std::size_t n, PngTextEntry& e) {
    std::size_t at = split_keyword(body, n, e.keyword);
    if (at == 0) return false;
    if (std::memcmp(type, "tEXt", 4) == 0) {
        e.text.assign(reinterpret_cast<const char*>(body + at), n - at);
        return true;
    }
    if (std::memcmp(type, "zTXt", 4) == 0) {
        if (at >= n || body[at] != 0) return false; // compression method 0 = zlib
        return inflate_zlib(body + at + 1, n - at - 1, e.text);
    }
    // iTXt: compression flag, method, language tag\0, translated keyword\0, text.
    if (at + 2 > n) return false;
    uint8_t flag = body[at];
    uint8_t method = body[at + 1];
    at += 2;
    for (int skip = 0; skip < 2; ++skip) {
        const void* nul = std::memchr(body + at, 0, n - at);
        if (!nul) return false;
        at = static_cast<std::size_t>(static_cast<const uint8_t*>(nul) - body) + 1;
    }
    if (flag == 0) {
        e.text.assign(reinterpret_cast<const char*>(body + at), n - at);
        return true;
    }
    if (flag != 1 || method != 0) return false;
    return inflate_zlib(body + at, n - at, e.text);
}

void append_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& body) {
    put_be32(out, static_cast<uint32_t>(body.size()));
    std::size_t crc_start = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), body.begin(), body.end());
    put_be32(out, crc32(out.data() + crc_start, out.size() - crc_start));
}

} // namespace

bool read_png_info(const uint8_t* data, std::size_t size, PngInfo& out, std::string* error) {
    auto fail = [&](const char* why) {
        if (error) *error = why;
        return false;
    };
    out = PngInfo{};
    if (!data || size < 8 + 25 || std::memcmp(data, kSignature, 8) != 0) return fail("not a PNG");
    if (be32(data + 8) != 13 || std::memcmp(data + 12, "IHDR", 4) != 0) return fail("PNG has no IHDR");
    const uint8_t* ihdr = data + 16;
    out.width = static_cast<int>(be32(ihdr));
    out.height = static_cast<int>(be32(ihdr + 4));
    out.bit_depth = ihdr[8];
    out.color_type = ihdr[9];
    if (out.width <= 0 || out.height <= 0) return fail("PNG has an invalid size");

    std::size_t pos = 8 + 25;
    while (pos + 12 <= size) {
        uint32_t len = be32(data + pos);
        if (len > size - pos - 12) break; // truncated
        const char* type = reinterpret_cast<const char*>(data + pos + 4);
        const uint8_t* body = data + pos + 8;
        if (std::memcmp(type, "IEND", 4) == 0) break;
        bool is_text = std::memcmp(type, "tEXt", 4) == 0 || std::memcmp(type, "zTXt", 4) == 0 ||
                       std::memcmp(type, "iTXt", 4) == 0;
        if (is_text && crc32(data + pos + 4, len + 4) == be32(body + len)) {
            PngTextEntry e;
            if (parse_text(type, body, len, e)) out.text.push_back(std::move(e));
        }
        pos += 12 + static_cast<std::size_t>(len);
    }
    return true;
}

bool encode_png_memory_with_text(std::vector<uint8_t>& out,
                                 const uint8_t* pixels, int width, int height, int channels,
                                 const std::vector<PngTextEntry>& text, int stride_bytes) {
    for (const auto& e : text) {
        if (!valid_keyword(e.keyword)) return false;
    }
    std::vector<uint8_t> png;
    if (!encode_png_memory(png, pixels, width, height, channels, stride_bytes)) return false;
    constexpr std::size_t kAfterIhdr = 8 + 25;
    if (png.size() < kAfterIhdr || std::memcmp(png.data() + 12, "IHDR", 4) != 0) return false;

    std::vector<uint8_t> chunks;
    for (const auto& e : text) {
        bool ascii = true;
        for (unsigned char c : e.text) ascii = ascii && c < 0x80;
        std::vector<uint8_t> body(e.keyword.begin(), e.keyword.end());
        body.push_back(0);
        if (ascii) {
            body.insert(body.end(), e.text.begin(), e.text.end());
            append_chunk(chunks, "tEXt", body);
        } else {
            body.insert(body.end(), {0, 0, 0, 0}); // uncompressed, method 0, no language, no translation
            body.insert(body.end(), e.text.begin(), e.text.end());
            append_chunk(chunks, "iTXt", body);
        }
    }

    out.clear();
    out.reserve(png.size() + chunks.size());
    out.insert(out.end(), png.begin(), png.begin() + kAfterIhdr);
    out.insert(out.end(), chunks.begin(), chunks.end());
    out.insert(out.end(), png.begin() + kAfterIhdr, png.end());
    return true;
}

} // namespace broimage
