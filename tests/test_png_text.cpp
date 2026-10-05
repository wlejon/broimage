// PNG text chunks: tEXt / iTXt written by encode_png_memory_with_text, and
// tEXt / zTXt / iTXt (plain and compressed) read back by read_png_info. The
// compressed chunks are built by hand here (zlib stored blocks) so the reader
// is checked against bytes this library did not write.
#include "broimage/buffer.h"
#include "broimage/decode.h"
#include "broimage/encode.h"
#include "broimage/png_text.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

namespace {

uint32_t crc32(const uint8_t* p, std::size_t n) {
    uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) {
        c ^= p[i];
        for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    }
    return c ^ 0xFFFFFFFFu;
}

void put_be32(std::vector<uint8_t>& v, uint32_t x) {
    for (int s = 24; s >= 0; s -= 8) v.push_back(uint8_t(x >> s));
}

// zlib stream holding one stored (uncompressed) deflate block.
std::vector<uint8_t> zlib_stored(const std::string& s) {
    std::vector<uint8_t> z = {0x78, 0x01, 0x01};
    uint16_t len = static_cast<uint16_t>(s.size());
    z.push_back(uint8_t(len));
    z.push_back(uint8_t(len >> 8));
    z.push_back(uint8_t(~len));
    z.push_back(uint8_t(~len >> 8));
    z.insert(z.end(), s.begin(), s.end());
    uint32_t a = 1, b = 0;
    for (unsigned char c : s) {
        a = (a + c) % 65521;
        b = (b + a) % 65521;
    }
    put_be32(z, (b << 16) | a);
    return z;
}

std::vector<uint8_t> chunk(const char* type, const std::vector<uint8_t>& body) {
    std::vector<uint8_t> c;
    put_be32(c, static_cast<uint32_t>(body.size()));
    c.insert(c.end(), type, type + 4);
    c.insert(c.end(), body.begin(), body.end());
    put_be32(c, crc32(c.data() + 4, body.size() + 4));
    return c;
}

std::vector<uint8_t> bytes(const std::string& s) { return {s.begin(), s.end()}; }

std::vector<uint8_t> concat(std::vector<uint8_t> a, const std::vector<uint8_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

} // namespace

int main() {
    const int W = 5, H = 3;
    std::vector<uint8_t> px(W * H * 4);
    for (std::size_t i = 0; i < px.size(); ++i) px[i] = static_cast<uint8_t>(i * 7);

    // ---- write, then read back ------------------------------------------
    std::vector<broimage::PngTextEntry> tags = {
        {"Thumb::URI", "file:///tmp/a%20b.png"},
        {"Thumb::MTime", "1700000000"},
        {"Title", "caf\xC3\xA9 \xE7\x94\xBB"}, // non-ASCII: written as iTXt
        {"Empty", ""},
    };
    std::vector<uint8_t> png;
    CHECK(broimage::encode_png_memory_with_text(png, px.data(), W, H, 4, tags));

    broimage::PngInfo info;
    std::string err;
    CHECK(broimage::read_png_info(png.data(), png.size(), info, &err));
    CHECK(info.width == W && info.height == H);
    CHECK(info.bit_depth == 8 && info.color_type == 6);
    CHECK(info.text.size() == tags.size());
    for (std::size_t i = 0; i < tags.size() && i < info.text.size(); ++i) {
        CHECK(info.text[i].keyword == tags[i].keyword);
        CHECK(info.text[i].text == tags[i].text);
    }
    // The iTXt went in as iTXt, the ASCII tags as tEXt.
    auto has = [&](const char* type) {
        for (std::size_t i = 0; i + 4 <= png.size(); ++i)
            if (std::memcmp(png.data() + i, type, 4) == 0) return true;
        return false;
    };
    CHECK(has("iTXt") && has("tEXt"));

    // Pixels are untouched by the inserted chunks.
    broimage::Image img;
    CHECK(broimage::decode_memory(png.data(), png.size(), img, &err));
    CHECK(img.width == W && img.height == H && img.pixels == px);

    // ---- a plain PNG has no text ----------------------------------------
    std::vector<uint8_t> plain;
    CHECK(broimage::encode_png_memory(plain, px.data(), W, H, 4));
    CHECK(broimage::read_png_info(plain.data(), plain.size(), info));
    CHECK(info.text.empty() && info.width == W);

    // ---- invalid keywords are refused -----------------------------------
    std::vector<uint8_t> out;
    CHECK(!broimage::encode_png_memory_with_text(out, px.data(), W, H, 4, {{"", "x"}}));
    CHECK(!broimage::encode_png_memory_with_text(out, px.data(), W, H, 4, {{std::string(80, 'k'), "x"}}));
    CHECK(!broimage::encode_png_memory_with_text(out, px.data(), W, H, 4, {{"tab\tkey", "x"}}));
    CHECK(!broimage::encode_png_memory_with_text(out, px.data(), W, H, 4, {{" lead", "x"}}));
    CHECK(broimage::encode_png_memory_with_text(out, px.data(), W, H, 4, {{std::string(79, 'k'), "x"}}));

    // ---- hand-built zTXt, compressed iTXt, plain iTXt, bad CRC ----------
    {
        const std::size_t after_ihdr = 8 + 25;
        std::vector<uint8_t> head(plain.begin(), plain.begin() + after_ihdr);
        std::vector<uint8_t> tail(plain.begin() + after_ihdr, plain.end());

        std::vector<uint8_t> ztxt = concat(bytes(std::string("Comment\0\0", 9)), zlib_stored("squeezed text"));
        std::vector<uint8_t> itxt_z = concat(bytes(std::string("Author\0\x01\x00" "en\0Autor\0", 18)),
                                             zlib_stored("\xC3\xBC" "ber"));
        std::vector<uint8_t> itxt = bytes(std::string("Note\0\0\0\0\0plain itxt", 19));
        std::vector<uint8_t> bad = chunk("tEXt", bytes(std::string("Bad\0crc", 7)));
        bad.back() ^= 0xFF;
        // A non-text ancillary chunk between them is walked over.
        std::vector<uint8_t> phys = chunk("pHYs", {0, 0, 0x0B, 0x13, 0, 0, 0x0B, 0x13, 1});

        std::vector<uint8_t> f = head;
        for (const auto& c : {chunk("zTXt", ztxt), phys, chunk("iTXt", itxt_z), bad, chunk("iTXt", itxt)})
            f = concat(f, c);
        f = concat(f, tail);

        CHECK(broimage::read_png_info(f.data(), f.size(), info, &err));
        CHECK(info.text.size() == 3);
        if (info.text.size() == 3) {
            CHECK(info.text[0].keyword == "Comment" && info.text[0].text == "squeezed text");
            CHECK(info.text[1].keyword == "Author" && info.text[1].text == "\xC3\xBC" "ber");
            CHECK(info.text[2].keyword == "Note" && info.text[2].text == "plain itxt");
        }
        // Still a decodable image.
        CHECK(broimage::decode_memory(f.data(), f.size(), img, &err) && img.pixels == px);

        // A truncated file keeps what was read before the cut.
        std::vector<uint8_t> cut(f.begin(), f.begin() + static_cast<std::ptrdiff_t>(after_ihdr + chunk("zTXt", ztxt).size() + 5));
        CHECK(broimage::read_png_info(cut.data(), cut.size(), info));
        CHECK(info.text.size() == 1);
    }

    // ---- not a PNG ------------------------------------------------------
    const uint8_t junk[40] = {'G', 'I', 'F', '8', '9', 'a'};
    CHECK(!broimage::read_png_info(junk, sizeof(junk), info, &err) && !err.empty());
    CHECK(!broimage::read_png_info(plain.data(), 20, info));

    if (g_failed) {
        std::fprintf(stderr, "broimage_test_png_text: %d failure(s)\n", g_failed);
        return 1;
    }
    std::printf("broimage_test_png_text: OK\n");
    return 0;
}
