// Windows icons (.ico): the container is read by broimage itself (stb has no
// reader for it). Each case builds an icon in memory and decodes it through
// the public API: the largest entry wins, PNG and BMP entries both decode, the
// AND mask makes pixels transparent, and the probe agrees with the decode.
#include "broimage/decode.h"
#include "broimage/encode.h"

#include <cstdint>
#include <cstdio>
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

void put16(std::vector<uint8_t>& v, uint32_t x) { v.push_back(uint8_t(x)); v.push_back(uint8_t(x >> 8)); }
void put32(std::vector<uint8_t>& v, uint32_t x) { put16(v, x & 0xFFFF); put16(v, x >> 16); }

struct IcoImage {
    int w, h, bpp;
    std::vector<uint8_t> bytes;  // a whole PNG file or a headerless DIB
};

std::vector<uint8_t> make_ico(const std::vector<IcoImage>& images) {
    std::vector<uint8_t> out;
    put16(out, 0); put16(out, 1); put16(out, uint32_t(images.size()));
    uint32_t off = 6 + 16 * uint32_t(images.size());
    for (const auto& im : images) {
        out.push_back(uint8_t(im.w >= 256 ? 0 : im.w));
        out.push_back(uint8_t(im.h >= 256 ? 0 : im.h));
        out.push_back(0); out.push_back(0);
        put16(out, 1); put16(out, uint32_t(im.bpp));
        put32(out, uint32_t(im.bytes.size())); put32(out, off);
        off += uint32_t(im.bytes.size());
    }
    for (const auto& im : images) out.insert(out.end(), im.bytes.begin(), im.bytes.end());
    return out;
}

// A 32-bit BMP entry from top-down RGBA, with an AND mask marking the pixels
// `masked` says are transparent (and their alpha zeroed when `useAlpha` is
// false, as an old icon that only has the mask would store it).
IcoImage make_dib32(int w, int h, const std::vector<uint8_t>& rgba, bool useAlpha,
                    const std::vector<bool>& masked) {
    IcoImage im{w, h, 32, {}};
    auto& v = im.bytes;
    put32(v, 40); put32(v, uint32_t(w)); put32(v, uint32_t(h * 2));
    put16(v, 1); put16(v, 32); put32(v, 0); put32(v, 0);
    put32(v, 0); put32(v, 0); put32(v, 0); put32(v, 0);
    for (int y = h - 1; y >= 0; --y)
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = &rgba[(std::size_t(y) * w + x) * 4];
            v.push_back(p[2]); v.push_back(p[1]); v.push_back(p[0]);
            v.push_back(useAlpha ? p[3] : 0);
        }
    const int stride = ((w + 31) / 32) * 4;
    for (int y = h - 1; y >= 0; --y) {
        std::vector<uint8_t> row(stride, 0);
        for (int x = 0; x < w; ++x)
            if (masked[std::size_t(y) * w + x]) row[x >> 3] |= uint8_t(0x80 >> (x & 7));
        v.insert(v.end(), row.begin(), row.end());
    }
    return im;
}

// An 8-bit palettized BMP entry: two palette colours, left half index 0,
// right half index 1, the bottom row masked out.
IcoImage make_dib8(int w, int h) {
    IcoImage im{w, h, 8, {}};
    auto& v = im.bytes;
    put32(v, 40); put32(v, uint32_t(w)); put32(v, uint32_t(h * 2));
    put16(v, 1); put16(v, 8); put32(v, 0); put32(v, 0);
    put32(v, 0); put32(v, 0); put32(v, 2); put32(v, 0);
    v.insert(v.end(), {0x00, 0x00, 0xFF, 0x00});  // index 0: red (BGRx)
    v.insert(v.end(), {0xFF, 0x00, 0x00, 0x00});  // index 1: blue
    const int stride = ((w * 8 + 31) / 32) * 4;
    for (int y = h - 1; y >= 0; --y) {
        std::vector<uint8_t> row(stride, 0);
        for (int x = 0; x < w; ++x) row[x] = x < w / 2 ? 0 : 1;
        v.insert(v.end(), row.begin(), row.end());
    }
    const int mstride = ((w + 31) / 32) * 4;
    for (int y = h - 1; y >= 0; --y) {
        std::vector<uint8_t> row(mstride, 0);
        if (y == h - 1) for (int x = 0; x < w; ++x) row[x >> 3] |= uint8_t(0x80 >> (x & 7));
        v.insert(v.end(), row.begin(), row.end());
    }
    return im;
}

std::vector<uint8_t> solid(int w, int h, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    std::vector<uint8_t> px(std::size_t(w) * h * 4);
    for (std::size_t i = 0; i < px.size(); i += 4) { px[i] = r; px[i + 1] = g; px[i + 2] = b; px[i + 3] = a; }
    return px;
}

const uint8_t* at(const broimage::Image& im, int x, int y) {
    return &im.pixels[(std::size_t(y) * im.width + x) * 4];
}

void test_png_entry_largest_wins() {
    // 16x16 BMP (green) and a 48x32 PNG (blue): the PNG is the larger one.
    auto green = solid(16, 16, 0, 255, 0, 255);
    IcoImage small = make_dib32(16, 16, green, true, std::vector<bool>(256, false));
    IcoImage png{48, 32, 32, {}};
    auto blue = solid(48, 32, 0, 0, 255, 200);
    CHECK(broimage::encode_png_memory(png.bytes, blue.data(), 48, 32, 4));
    const auto ico = make_ico({small, png});

    int w = 0, h = 0, c = 0;
    CHECK(broimage::probe_dimensions_memory(ico.data(), ico.size(), &w, &h, &c));
    CHECK(w == 48 && h == 32);

    broimage::Image im;
    std::string err;
    CHECK(broimage::decode_memory(ico.data(), ico.size(), im, &err));
    CHECK(im.width == 48 && im.height == 32 && im.channels == 4);
    if (im.width == 48) {
        const uint8_t* p = at(im, 10, 10);
        CHECK(p[0] == 0 && p[1] == 0 && p[2] == 255 && p[3] == 200);
    }

    // The probe reads a header-sized head of a file; the entry past the end
    // of that head still sizes the image.
    const std::size_t head = 6 + 16 * 2 + small.bytes.size() + 40;
    CHECK(broimage::probe_dimensions_memory(ico.data(), head, &w, &h, &c));
    CHECK(w == 48 && h == 32);
}

void test_bmp32_alpha_and_mask() {
    // Real alpha: kept as stored, the mask ignored.
    auto px = solid(8, 4, 10, 20, 30, 128);
    std::vector<bool> masked(32, false);
    masked[0] = true;
    auto ico = make_ico({make_dib32(8, 4, px, true, masked)});
    broimage::Image im;
    CHECK(broimage::decode_memory(ico.data(), ico.size(), im));
    CHECK(im.width == 8 && im.height == 4);
    if (im.width == 8) {
        CHECK(at(im, 3, 2)[0] == 10 && at(im, 3, 2)[1] == 20 && at(im, 3, 2)[2] == 30);
        CHECK(at(im, 3, 2)[3] == 128);
        CHECK(at(im, 0, 0)[3] == 128);
    }

    // No alpha channel in use: the AND mask decides, top-left transparent.
    ico = make_ico({make_dib32(8, 4, px, false, masked)});
    CHECK(broimage::decode_memory(ico.data(), ico.size(), im));
    if (im.width == 8) {
        CHECK(at(im, 0, 0)[3] == 0);
        CHECK(at(im, 1, 0)[3] == 255);
        CHECK(at(im, 7, 3)[3] == 255);
    }
}

void test_bmp8_palette() {
    // 20 wide: row strides that need padding on both the colour and mask rows.
    auto ico = make_ico({make_dib8(20, 6)});
    broimage::Image im;
    CHECK(broimage::decode_memory(ico.data(), ico.size(), im));
    CHECK(im.width == 20 && im.height == 6);
    if (im.width == 20) {
        const uint8_t* l = at(im, 2, 0);
        const uint8_t* r = at(im, 15, 0);
        CHECK(l[0] == 255 && l[1] == 0 && l[2] == 0 && l[3] == 255);
        CHECK(r[0] == 0 && r[1] == 0 && r[2] == 255 && r[3] == 255);
        CHECK(at(im, 2, 5)[3] == 0);  // bottom row masked out
    }
}

void test_truncated_and_garbage() {
    auto ico = make_ico({make_dib8(20, 6)});
    broimage::Image im;
    // Cut inside the pixel data: not decodable, and says so.
    std::string err;
    CHECK(!broimage::decode_memory(ico.data(), ico.size() - 40, im, &err));
    // An icon header pointing nowhere.
    std::vector<uint8_t> bad = {0, 0, 1, 0, 1, 0};
    bad.resize(6 + 16, 0);
    bad[6 + 8] = 100;  // length 100, offset 0
    CHECK(!broimage::decode_memory(bad.data(), bad.size(), im));
    int w = 0, h = 0, c = 0;
    CHECK(!broimage::probe_dimensions_memory(bad.data(), bad.size(), &w, &h, &c));
}

void test_bounded() {
    auto ico = make_ico({make_dib8(20, 6)});
    broimage::Animation anim;
    CHECK(broimage::decode_memory_bounded(ico.data(), ico.size(), {}, anim));
    CHECK(anim.width == 20 && anim.height == 6 && anim.frames.size() == 1);
    broimage::DecodeLimits lim;
    lim.max_width = 10;
    CHECK(!broimage::decode_memory_bounded(ico.data(), ico.size(), lim, anim));
}

void test_decode_file() {
    auto ico = make_ico({make_dib8(20, 6)});
    const std::string path = "broimage_test_decode_ico.ico";
    std::FILE* f = std::fopen(path.c_str(), "wb");
    CHECK(f != nullptr);
    if (!f) return;
    std::fwrite(ico.data(), 1, ico.size(), f);
    std::fclose(f);
    broimage::Image im;
    CHECK(broimage::decode_file(path, im));
    CHECK(im.width == 20 && im.height == 6);
    std::remove(path.c_str());
}

} // namespace

int main() {
    test_png_entry_largest_wins();
    test_bmp32_alpha_and_mask();
    test_bmp8_palette();
    test_truncated_and_garbage();
    test_bounded();
    test_decode_file();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test_decode_ico: all passed\n");
    return 0;
}
