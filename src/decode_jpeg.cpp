// JPEG through libjpeg-turbo (decode_jpeg.h).

#include "decode_jpeg.h"

#include <algorithm>

#if BROIMAGE_HAS_JPEG_TURBO
#include <csetjmp>
#include <cstdio>
#include <jpeglib.h>
#endif

namespace broimage::detail {

bool is_jpeg(const uint8_t* data, std::size_t size) {
    return data && size >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;
}

bool jpeg_probe(const uint8_t* data, std::size_t size, int& width, int& height, int& channels) {
    if (!is_jpeg(data, size)) return false;
    std::size_t i = 2;
    while (i + 4 <= size) {
        if (data[i] != 0xFF) return false;
        while (i + 1 < size && data[i + 1] == 0xFF) ++i;  // fill bytes
        if (i + 4 > size) break;
        const uint8_t marker = data[i + 1];
        if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0x01) {
            i += 2;
            continue;
        }
        if (marker == 0xDA || marker == 0xD9) return false;  // a scan or the end before any frame header
        const std::size_t len = std::size_t(data[i + 2]) << 8 | data[i + 3];
        if (len < 2 || i + 2 + len > size) return false;
        // SOF0..SOF15, less DHT (C4), JPG (C8) and DAC (CC).
        if (marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
            if (len < 8) return false;
            const uint8_t* p = data + i + 4;
            height = int(p[1]) << 8 | p[2];
            width = int(p[3]) << 8 | p[4];
            channels = p[5];
            return width > 0 && height > 0;
        }
        i += 2 + len;
    }
    return false;
}

#if BROIMAGE_HAS_JPEG_TURBO

namespace {

// libjpeg reports a fatal error by calling error_exit, which must not return:
// it jumps back to jpeg_decode's setjmp. Nothing between the two frames is
// C++ with a destructor (only libjpeg's C), so the jump skips no cleanup.
struct ErrorMgr {
    jpeg_error_mgr pub;  // first, so libjpeg's pointer to it is a pointer to this
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
};

void error_exit(j_common_ptr c) {
    ErrorMgr* e = reinterpret_cast<ErrorMgr*>(c->err);
    (*c->err->format_message)(c, e->message);
    std::longjmp(e->jump, 1);
}

// Warnings (a corrupt or truncated scan, which libjpeg pads and carries on
// through) are not printed: the decode stands, as stb_image's does.
void emit_message(j_common_ptr, int) {}

// stb_image's 8x8 multiply with rounding: x * y / 255.
inline uint8_t mul255(unsigned x, unsigned y) {
    const unsigned t = x * y + 128;
    return uint8_t((t + (t >> 8)) >> 8);
}

} // namespace

bool jpeg_turbo() { return true; }

bool jpeg_decode(const uint8_t* data, std::size_t size, int scale, Image& out, std::string* why) {
    out = Image{};
    out.channels = 0;
    if (!is_jpeg(data, size)) {
        if (why) *why = "not a JPEG";
        return false;
    }
    if constexpr (sizeof(unsigned long) < sizeof(std::size_t)) {  // jpeg_mem_src's length (Windows)
        if (size > std::size_t(~0UL)) {
            if (why) *why = "JPEG too large";
            return false;
        }
    }
    const unsigned scale_num = unsigned(std::clamp(scale, 1, 8));  // never written after the setjmp

    jpeg_decompress_struct cinfo;
    ErrorMgr err;
    err.message[0] = 0;
    cinfo.err = jpeg_std_error(&err.pub);
    err.pub.error_exit = error_exit;
    err.pub.emit_message = emit_message;
    if (setjmp(err.jump)) {
        jpeg_destroy_decompress(&cinfo);
        out.pixels.clear();
        out.width = out.height = out.channels = 0;
        if (why) {
            why->assign("JPEG: ");
            why->append(err.message[0] ? err.message : "decode failed");
        }
        return false;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, static_cast<unsigned long>(size));
    jpeg_read_header(&cinfo, TRUE);
    const bool cmyk = cinfo.jpeg_color_space == JCS_CMYK || cinfo.jpeg_color_space == JCS_YCCK;
    cinfo.out_color_space = cmyk ? JCS_CMYK : JCS_EXT_RGBA;
    cinfo.scale_num = scale_num;
    cinfo.scale_denom = 8;
    jpeg_start_decompress(&cinfo);

    const int w = int(cinfo.output_width), h = int(cinfo.output_height);
    const std::size_t stride = std::size_t(w) * 4;
    out.pixels.resize(stride * std::size_t(h));
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW rows[8];
        const int y = int(cinfo.output_scanline);
        const int n = std::min(8, h - y);
        for (int k = 0; k < n; ++k) rows[k] = out.pixels.data() + std::size_t(y + k) * stride;
        if (jpeg_read_scanlines(&cinfo, rows, JDIMENSION(n)) == 0) break;
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    if (cmyk) {
        for (std::size_t i = 0, n = std::size_t(w) * h; i < n; ++i) {
            uint8_t* p = out.pixels.data() + i * 4;
            const unsigned k = p[3];
            p[0] = mul255(p[0], k);
            p[1] = mul255(p[1], k);
            p[2] = mul255(p[2], k);
            p[3] = 255;
        }
    }
    out.width = w;
    out.height = h;
    out.channels = 4;
    return true;
}

#else

bool jpeg_turbo() { return false; }

bool jpeg_decode(const uint8_t*, std::size_t, int, Image& out, std::string* why) {
    out = Image{};
    out.channels = 0;
    if (why) *why = "this build has no libjpeg-turbo";
    return false;
}

#endif

} // namespace broimage::detail
