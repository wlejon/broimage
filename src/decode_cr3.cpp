// Canon CR3's embedded JPEGs and metadata (decode_cr3.h).

#include "decode_cr3.h"

#include <cstring>

namespace broimage::detail {

namespace {

uint32_t be32(const uint8_t* p) { return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3]; }
uint64_t be64(const uint8_t* p) { return uint64_t(be32(p)) << 32 | be32(p + 4); }

const uint8_t kCanonUuid[16] = {0x85, 0xc0, 0xb6, 0x87, 0x82, 0x0f, 0x11, 0xe0,
                                0x81, 0x11, 0xf4, 0xce, 0x46, 0x2b, 0x6a, 0x48};
const uint8_t kPreviewUuid[16] = {0xea, 0xf4, 0x2b, 0x5e, 0x1c, 0x98, 0x4b, 0x88,
                                  0xb9, 0xfb, 0xb7, 0xdc, 0x40, 0x6e, 0x4d, 0x16};

// A box: its type and payload (after the header, and a uuid's 16 bytes).
struct Box {
    char type[4];
    const uint8_t* uuid = nullptr;
    const uint8_t* body = nullptr;
    std::size_t size = 0;
};

// Walks the boxes in [p, end).
struct Boxes {
    const uint8_t* p;
    const uint8_t* end;
    bool next(Box& b) {
        if (end - p < 8) return false;
        uint64_t len = be32(p);
        std::size_t head = 8;
        if (len == 1) {
            if (end - p < 16) return false;
            len = be64(p + 8);
            head = 16;
        } else if (len == 0) {
            len = uint64_t(end - p);
        }
        if (len < head || len > uint64_t(end - p)) return false;
        std::memcpy(b.type, p + 4, 4);
        b.uuid = nullptr;
        if (std::memcmp(b.type, "uuid", 4) == 0) {
            if (len < head + 16) return false;
            b.uuid = p + head;
            head += 16;
        }
        b.body = p + head;
        b.size = std::size_t(len) - head;
        p += len;
        return true;
    }
};

bool is(const Box& b, const char* type) { return std::memcmp(b.type, type, 4) == 0; }

bool find(const uint8_t* p, std::size_t n, const char* type, Box& out) {
    Boxes it{p, p + n};
    while (it.next(out))
        if (is(out, type)) return true;
    return false;
}

// A JPEG `len` bytes long at `p`, within `end`: the span, or none.
Cr3Span jpeg_at(const uint8_t* p, std::size_t len, const uint8_t* end) {
    if (len < 4 || len > std::size_t(end - p) || p[0] != 0xFF || p[1] != 0xD8) return {};
    return Cr3Span{p, len};
}

// THMB: version/flags, width, height, JPEG length, two shorts, the JPEG.
Cr3Span thumbnail(const Box& b) {
    if (b.size < 16) return {};
    return jpeg_at(b.body + 16, be32(b.body + 8), b.body + b.size);
}

// PRVW: a long, a short, width, height, a short, JPEG length, the JPEG.
Cr3Span preview(const Box& b) {
    if (b.size < 16) return {};
    return jpeg_at(b.body + 16, be32(b.body + 12), b.body + b.size);
}

// trak[0]'s first sample, by its stbl: stsz for the size, co64 (or stco)
// for where it is.
Cr3Span first_sample(const uint8_t* data, std::size_t size, const Box& trak) {
    Box mdia, minf, stbl, stsz, co;
    if (!find(trak.body, trak.size, "mdia", mdia) || !find(mdia.body, mdia.size, "minf", minf) ||
        !find(minf.body, minf.size, "stbl", stbl) || !find(stbl.body, stbl.size, "stsz", stsz) || stsz.size < 12)
        return {};
    uint32_t len = be32(stsz.body + 4);
    if (len == 0) {
        if (stsz.size < 16 || be32(stsz.body + 8) == 0) return {};
        len = be32(stsz.body + 12);
    }
    uint64_t off = 0;
    if (find(stbl.body, stbl.size, "co64", co) && co.size >= 16 && be32(co.body + 4) > 0) off = be64(co.body + 8);
    else if (find(stbl.body, stbl.size, "stco", co) && co.size >= 12 && be32(co.body + 4) > 0) off = be32(co.body + 8);
    else return {};
    if (off >= size) return {};
    return jpeg_at(data + off, len, data + size);
}

} // namespace

bool cr3_sniff(const uint8_t* data, std::size_t size) {
    return data && size >= 12 && std::memcmp(data + 4, "ftypcrx ", 8) == 0;
}

bool cr3_parts(const uint8_t* data, std::size_t size, Cr3Parts& out) {
    out = Cr3Parts{};
    if (!cr3_sniff(data, size)) return false;
    Boxes top{data, data + size};
    Box b;
    while (top.next(b)) {
        if (is(b, "moov")) {
            Boxes in{b.body, b.body + b.size};
            Box c;
            bool first_trak = true;
            while (in.next(c)) {
                if (c.uuid && std::memcmp(c.uuid, kCanonUuid, 16) == 0) {
                    Boxes canon{c.body, c.body + c.size};
                    Box d;
                    while (canon.next(d)) {
                        if (std::memcmp(d.type, "CMT", 3) == 0 && d.type[3] >= '1' && d.type[3] <= '4')
                            out.cmt[d.type[3] - '1'] = Cr3Span{d.body, d.size};
                        else if (is(d, "THMB"))
                            out.thumbnail = thumbnail(d);
                    }
                } else if (is(c, "trak") && first_trak) {
                    first_trak = false;
                    out.full = first_sample(data, size, c);
                    // tkhd: width and height, 16.16 fixed point, close its end.
                    Box tkhd;
                    if (find(c.body, c.size, "tkhd", tkhd) && tkhd.size >= 84) {
                        const std::size_t at = tkhd.body[0] == 1 ? 88 : 76;
                        if (tkhd.size >= at + 8) {
                            out.full_width = int(be32(tkhd.body + at) >> 16);
                            out.full_height = int(be32(tkhd.body + at + 4) >> 16);
                        }
                    }
                }
            }
        } else if (b.uuid && std::memcmp(b.uuid, kPreviewUuid, 16) == 0 && b.size > 8) {
            Box d;
            if (find(b.body + 8, b.size - 8, "PRVW", d)) out.preview = preview(d);
        }
    }
    return true;
}

} // namespace broimage::detail
