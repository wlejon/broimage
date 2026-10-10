#pragma once
// Test helper: a small GIF89a writer. Every frame is LZW-coded as literals
// with a clear code often enough that the code size never grows, which any
// conforming decoder reads; interlaced frames are written in pass order.

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace gifw {

struct Frame {
    int x = 0, y = 0, w = 0, h = 0;
    std::vector<uint8_t> idx;  // w * h, row-major, top-down
    std::vector<uint8_t> lct;  // RGB triples; empty: the global table
    int disposal = 0;
    int transparent = -1;
    int delay_cs = 0;
    bool interlace = false;
    bool gce = true;
};

inline int table_bits(std::size_t colours) {
    int b = 1;
    while ((std::size_t(1) << b) < colours) ++b;
    return b;
}

inline void put16(std::vector<uint8_t>& o, int v) {
    o.push_back(uint8_t(v));
    o.push_back(uint8_t(v >> 8));
}

inline void put_table(std::vector<uint8_t>& o, const std::vector<uint8_t>& t) {
    const int bits = table_bits(t.size() / 3);
    std::vector<uint8_t> padded = t;
    padded.resize(std::size_t(3) << bits, 0);
    o.insert(o.end(), padded.begin(), padded.end());
}

inline void put_lzw(std::vector<uint8_t>& o, const std::vector<uint8_t>& idx, int min_size) {
    o.push_back(uint8_t(min_size));
    const int clear = 1 << min_size, eoi = clear + 1, size = min_size + 1;
    std::vector<uint8_t> bytes;
    uint32_t acc = 0;
    int nbits = 0;
    auto code = [&](int c) {
        acc |= uint32_t(c) << nbits;
        nbits += size;
        while (nbits >= 8) {
            bytes.push_back(uint8_t(acc));
            acc >>= 8;
            nbits -= 8;
        }
    };
    const int run = clear - 2;
    code(clear);
    int since = 0;
    for (uint8_t v : idx) {
        if (since == run) {
            code(clear);
            since = 0;
        }
        code(v);
        ++since;
    }
    code(eoi);
    if (nbits > 0) bytes.push_back(uint8_t(acc));
    for (std::size_t i = 0; i < bytes.size(); i += 255) {
        const std::size_t n = std::min<std::size_t>(255, bytes.size() - i);
        o.push_back(uint8_t(n));
        o.insert(o.end(), bytes.begin() + std::ptrdiff_t(i), bytes.begin() + std::ptrdiff_t(i + n));
    }
    o.push_back(0);
}

// loop: -1 writes no NETSCAPE2.0 extension, otherwise its stored count.
inline std::vector<uint8_t> make_gif(int W, int H, const std::vector<uint8_t>& gct, int loop,
                                     const std::vector<Frame>& frames) {
    std::vector<uint8_t> o = {'G', 'I', 'F', '8', '9', 'a'};
    put16(o, W);
    put16(o, H);
    o.push_back(gct.empty() ? 0 : uint8_t(0x80 | 0x70 | (table_bits(gct.size() / 3) - 1)));
    o.push_back(0);  // background index
    o.push_back(0);  // aspect
    if (!gct.empty()) put_table(o, gct);
    if (loop >= 0) {
        const std::string app = "NETSCAPE2.0";
        o.insert(o.end(), {0x21, 0xFF, 11});
        o.insert(o.end(), app.begin(), app.end());
        o.insert(o.end(), {3, 1});
        put16(o, loop);
        o.push_back(0);
    }
    for (const Frame& f : frames) {
        if (f.gce) {
            o.insert(o.end(), {0x21, 0xF9, 4});
            o.push_back(uint8_t((f.disposal << 2) | (f.transparent >= 0 ? 1 : 0)));
            put16(o, f.delay_cs);
            o.push_back(uint8_t(f.transparent >= 0 ? f.transparent : 0));
            o.push_back(0);
        }
        o.push_back(0x2C);
        put16(o, f.x);
        put16(o, f.y);
        put16(o, f.w);
        put16(o, f.h);
        uint8_t flags = f.interlace ? 0x40 : 0;
        if (!f.lct.empty()) flags |= uint8_t(0x80 | (table_bits(f.lct.size() / 3) - 1));
        o.push_back(flags);
        if (!f.lct.empty()) put_table(o, f.lct);
        std::vector<uint8_t> stream;
        if (f.interlace) {
            const int start[4] = {0, 4, 2, 1}, step[4] = {8, 8, 4, 2};
            for (int pass = 0; pass < 4; ++pass)
                for (int r = start[pass]; r < f.h; r += step[pass])
                    stream.insert(stream.end(), f.idx.begin() + std::ptrdiff_t(r * f.w),
                                  f.idx.begin() + std::ptrdiff_t((r + 1) * f.w));
        } else {
            stream = f.idx;
        }
        const std::size_t colours = (f.lct.empty() ? gct.size() : f.lct.size()) / 3;
        put_lzw(o, stream, std::max(2, table_bits(colours)));
    }
    o.push_back(0x3B);
    return o;
}

} // namespace gifw
