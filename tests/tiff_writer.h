#pragma once
// A small TIFF writer for the tests: every layout the decoder reads (strips
// or tiles, planar 1 or 2, 1..16-bit samples, predictor 2) and its own LZW,
// PackBits and Deflate encoders, so each fixture is built from known sample
// values and the decode can be checked exactly. The IFD goes after the
// pixel data, as libtiff writes it.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

extern "C" unsigned char* stbi_zlib_compress(unsigned char* data, int data_len, int* out_len, int quality);

namespace tiffw {

struct Buf {
    std::vector<uint8_t> b;
    bool le = true;
    void u8(uint32_t v) { b.push_back(uint8_t(v)); }
    void u16(uint32_t v) {
        if (le) { u8(v); u8(v >> 8); } else { u8(v >> 8); u8(v); }
    }
    void u32(uint32_t v) {
        if (le) { u16(v & 0xFFFF); u16(v >> 16); } else { u16(v >> 16); u16(v & 0xFFFF); }
    }
    void put32_at(std::size_t at, uint32_t v) {
        Buf t;
        t.le = le;
        t.u32(v);
        std::memcpy(b.data() + at, t.b.data(), 4);
    }
    void bytes(const std::vector<uint8_t>& v) { b.insert(b.end(), v.begin(), v.end()); }
    void align() { if (b.size() & 1) u8(0); }
};

struct Entry {
    uint16_t tag, type;
    std::vector<uint32_t> vals;  // SHORT / LONG / RATIONAL (num, den pairs)
    std::vector<uint8_t> raw;    // BYTE / ASCII / UNDEFINED
    uint32_t count() const {
        if (type == 5 || type == 10) return uint32_t(vals.size() / 2);
        return raw.empty() ? uint32_t(vals.size()) : uint32_t(raw.size());
    }
};

inline Entry shorts(uint16_t tag, std::vector<uint32_t> v) { return {tag, 3, std::move(v), {}}; }
inline Entry longs(uint16_t tag, std::vector<uint32_t> v) { return {tag, 4, std::move(v), {}}; }
inline Entry rationals(uint16_t tag, std::vector<uint32_t> num_den) { return {tag, 5, std::move(num_den), {}}; }
inline Entry ascii(uint16_t tag, const char* s) {
    Entry e{tag, 2, {}, {}};
    e.raw.assign(s, s + std::strlen(s) + 1);
    return e;
}
inline Entry undefined(uint16_t tag, std::vector<uint8_t> v) { return {tag, 7, {}, std::move(v)}; }

inline uint32_t type_size(uint16_t t) { return t == 3 ? 2 : (t == 4 ? 4 : (t == 5 || t == 10) ? 8 : 1); }

// Writes the out-of-line values, then the directory; returns its offset.
// `next_at` receives where the next-IFD link is, to patch.
inline uint32_t write_ifd(Buf& buf, std::vector<Entry> entries, std::size_t* next_at = nullptr) {
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.tag < b.tag; });
    std::vector<uint32_t> where(entries.size(), 0);
    auto put_values = [&](Buf& o, const Entry& e) {
        if (!e.raw.empty()) { for (uint8_t c : e.raw) o.u8(c); return; }
        for (uint32_t v : e.vals) {
            if (e.type == 3) o.u16(v); else o.u32(v);
        }
    };
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (uint64_t(entries[i].count()) * type_size(entries[i].type) <= 4) continue;
        buf.align();
        where[i] = uint32_t(buf.b.size());
        put_values(buf, entries[i]);
    }
    buf.align();
    const uint32_t ifd = uint32_t(buf.b.size());
    buf.u16(uint32_t(entries.size()));
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const Entry& e = entries[i];
        buf.u16(e.tag);
        buf.u16(e.type);
        buf.u32(e.count());
        if (uint64_t(e.count()) * type_size(e.type) <= 4) {
            Buf v;
            v.le = buf.le;
            put_values(v, e);
            while (v.b.size() < 4) v.u8(0);
            buf.bytes(v.b);
        } else {
            buf.u32(where[i]);
        }
    }
    if (next_at) *next_at = buf.b.size();
    buf.u32(0);
    return ifd;
}

// ---- compressors -------------------------------------------------------------

struct BitWriter {
    std::vector<uint8_t> out;
    uint32_t acc = 0, n = 0;
    void put(uint32_t code, uint32_t width) {
        acc = (acc << width) | code;
        n += width;
        while (n >= 8) { out.push_back(uint8_t(acc >> (n - 8))); n -= 8; }
        acc &= (1u << n) - 1;
    }
    void flush() { if (n) out.push_back(uint8_t(acc << (8 - n))); n = 0; acc = 0; }
};

// TIFF LZW the way libtiff writes it: early change, Clear when the table fills.
inline std::vector<uint8_t> lzw(const std::vector<uint8_t>& in) {
    BitWriter w;
    uint32_t nbits = 9, maxcode = 511, free_ent = 258;
    std::unordered_map<uint32_t, uint32_t> dict;
    w.put(256, nbits);
    if (in.empty()) { w.put(257, nbits); w.flush(); return w.out; }
    uint32_t ent = in[0];
    auto added = [&]() {
        ++free_ent;
        if (free_ent == 4094) {
            w.put(256, nbits);
            dict.clear();
            free_ent = 258; nbits = 9; maxcode = 511;
        } else if (free_ent > maxcode) {
            ++nbits;
            maxcode = (1u << nbits) - 1;
        }
    };
    for (std::size_t i = 1; i < in.size(); ++i) {
        const uint32_t key = ent << 8 | in[i];
        auto it = dict.find(key);
        if (it != dict.end()) { ent = it->second; continue; }
        w.put(ent, nbits);
        dict[key] = free_ent;
        added();
        ent = in[i];
    }
    w.put(ent, nbits);
    added();
    w.put(257, nbits);
    w.flush();
    return w.out;
}

inline std::vector<uint8_t> packbits(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> out;
    std::size_t i = 0;
    while (i < in.size()) {
        std::size_t run = 1;
        while (i + run < in.size() && in[i + run] == in[i] && run < 128) ++run;
        if (run >= 3) {
            out.push_back(uint8_t(int8_t(1 - int(run))));
            out.push_back(in[i]);
            i += run;
            continue;
        }
        std::size_t lit = 0;
        const std::size_t start = i;
        while (i < in.size() && lit < 128) {
            if (i + 2 < in.size() && in[i] == in[i + 1] && in[i] == in[i + 2]) break;
            ++i;
            ++lit;
        }
        out.push_back(uint8_t(lit - 1));
        out.insert(out.end(), in.begin() + long(start), in.begin() + long(start + lit));
    }
    out.push_back(0x80);  // a no-op the decoder must skip (it stops at the chunk's size first)
    return out;
}

inline std::vector<uint8_t> deflate(const std::vector<uint8_t>& in) {
    int len = 0;
    unsigned char* z = stbi_zlib_compress(const_cast<unsigned char*>(in.data()), int(in.size()), &len, 6);
    std::vector<uint8_t> out(z, z + len);
    std::free(z);
    return out;
}

// ---- the image -------------------------------------------------------------

struct Spec {
    bool le = true;
    uint32_t w = 1, h = 1, bits = 8, spp = 1, photometric = 1, compression = 1;
    uint32_t predictor = 1, planar = 1, rows_per_strip = 0;  // 0: one strip
    uint32_t tile_w = 0, tile_h = 0;                         // non-zero: tiled
    bool fill_order2 = false;
    std::vector<uint32_t> extra_samples;
    std::vector<uint32_t> colormap;
    std::vector<Entry> more;  // extra IFD0 entries (Orientation, Make, ...)
    bool second_page = false;
};

inline std::vector<uint8_t> compress(const Spec& s, const std::vector<uint8_t>& raw) {
    switch (s.compression) {
        case 5: return lzw(raw);
        case 32773: return packbits(raw);
        case 8: case 32946: return deflate(raw);
        default: return raw;
    }
}

// `samples`: s.w * s.h * s.spp values, pixel-interleaved.
inline std::vector<uint8_t> make(const Spec& s, const std::vector<uint32_t>& samples) {
    Buf buf;
    buf.le = s.le;
    buf.b = s.le ? std::vector<uint8_t>{'I', 'I'} : std::vector<uint8_t>{'M', 'M'};
    buf.u16(42);
    buf.u32(0);
    const bool tiled = s.tile_w > 0;
    const uint32_t cw = tiled ? s.tile_w : s.w;
    const uint32_t ch = tiled ? s.tile_h : (s.rows_per_strip ? std::min(s.rows_per_strip, s.h) : s.h);
    const uint32_t across = (s.w + cw - 1) / cw, down = (s.h + ch - 1) / ch;
    const uint32_t planes = s.planar == 2 ? s.spp : 1, cs = s.planar == 2 ? 1 : s.spp;
    const uint32_t mask = (1u << s.bits) - 1;
    std::vector<uint32_t> offs, counts;
    for (uint32_t p = 0; p < planes; ++p)
        for (uint32_t cy = 0; cy < down; ++cy)
            for (uint32_t cx = 0; cx < across; ++cx) {
                const uint32_t rows = tiled ? ch : std::min(ch, s.h - cy * ch);
                std::vector<uint8_t> raw;
                for (uint32_t y = 0; y < rows; ++y) {
                    std::vector<uint32_t> row(std::size_t(cw) * cs, 0);
                    for (uint32_t x = 0; x < cw; ++x)
                        for (uint32_t c = 0; c < cs; ++c) {
                            const uint32_t ix = cx * cw + x, iy = cy * ch + y;
                            if (ix < s.w && iy < s.h)
                                row[std::size_t(x) * cs + c] =
                                    samples[(std::size_t(iy) * s.w + ix) * s.spp + (s.planar == 2 ? p : c)];
                        }
                    if (s.predictor == 2)
                        for (std::size_t i = row.size(); i-- > cs;) row[i] = (row[i] - row[i - cs]) & mask;
                    if (s.bits == 8) {
                        for (uint32_t v : row) raw.push_back(uint8_t(v));
                    } else if (s.bits == 16) {
                        for (uint32_t v : row) {
                            if (s.le) { raw.push_back(uint8_t(v)); raw.push_back(uint8_t(v >> 8)); }
                            else { raw.push_back(uint8_t(v >> 8)); raw.push_back(uint8_t(v)); }
                        }
                    } else {
                        BitWriter bw;
                        for (uint32_t v : row) bw.put(v, s.bits);
                        bw.flush();
                        raw.insert(raw.end(), bw.out.begin(), bw.out.end());
                    }
                }
                if (s.fill_order2)
                    for (uint8_t& b : raw) {
                        uint8_t r = 0;
                        for (int i = 0; i < 8; ++i) if (b & (1 << i)) r |= uint8_t(0x80 >> i);
                        b = r;
                    }
                const std::vector<uint8_t> z = compress(s, raw);
                buf.align();
                offs.push_back(uint32_t(buf.b.size()));
                counts.push_back(uint32_t(z.size()));
                buf.bytes(z);
            }
    std::vector<Entry> e = {
        longs(256, {s.w}), longs(257, {s.h}), shorts(258, std::vector<uint32_t>(s.spp, s.bits)),
        shorts(259, {s.compression}), shorts(262, {s.photometric}), shorts(277, {s.spp}),
        shorts(284, {s.planar}),
    };
    if (tiled) {
        e.push_back(longs(322, {s.tile_w}));
        e.push_back(longs(323, {s.tile_h}));
        e.push_back(longs(324, offs));
        e.push_back(longs(325, counts));
    } else {
        e.push_back(longs(273, offs));
        e.push_back(longs(278, {ch}));
        e.push_back(longs(279, counts));
    }
    if (s.predictor != 1) e.push_back(shorts(317, {s.predictor}));
    if (s.fill_order2) e.push_back(shorts(266, {2}));
    if (!s.extra_samples.empty()) e.push_back(shorts(338, s.extra_samples));
    if (!s.colormap.empty()) e.push_back(shorts(320, s.colormap));
    for (const Entry& m : s.more) e.push_back(m);
    std::size_t next_at = 0;
    const uint32_t ifd = write_ifd(buf, e, &next_at);
    buf.put32_at(4, ifd);
    if (s.second_page) {
        // A 2x1 gray page that is not the one decoded.
        buf.align();
        const uint32_t px = uint32_t(buf.b.size());
        buf.u8(7); buf.u8(9);
        const uint32_t ifd2 = write_ifd(buf, {longs(256, {2}), longs(257, {1}), shorts(258, {8}), shorts(262, {1}),
                                              longs(273, {px}), longs(279, {2})});
        buf.put32_at(next_at, ifd2);
    }
    return buf.b;
}

// A TIFF whose chunks are given whole (JPEG-compressed strips or tiles).
inline std::vector<uint8_t> make_raw_chunks(const Spec& s, const std::vector<std::vector<uint8_t>>& chunks) {
    Buf buf;
    buf.le = s.le;
    buf.b = s.le ? std::vector<uint8_t>{'I', 'I'} : std::vector<uint8_t>{'M', 'M'};
    buf.u16(42);
    buf.u32(0);
    std::vector<uint32_t> offs, counts;
    for (const auto& c : chunks) {
        buf.align();
        offs.push_back(uint32_t(buf.b.size()));
        counts.push_back(uint32_t(c.size()));
        buf.bytes(c);
    }
    std::vector<Entry> e = {
        longs(256, {s.w}), longs(257, {s.h}), shorts(258, std::vector<uint32_t>(s.spp, s.bits)),
        shorts(259, {s.compression}), shorts(262, {s.photometric}), shorts(277, {s.spp}),
    };
    if (s.tile_w) {
        e.push_back(longs(322, {s.tile_w}));
        e.push_back(longs(323, {s.tile_h}));
        e.push_back(longs(324, offs));
        e.push_back(longs(325, counts));
    } else {
        e.push_back(longs(273, offs));
        e.push_back(longs(278, {s.rows_per_strip ? s.rows_per_strip : s.h}));
        e.push_back(longs(279, counts));
    }
    for (const Entry& m : s.more) e.push_back(m);
    const uint32_t ifd = write_ifd(buf, e);
    buf.put32_at(4, ifd);
    return buf.b;
}

} // namespace tiffw
