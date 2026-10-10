// A small HEIF writer for tests: items, their properties, references and
// where their bytes live (the file or idat, in one extent or two), written
// from ISO/IEC 23008-12 independently of the reader under test.
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace heifw {

using Bytes = std::vector<uint8_t>;

inline void u8(Bytes& v, uint32_t x) { v.push_back(uint8_t(x)); }
inline void u16(Bytes& v, uint32_t x) {
    v.push_back(uint8_t(x >> 8));
    v.push_back(uint8_t(x));
}
inline void u32(Bytes& v, uint32_t x) {
    u16(v, x >> 16);
    u16(v, x & 0xFFFF);
}
inline void cc(Bytes& v, const char* s) { v.insert(v.end(), s, s + 4); }
inline void str(Bytes& v, const std::string& s) {
    v.insert(v.end(), s.begin(), s.end());
    v.push_back(0);
}

inline Bytes box(const char* type, const Bytes& body) {
    Bytes b;
    u32(b, uint32_t(body.size() + 8));
    cc(b, type);
    b.insert(b.end(), body.begin(), body.end());
    return b;
}

inline Bytes fullbox(const char* type, int version, uint32_t flags, const Bytes& body) {
    Bytes b;
    u32(b, uint32_t(version) << 24 | flags);
    b.insert(b.end(), body.begin(), body.end());
    return box(type, b);
}

// ---- Properties ----------------------------------------------------------------

inline Bytes ispe(uint32_t w, uint32_t h) {
    Bytes b;
    u32(b, w);
    u32(b, h);
    return fullbox("ispe", 0, 0, b);
}
inline Bytes nclx(int cp, int tc, int mc, bool full) {
    Bytes b;
    cc(b, "nclx");
    u16(b, uint32_t(cp));
    u16(b, uint32_t(tc));
    u16(b, uint32_t(mc));
    u8(b, full ? 0x80 : 0);
    return box("colr", b);
}
inline Bytes irot(int angle) { return box("irot", Bytes{uint8_t(angle & 3)}); }
inline Bytes imir(int axis) { return box("imir", Bytes{uint8_t(axis & 1)}); }
inline Bytes clap(uint32_t w, uint32_t h, int32_t hoff, int32_t voff) {
    Bytes b;
    u32(b, w);
    u32(b, 1);
    u32(b, h);
    u32(b, 1);
    u32(b, uint32_t(hoff));
    u32(b, 1);
    u32(b, uint32_t(voff));
    u32(b, 1);
    return box("clap", b);
}
inline Bytes auxc(const std::string& urn) {
    Bytes b;
    str(b, urn);
    return fullbox("auxC", 0, 0, b);
}
inline Bytes pixi(std::vector<uint8_t> bits) {
    Bytes b;
    u8(b, uint32_t(bits.size()));
    b.insert(b.end(), bits.begin(), bits.end());
    return fullbox("pixi", 0, 0, b);
}
inline Bytes av1c() { return box("av1C", Bytes{0x81, 0x00, 0x0C, 0x00}); }
inline Bytes clli(uint16_t max_cll) {
    Bytes b;
    u16(b, max_cll);
    u16(b, max_cll / 4);
    return box("clli", b);
}

// ---- Items ---------------------------------------------------------------------

struct Item {
    uint32_t id = 0;
    std::string type = "av01";
    Bytes data;
    std::vector<Bytes> props;
    bool in_idat = false;
    bool split = false;  // two extents
    bool hidden = false;
};

struct Ref {
    std::string type;
    uint32_t from = 0;
    std::vector<uint32_t> to;
};

struct File {
    std::string major = "avif";
    std::vector<std::string> compat = {"mif1", "miaf"};
    uint32_t primary = 1;
    std::vector<Item> items;
    std::vector<Ref> refs;
    Bytes tail;  // a box after mdat (e.g. a moov), as given
};

inline Bytes grid_data(int rows, int cols, uint32_t w, uint32_t h) {
    Bytes b{0, 0, uint8_t(rows - 1), uint8_t(cols - 1)};
    u16(b, w);
    u16(b, h);
    return b;
}

// Exif item payload: offset 6 then "Exif\0\0" and a big-endian TIFF with
// Make and Orientation.
inline Bytes exif_payload(const char* make, int orientation) {
    Bytes t{'M', 'M', 0, 42, 0, 0, 0, 8};
    const std::string m(make);
    u16(t, 2);
    u16(t, 0x010F);
    u16(t, 2);
    u32(t, uint32_t(m.size() + 1));
    const std::size_t make_off_at = t.size();
    u32(t, 0);
    u16(t, 0x0112);
    u16(t, 3);
    u32(t, 1);
    u16(t, uint32_t(orientation));
    u16(t, 0);
    u32(t, 0);
    const uint32_t make_off = uint32_t(t.size());
    t[make_off_at] = uint8_t(make_off >> 24);
    t[make_off_at + 1] = uint8_t(make_off >> 16);
    t[make_off_at + 2] = uint8_t(make_off >> 8);
    t[make_off_at + 3] = uint8_t(make_off);
    str(t, m);
    Bytes p;
    u32(p, 6);
    p.insert(p.end(), {'E', 'x', 'i', 'f', 0, 0});
    p.insert(p.end(), t.begin(), t.end());
    return p;
}

inline Bytes write(const File& f) {
    Bytes ftyp;
    cc(ftyp, f.major.c_str());
    u32(ftyp, 0);
    cc(ftyp, f.major.c_str());
    for (const auto& c : f.compat) cc(ftyp, c.c_str());
    Bytes out = box("ftyp", ftyp);

    Bytes hdlr;
    u32(hdlr, 0);
    cc(hdlr, "pict");
    u32(hdlr, 0);
    u32(hdlr, 0);
    u32(hdlr, 0);
    str(hdlr, "");
    Bytes pitm;
    u16(pitm, f.primary);
    Bytes iinf;
    u16(iinf, uint32_t(f.items.size()));
    for (const auto& it : f.items) {
        Bytes e;
        u16(e, it.id);
        u16(e, 0);
        cc(e, it.type.c_str());
        str(e, "");
        const Bytes infe = fullbox("infe", 2, it.hidden ? 1 : 0, e);
        iinf.insert(iinf.end(), infe.begin(), infe.end());
    }
    Bytes iref;
    for (const auto& r : f.refs) {
        Bytes s;
        u16(s, r.from);
        u16(s, uint32_t(r.to.size()));
        for (uint32_t t : r.to) u16(s, t);
        const Bytes b = box(r.type.c_str(), s);
        iref.insert(iref.end(), b.begin(), b.end());
    }
    Bytes ipco, ipma;
    u32(ipma, uint32_t(f.items.size()));
    int index = 0;
    for (const auto& it : f.items) {
        u16(ipma, it.id);
        u8(ipma, uint32_t(it.props.size()));
        for (const auto& p : it.props) {
            ipco.insert(ipco.end(), p.begin(), p.end());
            ++index;
            const bool essential = std::memcmp(p.data() + 4, "ispe", 4) != 0 && std::memcmp(p.data() + 4, "pixi", 4) != 0;
            u8(ipma, uint32_t((essential ? 0x80 : 0) | index));
        }
    }
    Bytes iprp = box("ipco", ipco);
    const Bytes ipma_box = fullbox("ipma", 0, 0, ipma);
    iprp.insert(iprp.end(), ipma_box.begin(), ipma_box.end());

    // idat: the items stored there, in order.
    Bytes idat;
    std::vector<uint32_t> idat_off(f.items.size(), 0);
    for (std::size_t i = 0; i < f.items.size(); ++i)
        if (f.items[i].in_idat) {
            idat_off[i] = uint32_t(idat.size());
            idat.insert(idat.end(), f.items[i].data.begin(), f.items[i].data.end());
        }

    // iloc v1: offsets and lengths of 4 bytes, no base offset; file offsets
    // patched once the meta box's size is known.
    auto make_iloc = [&](uint32_t mdat_data_start, std::vector<uint32_t> file_off) {
        Bytes b;
        u8(b, 0x44);
        u8(b, 0x00);
        u16(b, uint32_t(f.items.size()));
        for (std::size_t i = 0; i < f.items.size(); ++i) {
            const Item& it = f.items[i];
            u16(b, it.id);
            u16(b, it.in_idat ? 1 : 0);
            u16(b, 0);
            const uint32_t base = it.in_idat ? idat_off[i] : mdat_data_start + file_off[i];
            const uint32_t n = uint32_t(it.data.size());
            if (it.split && n >= 2) {
                u16(b, 2);
                u32(b, base);
                u32(b, n / 2);
                u32(b, base + n / 2);
                u32(b, n - n / 2);
            } else {
                u16(b, 1);
                u32(b, base);
                u32(b, n);
            }
        }
        return fullbox("iloc", 1, 0, b);
    };
    Bytes mdat_body;
    std::vector<uint32_t> file_off(f.items.size(), 0);
    for (std::size_t i = 0; i < f.items.size(); ++i)
        if (!f.items[i].in_idat) {
            file_off[i] = uint32_t(mdat_body.size());
            mdat_body.insert(mdat_body.end(), f.items[i].data.begin(), f.items[i].data.end());
        }
    auto make_meta = [&](uint32_t mdat_data_start) {
        Bytes m;
        auto add = [&](const Bytes& b) { m.insert(m.end(), b.begin(), b.end()); };
        add(fullbox("hdlr", 0, 0, hdlr));
        add(fullbox("pitm", 0, 0, pitm));
        add(make_iloc(mdat_data_start, file_off));
        add(fullbox("iinf", 0, 0, iinf));
        if (!f.refs.empty()) add(fullbox("iref", 0, 0, iref));
        add(box("iprp", iprp));
        if (!idat.empty()) add(box("idat", idat));
        return fullbox("meta", 0, 0, m);
    };
    const Bytes probe_meta = make_meta(0);
    const uint32_t mdat_data_start = uint32_t(out.size() + probe_meta.size() + 8);
    const Bytes meta = make_meta(mdat_data_start);
    out.insert(out.end(), meta.begin(), meta.end());
    const Bytes mdat = box("mdat", mdat_body);
    out.insert(out.end(), mdat.begin(), mdat.end());
    out.insert(out.end(), f.tail.begin(), f.tail.end());
    return out;
}

}  // namespace heifw
