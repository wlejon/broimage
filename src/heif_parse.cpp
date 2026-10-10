// The HEIF container reader (heif_parse.h). Box layouts as ISO/IEC 14496-12
// (ISOBMFF) and ISO/IEC 23008-12 (HEIF) define them; every field is read
// through a bounds-checked reader, so a truncated or hostile file fails the
// parse instead of reading past the data.

#include "heif_parse.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace broimage::detail {

namespace {

struct Reader {
    const uint8_t* p = nullptr;
    const uint8_t* end = nullptr;
    bool ok = true;

    Reader(const uint8_t* b, std::size_t n) : p(b), end(b + n) {}

    std::size_t left() const { return std::size_t(end - p); }
    bool need(std::size_t n) {
        if (!ok || left() < n) {
            ok = false;
            return false;
        }
        return true;
    }
    uint64_t un(int bytes) {
        if (!need(std::size_t(bytes))) return 0;
        uint64_t v = 0;
        for (int i = 0; i < bytes; ++i) v = v << 8 | *p++;
        return v;
    }
    uint8_t u8() { return uint8_t(un(1)); }
    uint16_t u16() { return uint16_t(un(2)); }
    uint32_t u32() { return uint32_t(un(4)); }
    uint64_t u64() { return un(8); }
    void skip(std::size_t n) {
        if (need(n)) p += n;
    }
    FourCC cc() {
        FourCC f{};
        if (need(4)) {
            std::memcpy(f.data(), p, 4);
            p += 4;
        }
        return f;
    }
    // A null-terminated UTF-8 string; one running to the end is taken whole.
    std::string cstr() {
        const uint8_t* z = static_cast<const uint8_t*>(std::memchr(p, 0, left()));
        const uint8_t* stop = z ? z : end;
        std::string s(reinterpret_cast<const char*>(p), std::size_t(stop - p));
        p = z ? z + 1 : end;
        return s;
    }
    // version (8) + flags (24) of a FullBox.
    void full(int& version, uint32_t& flags) {
        const uint32_t v = u32();
        version = int(v >> 24);
        flags = v & 0xFFFFFF;
    }
};

struct Box {
    FourCC type{};
    const uint8_t* body = nullptr;
    std::size_t size = 0;
};

// The next box in `r`; false at the end, or with r.ok cleared on a box
// that runs past its parent.
bool next_box(Reader& r, Box& b) {
    if (!r.ok || r.left() < 8) return false;
    const uint8_t* start = r.p;
    uint64_t size = r.u32();
    b.type = r.cc();
    std::size_t header = 8;
    if (size == 1) {
        size = r.u64();
        header = 16;
    } else if (size == 0) {
        size = uint64_t(r.end - start);
    }
    if (fourcc_is(b.type, "uuid")) {
        r.skip(16);
        header += 16;
    }
    if (!r.ok || size < header || size > uint64_t(r.end - start)) {
        r.ok = false;
        return false;
    }
    b.body = start + header;
    b.size = std::size_t(size) - header;
    r.p = start + size;
    return true;
}

bool fail(std::string* why, const char* msg) {
    if (why) *why = msg;
    return false;
}

constexpr const char* kImageBrands[] = {"mif1", "mif2", "miaf", "msf1", "avif", "avis", "avio",
                                        "heic", "heix", "heim", "heis", "hevc", "hevx", "heif"};

bool image_brand(const uint8_t* b) {
    for (const char* k : kImageBrands)
        if (std::memcmp(b, k, 4) == 0) return true;
    return false;
}

bool parse_ftyp(const Box& b, HeifFile& out) {
    Reader r(b.body, b.size);
    out.major_brand = r.cc();
    r.u32();  // minor_version
    while (r.ok && r.left() >= 4) out.compatible_brands.push_back(r.cc());
    return r.ok;
}

bool parse_iloc(const Box& b, HeifFile& out) {
    Reader r(b.body, b.size);
    int version = 0;
    uint32_t flags = 0;
    r.full(version, flags);
    if (version > 2) return false;
    const uint8_t s1 = r.u8(), s2 = r.u8();
    const int offset_size = s1 >> 4, length_size = s1 & 15, base_offset_size = s2 >> 4;
    const int index_size = (version == 1 || version == 2) ? (s2 & 15) : 0;
    auto valid = [](int s) { return s == 0 || s == 4 || s == 8; };
    if (!valid(offset_size) || !valid(length_size) || !valid(base_offset_size) || !valid(index_size)) return false;
    const uint32_t count = version < 2 ? r.u16() : r.u32();
    for (uint32_t i = 0; i < count && r.ok; ++i) {
        const uint32_t id = version < 2 ? r.u16() : r.u32();
        HeifItem* it = nullptr;
        for (auto& x : out.items)
            if (x.id == id) it = &x;
        HeifItem scratch;
        if (!it) it = &scratch;  // located but never declared in iinf: read past it
        it->located = true;
        it->construction_method = (version == 1 || version == 2) ? (r.u16() & 15) : 0;
        it->data_reference_index = r.u16();
        it->base_offset = r.un(base_offset_size);
        const uint16_t extents = r.u16();
        it->extents.clear();
        for (uint16_t e = 0; e < extents && r.ok; ++e) {
            if (index_size) r.un(index_size);
            HeifExtent x;
            x.offset = r.un(offset_size);
            x.length = r.un(length_size);
            it->extents.push_back(x);
        }
    }
    return r.ok;
}

bool parse_infe(const Box& b, HeifItem& it) {
    Reader r(b.body, b.size);
    int version = 0;
    uint32_t flags = 0;
    r.full(version, flags);
    it.hidden = (flags & 1) != 0;
    if (version >= 2) {
        it.id = version == 2 ? r.u16() : r.u32();
        r.u16();  // item_protection_index
        it.type = r.cc();
        if (!r.ok) return false;
        it.name = r.cstr();
        if (fourcc_is(it.type, "mime") && r.left()) it.content_type = r.cstr();
    } else {
        it.id = r.u16();
        r.u16();
        if (!r.ok) return false;
        std::memcpy(it.type.data(), "mime", 4);
        it.name = r.cstr();
        if (r.left()) it.content_type = r.cstr();
    }
    return r.ok;
}

bool parse_iinf(const Box& b, HeifFile& out) {
    Reader r(b.body, b.size);
    int version = 0;
    uint32_t flags = 0;
    r.full(version, flags);
    const uint32_t count = version == 0 ? r.u16() : r.u32();
    Box c;
    uint32_t seen = 0;
    while (seen < count && next_box(r, c)) {
        if (!fourcc_is(c.type, "infe")) continue;
        HeifItem it;
        if (!parse_infe(c, it)) return false;
        // A second infe for an id replaces the first (they should not repeat).
        bool replaced = false;
        for (auto& x : out.items)
            if (x.id == it.id) {
                x = std::move(it);
                replaced = true;
                break;
            }
        if (!replaced) out.items.push_back(std::move(it));
        ++seen;
    }
    return r.ok;
}

bool parse_iref(const Box& b, HeifFile& out) {
    Reader r(b.body, b.size);
    int version = 0;
    uint32_t flags = 0;
    r.full(version, flags);
    const int id_bytes = version == 0 ? 2 : 4;
    Box c;
    while (next_box(r, c)) {
        Reader rr(c.body, c.size);
        HeifRef ref;
        ref.type = c.type;
        ref.from = uint32_t(rr.un(id_bytes));
        const uint16_t n = rr.u16();
        for (uint16_t i = 0; i < n && rr.ok; ++i) ref.to.push_back(uint32_t(rr.un(id_bytes)));
        if (!rr.ok) return false;
        out.refs.push_back(std::move(ref));
    }
    return r.ok;
}

bool parse_ipma(const Box& b, HeifFile& out) {
    Reader r(b.body, b.size);
    int version = 0;
    uint32_t flags = 0;
    r.full(version, flags);
    const uint32_t count = r.u32();
    for (uint32_t i = 0; i < count && r.ok; ++i) {
        const uint32_t id = version < 1 ? r.u16() : r.u32();
        const uint8_t n = r.u8();
        std::vector<HeifPropRef> assoc;
        for (uint8_t a = 0; a < n && r.ok; ++a) {
            HeifPropRef pr;
            if (flags & 1) {
                const uint16_t v = r.u16();
                pr.essential = (v & 0x8000) != 0;
                pr.index = v & 0x7FFF;
            } else {
                const uint8_t v = r.u8();
                pr.essential = (v & 0x80) != 0;
                pr.index = v & 0x7F;
            }
            if (pr.index) assoc.push_back(pr);
        }
        for (auto& it : out.items)
            if (it.id == id) it.props.insert(it.props.end(), assoc.begin(), assoc.end());
    }
    return r.ok;
}

bool parse_iprp(const Box& b, HeifFile& out) {
    Reader r(b.body, b.size);
    Box c;
    std::vector<Box> ipmas;
    while (next_box(r, c)) {
        if (fourcc_is(c.type, "ipco")) {
            Reader pr(c.body, c.size);
            Box p;
            while (next_box(pr, p)) out.props.push_back(HeifProp{p.type, p.body, p.size});
            if (!pr.ok) return false;
        } else if (fourcc_is(c.type, "ipma")) {
            ipmas.push_back(c);
        }
    }
    if (!r.ok) return false;
    for (const Box& m : ipmas)
        if (!parse_ipma(m, out)) return false;
    return true;
}

bool parse_meta(const Box& b, HeifFile& out, std::string* why) {
    Reader r(b.body, b.size);
    int version = 0;
    uint32_t flags = 0;
    r.full(version, flags);
    Box c;
    bool pict = false;
    // iinf before iloc and iprp: those attach to declared items.
    std::vector<Box> later;
    while (next_box(r, c)) {
        if (fourcc_is(c.type, "hdlr")) {
            Reader h(c.body, c.size);
            h.u32();  // version / flags
            h.u32();  // pre_defined
            pict = fourcc_is(h.cc(), "pict") && h.ok;
        } else if (fourcc_is(c.type, "pitm")) {
            Reader p(c.body, c.size);
            int v = 0;
            uint32_t f = 0;
            p.full(v, f);
            out.primary = v == 0 ? p.u16() : p.u32();
            if (!p.ok) return fail(why, "HEIF: a malformed pitm box");
        } else if (fourcc_is(c.type, "iinf")) {
            if (!parse_iinf(c, out)) return fail(why, "HEIF: a malformed iinf box");
        } else if (fourcc_is(c.type, "idat")) {
            out.idat = c.body;
            out.idat_size = c.size;
        } else {
            later.push_back(c);
        }
    }
    if (!r.ok) return fail(why, "HEIF: a box runs past the meta box");
    if (!pict) return fail(why, "HEIF: the meta box has no 'pict' handler");
    for (const Box& l : later) {
        if (fourcc_is(l.type, "iloc") && !parse_iloc(l, out)) return fail(why, "HEIF: a malformed iloc box");
        if (fourcc_is(l.type, "iref") && !parse_iref(l, out)) return fail(why, "HEIF: a malformed iref box");
        if (fourcc_is(l.type, "iprp") && !parse_iprp(l, out)) return fail(why, "HEIF: a malformed iprp box");
    }
    out.has_meta = true;
    return true;
}

// ---- moov: the first picture track of an image sequence ----------------------

bool find_child(const uint8_t* body, std::size_t size, const char* type, Box& out) {
    Reader r(body, size);
    Box c;
    while (next_box(r, c))
        if (fourcc_is(c.type, type)) {
            out = c;
            return true;
        }
    return false;
}

bool parse_stbl(const Box& stbl, HeifTrack& t) {
    Box b;
    if (!find_child(stbl.body, stbl.size, "stsd", b)) return false;
    {
        Reader r(b.body, b.size);
        r.u32();  // version / flags
        if (r.u32() < 1) return false;
        Box entry;
        if (!next_box(r, entry)) return false;
        t.codec = entry.type;
        // SampleEntry (8) + VisualSampleEntry fields (70) before the child boxes.
        if (entry.size < 78) return false;
        Reader e(entry.body, entry.size);
        e.skip(24);
        t.width = e.u16();
        t.height = e.u16();
        Box cfg;
        Reader kids(entry.body + 78, entry.size - 78);
        while (next_box(kids, cfg)) {
            if (fourcc_is(cfg.type, "av1C") || fourcc_is(cfg.type, "hvcC")) {
                t.config = cfg.body;
                t.config_size = cfg.size;
                t.config_type = cfg.type;
                break;
            }
        }
    }
    if (find_child(stbl.body, stbl.size, "stsz", b)) {
        Reader r(b.body, b.size);
        r.u32();
        const uint32_t fixed = r.u32();
        t.sample_count = r.u32();
        t.first_sample_size = fixed ? fixed : (t.sample_count ? r.u32() : 0);
        if (!r.ok) return false;
    } else if (find_child(stbl.body, stbl.size, "stz2", b)) {
        Reader r(b.body, b.size);
        r.u32();
        r.skip(3);
        const uint8_t field = r.u8();
        t.sample_count = r.u32();
        if (field == 4) t.first_sample_size = r.u8() >> 4;
        else if (field == 8 || field == 16) t.first_sample_size = r.un(field / 8);
        else return false;
        if (!r.ok) return false;
    } else {
        return false;
    }
    // The first sample is the first in the first chunk.
    if (find_child(stbl.body, stbl.size, "stco", b)) {
        Reader r(b.body, b.size);
        r.u32();
        if (r.u32() < 1) return false;
        t.first_sample_offset = r.u32();
        if (!r.ok) return false;
    } else if (find_child(stbl.body, stbl.size, "co64", b)) {
        Reader r(b.body, b.size);
        r.u32();
        if (r.u32() < 1) return false;
        t.first_sample_offset = r.u64();
        if (!r.ok) return false;
    } else {
        return false;
    }
    return t.sample_count > 0 && t.first_sample_size > 0;
}

void parse_moov(const Box& moov, HeifFile& out) {
    Reader r(moov.body, moov.size);
    Box trak;
    while (next_box(r, trak)) {
        if (!fourcc_is(trak.type, "trak")) continue;
        Box mdia, hdlr, minf, stbl;
        if (!find_child(trak.body, trak.size, "mdia", mdia)) continue;
        if (!find_child(mdia.body, mdia.size, "hdlr", hdlr)) continue;
        Reader h(hdlr.body, hdlr.size);
        h.u32();
        h.u32();
        const FourCC handler = h.cc();
        if (!h.ok || !(fourcc_is(handler, "pict") || fourcc_is(handler, "vide"))) continue;
        if (!find_child(mdia.body, mdia.size, "minf", minf)) continue;
        if (!find_child(minf.body, minf.size, "stbl", stbl)) continue;
        HeifTrack t;
        t.handler = handler;
        if (!parse_stbl(stbl, t)) continue;
        t.present = true;
        out.track = t;
        return;
    }
}

} // namespace

bool HeifFile::has_brand(const char* b) const {
    if (fourcc_is(major_brand, b)) return true;
    for (const auto& c : compatible_brands)
        if (fourcc_is(c, b)) return true;
    return false;
}

const HeifItem* HeifFile::item(uint32_t id) const {
    for (const auto& it : items)
        if (it.id == id) return &it;
    return nullptr;
}

const HeifProp* HeifFile::prop(const HeifItem& it, const char* type) const {
    for (const auto& pr : it.props) {
        if (pr.index == 0 || pr.index > props.size()) continue;
        const HeifProp& p = props[pr.index - 1];
        if (fourcc_is(p.type, type)) return &p;
    }
    return nullptr;
}

std::vector<uint32_t> HeifFile::refs_from(uint32_t from, const char* type) const {
    std::vector<uint32_t> out;
    for (const auto& r : refs)
        if (r.from == from && fourcc_is(r.type, type)) out.insert(out.end(), r.to.begin(), r.to.end());
    return out;
}

std::vector<uint32_t> HeifFile::refs_to(uint32_t to, const char* type) const {
    std::vector<uint32_t> out;
    for (const auto& r : refs)
        if (fourcc_is(r.type, type) && std::find(r.to.begin(), r.to.end(), to) != r.to.end())
            out.push_back(r.from);
    return out;
}

namespace {

// The source of an item's extents: the file (method 0) or idat (method 1).
bool item_source(const HeifFile& f, const HeifItem& it, const uint8_t*& src, std::size_t& n, std::string* why) {
    if (!it.located) return fail(why, "HEIF: the item has no location");
    if (it.data_reference_index != 0) return fail(why, "HEIF: the item's data is in another file");
    if (it.construction_method == 0) {
        src = f.data;
        n = f.size;
    } else if (it.construction_method == 1) {
        if (!f.idat) return fail(why, "HEIF: the item is in idat, and there is none");
        src = f.idat;
        n = f.idat_size;
    } else {
        return fail(why, "HEIF: item-offset construction (iloc method 2) is not supported");
    }
    return true;
}

bool extent_range(const HeifItem& it, const HeifExtent& e, std::size_t n, std::size_t& off, std::size_t& len) {
    const uint64_t start = it.base_offset + e.offset;
    if (start < it.base_offset || start > n) return false;
    const uint64_t length = e.length ? e.length : n - start;
    if (length > n - start) return false;
    off = std::size_t(start);
    len = std::size_t(length);
    return true;
}

} // namespace

bool HeifFile::item_data(const HeifItem& it, std::vector<uint8_t>& out, std::string* why) const {
    out.clear();
    const uint8_t* src = nullptr;
    std::size_t n = 0;
    if (!item_source(*this, it, src, n, why)) return false;
    if (it.extents.empty()) return fail(why, "HEIF: the item has no extents");
    for (const auto& e : it.extents) {
        std::size_t off = 0, len = 0;
        if (!extent_range(it, e, n, off, len)) return fail(why, "HEIF: an item extent lies outside the file (truncated?)");
        out.insert(out.end(), src + off, src + off + len);
    }
    return true;
}

bool HeifFile::item_span(const HeifItem& it, const uint8_t*& p, std::size_t& len) const {
    const uint8_t* src = nullptr;
    std::size_t n = 0;
    if (!item_source(*this, it, src, n, nullptr) || it.extents.size() != 1) return false;
    std::size_t off = 0;
    if (!extent_range(it, it.extents[0], n, off, len)) return false;
    p = src + off;
    return true;
}

bool heif_sniff(const uint8_t* data, std::size_t size) {
    if (!data || size < 16 || std::memcmp(data + 4, "ftyp", 4) != 0) return false;
    const uint32_t box = uint32_t(data[0]) << 24 | uint32_t(data[1]) << 16 | uint32_t(data[2]) << 8 | data[3];
    if (box < 16) return false;
    if (image_brand(data + 8)) return true;
    const std::size_t end = std::min<std::size_t>(box, size);
    for (std::size_t p = 16; p + 4 <= end; p += 4)
        if (image_brand(data + p)) return true;
    return false;
}

bool heif_parse(const uint8_t* data, std::size_t size, HeifFile& out, std::string* why) {
    out = HeifFile{};
    out.data = data;
    out.size = size;
    if (!heif_sniff(data, size)) return fail(why, "not a HEIF file");
    Reader r(data, size);
    Box b;
    bool first = true;
    Box meta{}, moov{};
    bool have_meta = false, have_moov = false;
    while (next_box(r, b)) {
        if (first) {
            if (!fourcc_is(b.type, "ftyp") || !parse_ftyp(b, out)) return fail(why, "HEIF: a malformed ftyp box");
            first = false;
        } else if (fourcc_is(b.type, "meta") && !have_meta) {
            meta = b;
            have_meta = true;
        } else if (fourcc_is(b.type, "moov") && !have_moov) {
            moov = b;
            have_moov = true;
        }
    }
    // A truncated tail (an mdat cut short) still leaves the meta box readable.
    if (have_meta && !parse_meta(meta, out, why)) return false;
    if (have_moov) parse_moov(moov, out);
    if (!out.has_meta && !out.track.present) return fail(why, "HEIF: no meta box and no image track (truncated?)");
    return true;
}

// ---- Properties ---------------------------------------------------------------

bool read_ispe(const HeifProp& p, HeifIspe& out) {
    Reader r(p.body, p.size);
    r.u32();
    out.width = r.u32();
    out.height = r.u32();
    return r.ok && out.width && out.height;
}

bool read_nclx(const HeifProp& p, HeifNclx& out) {
    Reader r(p.body, p.size);
    const FourCC t = r.cc();
    if (!r.ok || !fourcc_is(t, "nclx")) return false;
    out.primaries = r.u16();
    out.transfer = r.u16();
    out.matrix = r.u16();
    out.full_range = (r.u8() & 0x80) != 0;
    out.present = r.ok;
    return r.ok;
}

bool read_icc(const HeifProp& p, const uint8_t*& icc, std::size_t& n) {
    Reader r(p.body, p.size);
    const FourCC t = r.cc();
    if (!r.ok || !(fourcc_is(t, "prof") || fourcc_is(t, "rICC"))) return false;
    icc = r.p;
    n = r.left();
    return n > 0;
}

bool read_irot(const HeifProp& p, int& angle) {
    if (p.size < 1) return false;
    angle = p.body[0] & 3;
    return true;
}

bool read_imir(const HeifProp& p, int& mode) {
    if (p.size < 1) return false;
    mode = p.body[0] & 1;
    return true;
}

bool read_clap(const HeifProp& p, HeifClap& c) {
    Reader r(p.body, p.size);
    c.width_n = r.u32();
    c.width_d = r.u32();
    c.height_n = r.u32();
    c.height_d = r.u32();
    c.horiz_off_n = int32_t(r.u32());
    c.horiz_off_d = r.u32();
    c.vert_off_n = int32_t(r.u32());
    c.vert_off_d = r.u32();
    return r.ok && c.width_d && c.height_d && c.horiz_off_d && c.vert_off_d;
}

bool read_auxc(const HeifProp& p, std::string& aux_type) {
    Reader r(p.body, p.size);
    r.u32();
    if (!r.ok) return false;
    aux_type = r.cstr();
    return true;
}

bool read_pixi(const HeifProp& p, std::vector<uint8_t>& bits) {
    Reader r(p.body, p.size);
    r.u32();
    const uint8_t n = r.u8();
    bits.clear();
    for (uint8_t i = 0; i < n && r.ok; ++i) bits.push_back(r.u8());
    return r.ok;
}

bool read_clli(const HeifProp& p, uint16_t& max_cll, uint16_t& max_fall) {
    Reader r(p.body, p.size);
    max_cll = r.u16();
    max_fall = r.u16();
    return r.ok;
}

bool clap_rect(const HeifClap& c, int w, int h, int& x, int& y, int& cw, int& ch) {
    // ISO/IEC 14496-12 12.1.4: the aperture is centred at
    // ((w - 1) / 2 + horizOff, (h - 1) / 2 + vertOff); its left edge is
    // that centre minus (cleanWidth - 1) / 2. Whole pixels only: MIAF
    // requires the result to land on them.
    const double fw = double(c.width_n) / c.width_d;
    const double fh = double(c.height_n) / c.height_d;
    const double cx = (w - 1) / 2.0 + double(c.horiz_off_n) / c.horiz_off_d;
    const double cy = (h - 1) / 2.0 + double(c.vert_off_n) / c.vert_off_d;
    const double left = cx - (fw - 1) / 2.0;
    const double top = cy - (fh - 1) / 2.0;
    const double iw = std::floor(fw), ih = std::floor(fh);
    const double ix = std::round(left), iy = std::round(top);
    if (iw < 1 || ih < 1 || ix < 0 || iy < 0 || ix + iw > w || iy + ih > h) return false;
    x = int(ix);
    y = int(iy);
    cw = int(iw);
    ch = int(ih);
    return true;
}

}  // namespace broimage::detail
