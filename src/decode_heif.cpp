// HEIF decoding (heif.h): the primary item, through grids, the alpha
// auxiliary image, premultiplication and the clean aperture, into straight
// RGBA8; AV1 items through the registered decoder, HEVC through the OS.

#include "broimage/heif.h"

#include "broimage/codec.h"

#include "decode_heif.h"
#include "heif_os.h"
#include "heif_parse.h"
#include "heif_yuv.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace broimage {

namespace {

std::atomic<Av1Decoder> g_av1{nullptr};

}  // namespace

void set_av1_decoder(Av1Decoder decoder) { g_av1.store(decoder, std::memory_order_release); }
bool has_av1_decoder() { return g_av1.load(std::memory_order_acquire) != nullptr; }

bool is_heif(const uint8_t* data, std::size_t size) { return detail::heif_sniff(data, size); }

namespace detail {

namespace {

// A canvas no decode allocates past (each side and in all): a grid header
// can name any size.
constexpr int kMaxSide = 1 << 16;
constexpr uint64_t kMaxPixels = uint64_t(1) << 28;

bool fail(std::string* why, std::string msg) {
    if (why) *why = std::move(msg);
    return false;
}

bool coded_av1(const FourCC& t) { return fourcc_is(t, "av01"); }
bool coded_hevc(const FourCC& t) { return fourcc_is(t, "hvc1") || fourcc_is(t, "hev1"); }

std::string fourcc_text(const FourCC& t) {
    std::string s(t.data(), 4);
    for (char& c : s)
        if (!std::isprint(static_cast<unsigned char>(c))) c = '?';
    return s;
}

const char* kAlphaUrns[] = {"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha", "urn:mpeg:hevc:2015:auxid:1"};

bool is_alpha_aux(const HeifFile& f, const HeifItem& it) {
    const HeifProp* p = f.prop(it, "auxC");
    std::string t;
    if (!p || !read_auxc(*p, t)) return false;
    for (const char* u : kAlphaUrns)
        if (t == u) return true;
    return false;
}

// The alpha auxiliary item of `master`, or 0.
uint32_t alpha_of(const HeifFile& f, uint32_t master) {
    for (uint32_t id : f.refs_to(master, "auxl"))
        if (const HeifItem* it = f.item(id); it && is_alpha_aux(f, *it)) return id;
    return 0;
}

struct Grid {
    int rows = 0, columns = 0;
    uint32_t width = 0, height = 0;
};

bool read_grid(const HeifFile& f, const HeifItem& it, Grid& g, std::string* why) {
    std::vector<uint8_t> d;
    if (!f.item_data(it, d, why)) return false;
    if (d.size() < 8 || d[0] != 0) return fail(why, "HEIF: a malformed grid item");
    const bool wide = (d[1] & 1) != 0;
    if (wide && d.size() < 12) return fail(why, "HEIF: a malformed grid item");
    g.rows = d[2] + 1;
    g.columns = d[3] + 1;
    if (wide) {
        g.width = uint32_t(d[4]) << 24 | uint32_t(d[5]) << 16 | uint32_t(d[6]) << 8 | d[7];
        g.height = uint32_t(d[8]) << 24 | uint32_t(d[9]) << 16 | uint32_t(d[10]) << 8 | d[11];
    } else {
        g.width = uint32_t(d[4]) << 8 | d[5];
        g.height = uint32_t(d[6]) << 8 | d[7];
    }
    if (!g.width || !g.height) return fail(why, "HEIF: a grid of no size");
    return true;
}

// The coding of an image item: its own type, or a grid's first tile's.
FourCC coding_of(const HeifFile& f, const HeifItem& it) {
    if (fourcc_is(it.type, "grid")) {
        const auto tiles = f.refs_from(it.id, "dimg");
        if (!tiles.empty())
            if (const HeifItem* t = f.item(tiles[0])) return t->type;
    }
    return it.type;
}

// The size an item decodes to, before its transforms: a grid's output
// size, else its ispe.
bool item_size(const HeifFile& f, const HeifItem& it, int& w, int& h) {
    if (fourcc_is(it.type, "grid")) {
        Grid g;
        if (!read_grid(f, it, g, nullptr)) return false;
        w = int(std::min<uint32_t>(g.width, kMaxSide));
        h = int(std::min<uint32_t>(g.height, kMaxSide));
        return true;
    }
    HeifIspe s;
    const HeifProp* p = f.prop(it, "ispe");
    if (!p || !read_ispe(*p, s)) return false;
    w = int(std::min<uint32_t>(s.width, kMaxSide));
    h = int(std::min<uint32_t>(s.height, kMaxSide));
    return true;
}

// The clean aperture of an item w x h, when it has a valid one.
bool item_clap(const HeifFile& f, const HeifItem& it, int w, int h, int& x, int& y, int& cw, int& ch) {
    const HeifProp* p = f.prop(it, "clap");
    HeifClap c;
    return p && read_clap(*p, c) && clap_rect(c, w, h, x, y, cw, ch);
}

// ---- Orientation -------------------------------------------------------------

using M2 = std::array<int, 4>;  // row-major 2x2, y down

M2 mul(const M2& a, const M2& b) {
    return {a[0] * b[0] + a[1] * b[2], a[0] * b[1] + a[1] * b[3], a[2] * b[0] + a[3] * b[2],
            a[2] * b[1] + a[3] * b[3]};
}

// The matrix of each EXIF orientation 1..8: stored to displayed.
constexpr M2 kExif[8] = {{1, 0, 0, 1},  {-1, 0, 0, 1}, {-1, 0, 0, -1}, {1, 0, 0, -1},
                         {0, 1, 1, 0},  {0, -1, 1, 0}, {0, -1, -1, 0}, {0, 1, -1, 0}};

// irot / imir in their association order (MIAF: irot, then imir).
ExifOrientation transform_of(const HeifFile& f, const HeifItem& it) {
    M2 m{1, 0, 0, 1};
    for (const auto& pr : it.props) {
        if (pr.index == 0 || pr.index > f.props.size()) continue;
        const HeifProp& p = f.props[pr.index - 1];
        int v = 0;
        if (fourcc_is(p.type, "irot") && read_irot(p, v)) {
            for (int i = 0; i < v; ++i) m = mul(M2{0, 1, -1, 0}, m);  // a quarter turn anti-clockwise
        } else if (fourcc_is(p.type, "imir") && read_imir(p, v)) {
            // HEIF (2022 revision, the field renamed `mode`): 0 mirrors top
            // to bottom, 1 left to right. (The first edition's "vertical
            // axis" wording for 0 meant the same; WIC, libavif and Chrome agree.)
            m = mul(v == 0 ? M2{1, 0, 0, -1} : M2{-1, 0, 0, 1}, m);
        }
    }
    for (int i = 0; i < 8; ++i)
        if (kExif[i] == m) return ExifOrientation(i + 1);
    return ExifOrientation::Normal;
}

ExifOrientation inverse_of(ExifOrientation o) {
    if (o == ExifOrientation::Rotate90CW) return ExifOrientation::Rotate90CCW;
    if (o == ExifOrientation::Rotate90CCW) return ExifOrientation::Rotate90CW;
    return o;  // the others undo themselves
}

// ---- Rendering items ---------------------------------------------------------

enum class Plane { Color, Alpha };

struct Target {
    Image* canvas = nullptr;
    Plane plane = Plane::Color;
    int x0 = 0, y0 = 0;
    const HeifNclx* nclx = nullptr;  // the item's (or its grid's) colr
    uint16_t max_cll = 0;
    std::string error;
};

bool alloc_canvas(Image& c, uint64_t w, uint64_t h, std::string* why) {
    if (w == 0 || h == 0 || w > uint64_t(kMaxSide) || h > uint64_t(kMaxSide) || w * h > kMaxPixels)
        return fail(why, "HEIF: the image is too large to decode");
    c.width = int(w);
    c.height = int(h);
    c.channels = 4;
    c.pixels.assign(std::size_t(w * h * 4), 0);
    for (std::size_t i = 3; i < c.pixels.size(); i += 4) c.pixels[i] = 255;
    return true;
}

bool sink_frame(void* ctx, const YuvPlanes& fr) {
    Target& t = *static_cast<Target*>(ctx);
    if (fr.width <= 0 || fr.height <= 0 || !fr.planes[0] || (!fr.monochrome && (!fr.planes[1] || !fr.planes[2])) ||
        (fr.bit_depth != 8 && fr.bit_depth != 10 && fr.bit_depth != 12)) {
        t.error = "AVIF: the AV1 decoder returned a frame broimage cannot read";
        return false;
    }
    if (t.canvas->pixels.empty() && !alloc_canvas(*t.canvas, uint64_t(fr.width), uint64_t(fr.height), &t.error))
        return false;
    Image& c = *t.canvas;
    if (t.plane == Plane::Alpha) {
        yuv_to_alpha(fr, fr.full_range, c.pixels.data(), c.width, c.height, t.x0, t.y0);
        return true;
    }
    ColorSpec cs;
    if (t.nclx && t.nclx->present) {
        cs.primaries = t.nclx->primaries;
        cs.transfer = t.nclx->transfer;
        cs.matrix = t.nclx->matrix;
        cs.full_range = t.nclx->full_range;
    } else {
        cs.primaries = fr.color_primaries;
        cs.transfer = fr.transfer;
        cs.matrix = fr.matrix;
        cs.full_range = fr.full_range;
    }
    // An unspecified matrix in the item's colr defers to the bitstream.
    if (cs.matrix == 2) cs.matrix = fr.matrix;
    cs.max_cll = t.max_cll;
    yuv_to_rgba(fr, cs, c.pixels.data(), c.width, c.height, t.x0, t.y0);
    return true;
}

// An AV1 coded image (item data, or a sample) with its av1C, into `t`.
bool render_av1(const uint8_t* config, std::size_t config_size, std::vector<uint8_t> data, Target& t,
                std::string* why) {
    const Av1Decoder dec = g_av1.load(std::memory_order_acquire);
    if (!dec) return fail(why, "AVIF: this build has no AV1 decoder");
    // av1C's configOBUs (a sequence header, when the writer copied it there)
    // ahead of the item's own OBUs; a repeated sequence header is harmless.
    if (config && config_size > 4) data.insert(data.begin(), config + 4, config + config_size);
    std::string dwhy;
    if (!dec(data.data(), data.size(), sink_frame, &t, &dwhy)) {
        if (!t.error.empty()) return fail(why, t.error);
        return fail(why, dwhy.empty() ? std::string("AVIF: the AV1 data did not decode") : "AVIF: " + dwhy);
    }
    if (t.canvas->pixels.empty()) return fail(why, "AVIF: the AV1 decoder returned no frame");
    return true;
}

bool render_item(const HeifFile& f, uint32_t id, Target t, int depth, std::string* why);

bool render_grid(const HeifFile& f, const HeifItem& it, const Target& t, int depth, std::string* why) {
    Grid g;
    if (!read_grid(f, it, g, why)) return false;
    const auto tiles = f.refs_from(it.id, "dimg");
    if (tiles.size() != std::size_t(g.rows) * std::size_t(g.columns))
        return fail(why, "HEIF: a grid's tile count does not match its rows and columns");
    if (t.canvas->pixels.empty() && !alloc_canvas(*t.canvas, g.width, g.height, why)) return false;
    // Every tile has the same size (HEIF 6.6.2.3); its ispe gives it.
    const HeifItem* first = f.item(tiles[0]);
    int tw = 0, th = 0;
    if (!first || !item_size(f, *first, tw, th)) return fail(why, "HEIF: a grid tile has no size (ispe)");
    if (uint64_t(tw) * uint64_t(g.columns) < g.width || uint64_t(th) * uint64_t(g.rows) < g.height)
        return fail(why, "HEIF: a grid's tiles do not cover its output size");
    HeifNclx grid_nclx;
    if (const HeifProp* p = f.prop(it, "colr")) read_nclx(*p, grid_nclx);
    for (int r = 0; r < g.rows; ++r) {
        for (int c = 0; c < g.columns; ++c) {
            Target tt = t;
            tt.x0 = t.x0 + c * tw;
            tt.y0 = t.y0 + r * th;
            if (grid_nclx.present) tt.nclx = &grid_nclx;
            if (tt.x0 >= t.canvas->width || tt.y0 >= t.canvas->height) continue;
            if (!render_item(f, tiles[std::size_t(r) * std::size_t(g.columns) + std::size_t(c)], tt, depth + 1, why))
                return false;
        }
    }
    return true;
}

bool render_item(const HeifFile& f, uint32_t id, Target t, int depth, std::string* why) {
    if (depth > 2) return fail(why, "HEIF: derived images nest too deeply");
    const HeifItem* it = f.item(id);
    if (!it) return fail(why, "HEIF: a referenced item does not exist");
    if (fourcc_is(it->type, "grid")) return render_grid(f, *it, t, depth, why);
    HeifNclx own;
    if (const HeifProp* p = f.prop(*it, "colr"); p && read_nclx(*p, own)) t.nclx = &own;
    if (const HeifProp* p = f.prop(*it, "clli")) {
        uint16_t fall = 0;
        read_clli(*p, t.max_cll, fall);
    }
    if (coded_av1(it->type)) {
        std::vector<uint8_t> data;
        if (!f.item_data(*it, data, why)) return false;
        const HeifProp* cfg = f.prop(*it, "av1C");
        return render_av1(cfg ? cfg->body : nullptr, cfg ? cfg->size : 0, std::move(data), t, why);
    }
    if (fourcc_is(it->type, "jpeg")) {
        std::vector<uint8_t> data;
        if (!f.item_data(*it, data, why)) return false;
        Image j;
        if (!decode_memory(data.data(), data.size(), j, why)) return false;
        if (t.canvas->pixels.empty() && !alloc_canvas(*t.canvas, uint64_t(j.width), uint64_t(j.height), why))
            return false;
        Image& c = *t.canvas;
        for (int y = 0; y < j.height && t.y0 + y < c.height; ++y)
            for (int x = 0; x < j.width && t.x0 + x < c.width; ++x) {
                const uint8_t* s = j.pixels.data() + (std::size_t(y) * j.width + x) * 4;
                uint8_t* d = c.pixels.data() + (std::size_t(t.y0 + y) * c.width + std::size_t(t.x0 + x)) * 4;
                if (t.plane == Plane::Alpha) d[3] = s[0];
                else std::memcpy(d, s, 3);
            }
        return true;
    }
    if (coded_hevc(it->type))
        return fail(why, "HEIC: an HEVC item can only be decoded by the operating system, whole");
    return fail(why, "HEIF: '" + fourcc_text(it->type) + "' items are not supported");
}

void crop(Image& img, int x, int y, int w, int h) {
    if (x == 0 && y == 0 && w == img.width && h == img.height) return;
    std::vector<uint8_t> out(std::size_t(w) * std::size_t(h) * 4);
    for (int r = 0; r < h; ++r)
        std::memcpy(out.data() + std::size_t(r) * w * 4,
                    img.pixels.data() + (std::size_t(y + r) * img.width + std::size_t(x)) * 4, std::size_t(w) * 4);
    img.width = w;
    img.height = h;
    img.pixels = std::move(out);
}

void unpremultiply(Image& img) {
    for (std::size_t i = 0; i + 3 < img.pixels.size(); i += 4) {
        const unsigned a = img.pixels[i + 3];
        if (a == 0 || a == 255) continue;
        for (int k = 0; k < 3; ++k)
            img.pixels[i + std::size_t(k)] = uint8_t(std::min(255u, (img.pixels[i + std::size_t(k)] * 255u + a / 2) / a));
    }
}

// An image item with its alpha, premultiplication and clean aperture: the
// stored picture of `id` (no irot / imir).
bool render_image(const HeifFile& f, uint32_t id, Image& out, std::string* why) {
    out = Image{};
    Target t;
    t.canvas = &out;
    if (!render_item(f, id, t, 0, why)) {
        out = Image{};
        return false;
    }
    if (const uint32_t alpha = alpha_of(f, id)) {
        Image a;
        Target at;
        at.canvas = &a;
        at.plane = Plane::Alpha;
        std::string awhy;
        if (render_item(f, alpha, at, 0, &awhy)) {
            // The same size as the colour image (MIAF); nearest otherwise.
            for (int y = 0; y < out.height; ++y) {
                const int sy = int(int64_t(y) * a.height / out.height);
                for (int x = 0; x < out.width; ++x) {
                    const int sx = int(int64_t(x) * a.width / out.width);
                    out.pixels[(std::size_t(y) * out.width + std::size_t(x)) * 4 + 3] =
                        a.pixels[(std::size_t(sy) * a.width + std::size_t(sx)) * 4 + 3];
                }
            }
            const auto prem = f.refs_from(id, "prem");
            if (std::find(prem.begin(), prem.end(), alpha) != prem.end()) unpremultiply(out);
        }
        // An alpha plane that does not decode leaves the image opaque.
    }
    if (const HeifItem* it = f.item(id)) {
        int x = 0, y = 0, cw = 0, ch = 0;
        if (item_clap(f, *it, out.width, out.height, x, y, cw, ch)) crop(out, x, y, cw, ch);
    }
    return true;
}

// ---- A single-item file, for the OS ----------------------------------------------

void put16(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(uint8_t(x >> 8));
    v.push_back(uint8_t(x));
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    put16(v, x >> 16);
    put16(v, x & 0xFFFF);
}
void put4(std::vector<uint8_t>& v, const char* s) { v.insert(v.end(), s, s + 4); }
std::size_t open_box(std::vector<uint8_t>& v, const char* type) {
    const std::size_t at = v.size();
    put32(v, 0);
    put4(v, type);
    return at;
}
void close_box(std::vector<uint8_t>& v, std::size_t at) {
    const uint32_t n = uint32_t(v.size() - at);
    v[at] = uint8_t(n >> 24);
    v[at + 1] = uint8_t(n >> 16);
    v[at + 2] = uint8_t(n >> 8);
    v[at + 3] = uint8_t(n);
}

// `it` (a coded item) alone as the primary image of a new file: what an OS
// decoder that only reads primary images is given for a thumbnail.
bool single_item_file(const HeifFile& f, const HeifItem& it, std::vector<uint8_t>& out) {
    std::vector<uint8_t> data;
    if (!f.item_data(it, data) || it.props.size() > 127) return false;
    out.clear();
    std::size_t b = open_box(out, "ftyp");
    put4(out, coded_av1(it.type) ? "avif" : "heic");
    put32(out, 0);
    put4(out, "mif1");
    put4(out, coded_av1(it.type) ? "avif" : "heic");
    close_box(out, b);
    const std::size_t meta = open_box(out, "meta");
    put32(out, 0);
    b = open_box(out, "hdlr");
    put32(out, 0);
    put32(out, 0);
    put4(out, "pict");
    put32(out, 0);
    put32(out, 0);
    put32(out, 0);
    out.push_back(0);
    close_box(out, b);
    b = open_box(out, "pitm");
    put32(out, 0);
    put16(out, 1);
    close_box(out, b);
    b = open_box(out, "iinf");
    put32(out, 0);
    put16(out, 1);
    const std::size_t infe = open_box(out, "infe");
    put32(out, 2u << 24);
    put16(out, 1);
    put16(out, 0);
    out.insert(out.end(), it.type.begin(), it.type.end());
    out.push_back(0);
    close_box(out, infe);
    close_box(out, b);
    b = open_box(out, "iloc");
    put32(out, 0);
    out.push_back(0x44);  // offset and length: 4 bytes
    out.push_back(0x00);
    put16(out, 1);
    put16(out, 1);
    put16(out, 0);
    put16(out, 1);
    const std::size_t offset_at = out.size();
    put32(out, 0);
    put32(out, uint32_t(data.size()));
    close_box(out, b);
    const std::size_t iprp = open_box(out, "iprp");
    const std::size_t ipco = open_box(out, "ipco");
    for (const auto& pr : it.props) {
        if (pr.index == 0 || pr.index > f.props.size()) return false;
        const HeifProp& p = f.props[pr.index - 1];
        put32(out, uint32_t(p.size + 8));
        out.insert(out.end(), p.type.begin(), p.type.end());
        out.insert(out.end(), p.body, p.body + p.size);
    }
    close_box(out, ipco);
    b = open_box(out, "ipma");
    put32(out, 0);
    put32(out, 1);
    put16(out, 1);
    out.push_back(uint8_t(it.props.size()));
    for (std::size_t i = 0; i < it.props.size(); ++i)
        out.push_back(uint8_t((it.props[i].essential ? 0x80 : 0) | (i + 1)));
    close_box(out, b);
    close_box(out, iprp);
    close_box(out, meta);
    b = open_box(out, "mdat");
    const uint32_t at = uint32_t(out.size());
    out.insert(out.end(), data.begin(), data.end());
    close_box(out, b);
    out[offset_at] = uint8_t(at >> 24);
    out[offset_at + 1] = uint8_t(at >> 16);
    out[offset_at + 2] = uint8_t(at >> 8);
    out[offset_at + 3] = uint8_t(at);
    return true;
}

// ---- The primary image -------------------------------------------------------------

const char* kNoHevc =
    "HEIC: no HEVC decoder (HEIC images are decoded by the operating system, and this one has none)";

bool decode_os(const uint8_t* data, std::size_t size, const HeifFile& f, const HeifItem* primary, Image& out,
               std::string* why) {
    std::string avail;
    if (!os_heif_available(&avail)) return fail(why, avail.empty() ? kNoHevc : avail);
    bool oriented = false;
    if (!os_heif_decode(data, size, out, oriented, why)) {
        out = Image{};
        return false;
    }
    if (!primary) return true;
    const ExifOrientation o = transform_of(f, *primary);
    // Where a quarter turn tells, the size says whether the OS turned it.
    int w = 0, h = 0, c = 0;
    if (int(o) >= 5 && heif_probe(data, size, w, h, c) && w != h) {
        if (out.width == h && out.height == w) oriented = true;
        else if (out.width == w && out.height == h) oriented = false;
    }
    if (oriented) apply_exif_orientation(out, inverse_of(o));
    return true;
}

bool decode_track(const HeifFile& f, Image& out, std::string* why) {
    const HeifTrack& tr = f.track;
    if (!coded_av1(tr.codec)) {
        if (coded_hevc(tr.codec)) return decode_os(f.data, f.size, f, nullptr, out, why);
        return fail(why, "HEIF: the image sequence's '" + fourcc_text(tr.codec) + "' coding is not supported");
    }
    if (tr.first_sample_offset > f.size || tr.first_sample_size > f.size - tr.first_sample_offset)
        return fail(why, "AVIF: the first frame lies outside the file (truncated?)");
    std::vector<uint8_t> sample(f.data + tr.first_sample_offset,
                                f.data + tr.first_sample_offset + tr.first_sample_size);
    Target t;
    t.canvas = &out;
    return render_av1(fourcc_is(tr.config_type, "av1C") ? tr.config : nullptr, tr.config_size, std::move(sample), t,
                      why);
}

}  // namespace

bool heif_decode(const uint8_t* data, std::size_t size, Image& out, std::string* why) {
    out = Image{};
    HeifFile f;
    if (!heif_parse(data, size, f, why)) return false;
    const HeifItem* primary = f.primary ? f.item(f.primary) : nullptr;
    if (!primary) {
        if (f.track.present) return decode_track(f, out, why);
        return fail(why, "HEIF: no primary image");
    }
    const FourCC coding = coding_of(f, *primary);
    if (coded_hevc(coding)) return decode_os(data, size, f, primary, out, why);
    return render_image(f, primary->id, out, why);
}

bool heif_probe(const uint8_t* data, std::size_t size, int& width, int& height, int& channels) {
    HeifFile f;
    if (!heif_parse(data, size, f)) return false;
    const HeifItem* primary = f.primary ? f.item(f.primary) : nullptr;
    if (!primary) {
        if (!f.track.present || !f.track.width || !f.track.height) return false;
        width = int(f.track.width);
        height = int(f.track.height);
        channels = 3;
        return true;
    }
    int w = 0, h = 0;
    if (!item_size(f, *primary, w, h)) return false;
    int x = 0, y = 0, cw = 0, ch = 0;
    if (item_clap(f, *primary, w, h, x, y, cw, ch)) {
        w = cw;
        h = ch;
    }
    width = w;
    height = h;
    channels = alpha_of(f, primary->id) ? 4 : 3;
    return true;
}

ExifOrientation heif_orientation(const uint8_t* data, std::size_t size) {
    HeifFile f;
    if (!heif_parse(data, size, f) || !f.primary) return ExifOrientation::Normal;
    const HeifItem* primary = f.item(f.primary);
    return primary ? transform_of(f, *primary) : ExifOrientation::Normal;
}

bool heif_find_exif(const uint8_t* data, std::size_t size, const uint8_t*& tiff, std::size_t& tiff_size) {
    HeifFile f;
    if (!heif_parse(data, size, f)) return false;
    // The Exif item describing the primary image, else any.
    const HeifItem* exif = nullptr;
    for (const auto& it : f.items) {
        if (!fourcc_is(it.type, "Exif")) continue;
        const auto described = f.refs_from(it.id, "cdsc");
        const bool of_primary = std::find(described.begin(), described.end(), f.primary) != described.end();
        if (!exif || of_primary) exif = &it;
        if (of_primary) break;
    }
    if (!exif) return false;
    const uint8_t* p = nullptr;
    std::size_t n = 0;
    // Split across extents (rare): a copy that lives as long as the thread's next call.
    thread_local std::vector<uint8_t> joined;
    if (!f.item_span(*exif, p, n)) {
        if (!f.item_data(*exif, joined)) return false;
        p = joined.data();
        n = joined.size();
    }
    // exif_tiff_header_offset, then the block: "Exif\0\0" or straight to the
    // TIFF header, which some writers put the offset past and some do not.
    if (n < 4) return false;
    const uint32_t off = uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 | uint32_t(p[2]) << 8 | p[3];
    auto tiff_at = [&](std::size_t at) {
        return at + 8 <= n && (std::memcmp(p + at, "II*\0", 4) == 0 || std::memcmp(p + at, "MM\0*", 4) == 0);
    };
    std::size_t at = 0;
    if (off <= n - 4 && tiff_at(4 + std::size_t(off))) at = 4 + off;
    else if (n >= 10 && std::memcmp(p + 4, "Exif\0\0", 6) == 0 && tiff_at(10)) at = 10;
    else if (tiff_at(4)) at = 4;
    else return false;
    tiff = p + at;
    tiff_size = n - at;
    return true;
}

}  // namespace detail

using namespace detail;

bool read_heif_info(const uint8_t* data, std::size_t size, HeifInfo& out, std::string* why) {
    out = HeifInfo{};
    HeifFile f;
    if (!heif_parse(data, size, f, why)) return false;
    out.major_brand.assign(f.major_brand.data(), 4);
    for (const auto& b : f.compatible_brands) out.compatible_brands.emplace_back(b.data(), 4);
    out.sequence = f.track.present;
    const HeifItem* primary = f.primary ? f.item(f.primary) : nullptr;
    FourCC coding{};
    if (primary) {
        coding = coding_of(f, *primary);
        int ch = 3;
        heif_probe(data, size, out.width, out.height, ch);
        out.has_alpha = ch == 4;
        out.orientation = transform_of(f, *primary);
        if (fourcc_is(primary->type, "grid")) {
            Grid g;
            if (read_grid(f, *primary, g, nullptr)) {
                out.grid = true;
                out.grid_rows = g.rows;
                out.grid_columns = g.columns;
            }
        }
        // Colour: the primary's colr, else (a grid) its first tile's.
        const HeifItem* colour_item = primary;
        if (fourcc_is(primary->type, "grid") && !f.prop(*primary, "colr")) {
            const auto tiles = f.refs_from(primary->id, "dimg");
            if (!tiles.empty() && f.item(tiles[0])) colour_item = f.item(tiles[0]);
        }
        for (const auto& pr : colour_item->props) {
            if (pr.index == 0 || pr.index > f.props.size()) continue;
            const HeifProp& p = f.props[pr.index - 1];
            if (!fourcc_is(p.type, "colr")) continue;
            HeifNclx n;
            const uint8_t* icc = nullptr;
            std::size_t icc_n = 0;
            if (read_nclx(p, n)) {
                out.primaries = n.primaries;
                out.transfer = n.transfer;
                out.matrix = n.matrix;
                out.full_range = n.full_range;
            } else if (read_icc(p, icc, icc_n)) {
                out.has_icc = true;
            }
        }
        out.hdr = is_hdr_transfer(out.transfer);
        std::vector<uint8_t> bits;
        if (const HeifProp* p = f.prop(*colour_item, "pixi"); p && read_pixi(*p, bits) && !bits.empty())
            out.bit_depth = bits[0];
        else if (const HeifProp* c = f.prop(*colour_item, "av1C"); c && c->size >= 3)
            out.bit_depth = (c->body[2] & 0x40) ? ((c->body[2] & 0x20) ? 12 : 10) : 8;
        else if (const HeifProp* h = f.prop(*colour_item, "hvcC"); h && h->size >= 21)
            out.bit_depth = (h->body[20] & 7) + 8;
        for (uint32_t id : f.refs_to(primary->id, "thmb")) {
            const HeifItem* t = f.item(id);
            HeifInfo::Thumb th;
            if (!t || !item_size(f, *t, th.width, th.height)) continue;
            th.codec.assign(coding_of(f, *t).data(), 4);
            out.thumbnails.push_back(th);
        }
    } else if (f.track.present) {
        coding = f.track.codec;
        out.width = int(f.track.width);
        out.height = int(f.track.height);
    }
    out.codec.assign(coding.data(), 4);
    const uint8_t* tiff = nullptr;
    std::size_t tn = 0;
    out.has_exif = heif_find_exif(data, size, tiff, tn);
    if (coded_av1(coding)) out.mime = "image/avif";
    else if (coded_hevc(coding)) out.mime = "image/heic";
    else out.mime = "image/heif";
    return true;
}

bool decode_heif_thumbnail(const uint8_t* data, std::size_t size, int min_side, Image& out, std::string* why) {
    out = Image{};
    HeifFile f;
    if (!heif_parse(data, size, f, why)) return false;
    if (!f.primary) return fail(why, "HEIF: no primary image");
    // The smallest thumbnail at least min_side on its shorter side, else the largest.
    const HeifItem* best = nullptr;
    int best_min = 0;
    for (uint32_t id : f.refs_to(f.primary, "thmb")) {
        const HeifItem* t = f.item(id);
        int w = 0, h = 0;
        if (!t || !item_size(f, *t, w, h)) continue;
        const int m = std::min(w, h);
        const bool big = m >= min_side, best_big = best && best_min >= min_side;
        if (!best || (big && (!best_big || m < best_min)) || (!big && !best_big && m > best_min)) {
            best = t;
            best_min = m;
        }
    }
    if (!best) return fail(why, "HEIF: the file has no thumbnail");
    const FourCC coding = coding_of(f, *best);
    if (coded_hevc(coding)) {
        std::vector<uint8_t> single;
        if (fourcc_is(best->type, "grid") || !single_item_file(f, *best, single))
            return fail(why, "HEIC: the thumbnail is not a single coded item");
        if (!decode_os(single.data(), single.size(), f, best, out, why)) return false;
    } else if (!render_image(f, best->id, out, why)) {
        return false;
    }
    apply_exif_orientation(out, transform_of(f, *best));
    return true;
}

// ---- Capability --------------------------------------------------------------------

namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = char(std::tolower(static_cast<unsigned char>(c)));
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.back()))) out.pop_back();
    while (!out.empty() && std::isspace(static_cast<unsigned char>(out.front()))) out.erase(out.begin());
    return out;
}

}  // namespace

bool can_decode(std::string_view type, std::string* why) {
    if (why) why->clear();
    std::string t = lower(type.substr(0, type.find(';')));
    if (t.find('/') == std::string::npos) {
        if (!t.empty() && t[0] == '.') t.erase(0, 1);
        static const std::pair<const char*, const char*> kExt[] = {
            {"png", "image/png"},   {"jpg", "image/jpeg"},   {"jpeg", "image/jpeg"}, {"jpe", "image/jpeg"},
            {"gif", "image/gif"},   {"bmp", "image/bmp"},    {"tga", "image/x-tga"}, {"psd", "image/vnd.adobe.photoshop"},
            {"hdr", "image/vnd.radiance"}, {"pnm", "image/x-portable-anymap"}, {"ppm", "image/x-portable-pixmap"},
            {"pgm", "image/x-portable-graymap"}, {"tif", "image/tiff"}, {"tiff", "image/tiff"},
            {"ico", "image/x-icon"}, {"cur", "image/x-icon"}, {"webp", "image/webp"}, {"svg", "image/svg+xml"},
            {"avif", "image/avif"}, {"heic", "image/heic"}, {"heif", "image/heif"}, {"hif", "image/heif"},
        };
        std::string mime;
        for (const auto& [e, m] : kExt)
            if (t == e) mime = m;
        if (mime.empty()) return false;
        t = mime;
    }
    static const char* kBuiltIn[] = {"image/png", "image/jpeg", "image/pjpeg", "image/gif", "image/bmp",
                                     "image/x-bmp", "image/x-ms-bmp", "image/x-tga", "image/tga",
                                     "image/vnd.adobe.photoshop", "image/vnd.radiance",
                                     "image/x-portable-anymap", "image/x-portable-pixmap",
                                     "image/x-portable-graymap", "image/tiff", "image/x-icon",
                                     "image/vnd.microsoft.icon"};
    for (const char* b : kBuiltIn)
        if (t == b) return true;
    if (t == "image/webp" || t == "image/svg+xml") {
        if (has_registered_codec(t == "image/webp" ? "webp" : "svg")) return true;
        if (why) *why = "this build registers no " + std::string(t == "image/webp" ? "WebP" : "SVG") + " decoder";
        return false;
    }
    if (t == "image/avif" || t == "image/avif-sequence") {
        if (has_av1_decoder()) return true;
        if (why) *why = "this build has no AV1 decoder";
        return false;
    }
    if (t == "image/heic" || t == "image/heif" || t == "image/heic-sequence" || t == "image/heif-sequence") {
        return os_heif_available(why);
    }
    return false;
}

}  // namespace broimage
