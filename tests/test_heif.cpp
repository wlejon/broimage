// HEIF (heif.h): the container reader, grids, alpha, transforms, the clean
// aperture, Exif and thumbnail items, and the YUV-to-RGB maths — through a
// mock AV1 decoder whose "bitstream" names a solid frame, so every path
// runs without a real codec. (bro's tests decode real AVIFs with dav1d.)
#include "broimage/decode.h"
#include "broimage/exif.h"
#include "broimage/heif.h"

#include "heif_writer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

using heifw::Bytes;

namespace {

// "MOCK" w h (u16) depth layout(0 420, 1 422, 2 444, 3 400) full matrix Y U V (u16)
struct Mock {
    int w = 16, h = 16, depth = 8, layout = 0;
    bool full = false;
    int matrix = 6;
    int y = 0, u = 0, v = 0;
};

Bytes mock(const Mock& m) {
    Bytes b{'M', 'O', 'C', 'K'};
    heifw::u16(b, uint32_t(m.w));
    heifw::u16(b, uint32_t(m.h));
    b.push_back(uint8_t(m.depth));
    b.push_back(uint8_t(m.layout));
    b.push_back(m.full ? 1 : 0);
    b.push_back(uint8_t(m.matrix));
    heifw::u16(b, uint32_t(m.y));
    heifw::u16(b, uint32_t(m.u));
    heifw::u16(b, uint32_t(m.v));
    return b;
}

int g_decodes = 0;

bool mock_decode(const uint8_t* d, std::size_t n, broimage::YuvSink sink, void* ctx, std::string* why) {
    ++g_decodes;
    // The av1C config OBUs may come first; find the payload.
    const uint8_t* p = nullptr;
    for (std::size_t i = 0; i + 4 <= n; ++i)
        if (std::memcmp(d + i, "MOCK", 4) == 0) {
            p = d + i;
            n -= i;
            break;
        }
    if (!p || n < 18) {
        if (why) *why = "not a mock frame";
        return false;
    }
    auto r16 = [&](int at) { return int(p[at]) << 8 | p[at + 1]; };
    Mock m;
    m.w = r16(4);
    m.h = r16(6);
    m.depth = p[8];
    m.layout = p[9];
    m.full = p[10] != 0;
    m.matrix = p[11];
    m.y = r16(12);
    m.u = r16(14);
    m.v = r16(16);
    broimage::YuvPlanes f;
    f.width = m.w;
    f.height = m.h;
    f.bit_depth = m.depth;
    f.monochrome = m.layout == 3;
    f.chroma_shift_x = (m.layout == 0 || m.layout == 1) ? 1 : 0;
    f.chroma_shift_y = m.layout == 0 ? 1 : 0;
    f.matrix = m.matrix;
    f.full_range = m.full;
    const int cw = (m.w + (1 << f.chroma_shift_x) - 1) >> f.chroma_shift_x;
    const int ch = (m.h + (1 << f.chroma_shift_y) - 1) >> f.chroma_shift_y;
    const int bpp = m.depth > 8 ? 2 : 1;
    std::vector<uint8_t> planes[3];
    const int vals[3] = {m.y, m.u, m.v};
    for (int k = 0; k < 3; ++k) {
        const int pw = k ? cw : m.w, ph = k ? ch : m.h;
        planes[k].resize(std::size_t(pw) * ph * bpp);
        for (int i = 0; i < pw * ph; ++i) {
            if (bpp == 2) {
                const uint16_t v = uint16_t(vals[k]);
                std::memcpy(&planes[k][std::size_t(i) * 2], &v, 2);
            } else {
                planes[k][std::size_t(i)] = uint8_t(vals[k]);
            }
        }
        f.planes[k] = planes[k].data();
        f.strides[k] = std::ptrdiff_t(pw) * bpp;
    }
    return sink(ctx, f);
}

// Full or limited Y'CbCr codes of an 8-bit RGB colour.
Mock yuv_of(int R, int G, int B, double kr, double kb, int depth, bool full, int matrix) {
    const double r = R / 255.0, g = G / 255.0, b = B / 255.0;
    const double y = kr * r + (1 - kr - kb) * g + kb * b;
    const double cb = (b - y) / (2 * (1 - kb)), cr = (r - y) / (2 * (1 - kr));
    const double maxv = (1 << depth) - 1, s = double(1 << (depth - 8));
    Mock m;
    m.depth = depth;
    m.full = full;
    m.matrix = matrix;
    if (full) {
        m.y = int(std::lround(y * maxv));
        m.u = int(std::lround(cb * maxv + (1 << (depth - 1))));
        m.v = int(std::lround(cr * maxv + (1 << (depth - 1))));
        m.u = std::min(m.u, int(maxv));  // Clip1: +0.5 lands one past the top code
        m.v = std::min(m.v, int(maxv));
    } else {
        m.y = int(std::lround((16 + 219 * y) * s));
        m.u = int(std::lround((128 + 224 * cb) * s));
        m.v = int(std::lround((128 + 224 * cr) * s));
    }
    return m;
}

bool near(const uint8_t* px, int r, int g, int b, int tol = 2) {
    return std::abs(px[0] - r) <= tol && std::abs(px[1] - g) <= tol && std::abs(px[2] - b) <= tol;
}

const uint8_t* at(const broimage::Image& img, int x, int y) {
    return img.pixels.data() + (std::size_t(y) * img.width + std::size_t(x)) * 4;
}

heifw::Item coded(uint32_t id, const Mock& m, std::vector<Bytes> extra = {}) {
    heifw::Item it;
    it.id = id;
    it.data = mock(m);
    it.props = {heifw::ispe(uint32_t(m.w), uint32_t(m.h)), heifw::av1c()};
    for (auto& e : extra) it.props.push_back(e);
    return it;
}

Bytes single(const Mock& m, std::vector<Bytes> extra = {}) {
    heifw::File f;
    f.items.push_back(coded(1, m, std::move(extra)));
    return heifw::write(f);
}

void test_colour() {
    // 8-bit 4:2:0, limited range, BT.601 from the bitstream (no colr).
    {
        Mock m = yuv_of(200, 50, 30, 0.299, 0.114, 8, false, 6);
        m.w = 17;  // odd: the last chroma column covers one luma column
        m.h = 9;
        const Bytes file = single(m);
        CHECK(broimage::is_heif(file.data(), file.size()));
        broimage::Image img;
        std::string why;
        CHECK(broimage::decode_memory(file.data(), file.size(), img, &why));
        CHECK(img.width == 17 && img.height == 9 && img.channels == 4);
        if (!img.pixels.empty()) {
            CHECK(near(at(img, 0, 0), 200, 50, 30));
            CHECK(near(at(img, 16, 8), 200, 50, 30));
            CHECK(at(img, 3, 3)[3] == 255);
        }
        int w = 0, h = 0, c = 0;
        CHECK(broimage::probe_dimensions_memory(file.data(), file.size(), &w, &h, &c));
        CHECK(w == 17 && h == 9 && c == 3);
    }
    // 10-bit 4:4:4, full range, BT.709 named by the item's nclx colr (the
    // bitstream says BT.601: the item wins).
    {
        Mock m = yuv_of(30, 160, 240, 0.2126, 0.0722, 10, true, 6);
        m.layout = 2;
        const Bytes file = single(m, {heifw::nclx(1, 13, 1, true)});
        broimage::Image img;
        CHECK(broimage::decode_memory(file.data(), file.size(), img));
        if (!img.pixels.empty()) CHECK(near(at(img, 5, 5), 30, 160, 240));
    }
    // 12-bit 4:2:2 limited BT.2020.
    {
        Mock m = yuv_of(90, 200, 120, 0.2627, 0.0593, 12, false, 9);
        m.layout = 1;
        const Bytes file = single(m, {heifw::nclx(9, 1, 9, false)});
        broimage::Image img;
        CHECK(broimage::decode_memory(file.data(), file.size(), img));
        if (!img.pixels.empty()) CHECK(near(at(img, 8, 2), 90, 200, 120));
    }
    // Monochrome, 12-bit full range.
    {
        Mock m;
        m.depth = 12;
        m.layout = 3;
        m.full = true;
        m.y = 2048;
        const Bytes file = single(m);
        broimage::Image img;
        CHECK(broimage::decode_memory(file.data(), file.size(), img));
        if (!img.pixels.empty()) CHECK(near(at(img, 1, 1), 128, 128, 128, 1));
    }
    // Identity matrix (GBR in Y, U, V), 4:4:4 full range.
    {
        Mock m;
        m.layout = 2;
        m.full = true;
        m.matrix = 0;
        m.y = 70;   // G
        m.u = 20;   // B
        m.v = 230;  // R
        const Bytes file = single(m);
        broimage::Image img;
        CHECK(broimage::decode_memory(file.data(), file.size(), img));
        if (!img.pixels.empty()) CHECK(near(at(img, 0, 0), 230, 70, 20, 0));
    }
}

void test_hdr() {
    // PQ 10-bit grey ramps: monotonic, SDR white (203 nits, PQ code ~0.58)
    // below full white, and the PQ peak at full white.
    auto grey = [](int code) {
        Mock m;
        m.depth = 10;
        m.layout = 2;
        m.full = true;
        m.matrix = 9;
        m.y = code;
        m.u = m.v = 512;
        const Bytes file = single(m, {heifw::nclx(9, 16, 9, true)});
        broimage::Image img;
        broimage::decode_memory(file.data(), file.size(), img);
        return img.pixels.empty() ? -1 : int(img.pixels[0]);
    };
    const int black = grey(0), white_sdr = grey(594), peak = grey(1023), mid = grey(300);
    CHECK(black == 0);
    CHECK(mid > black && white_sdr > mid && peak >= white_sdr);
    CHECK(white_sdr >= 220 && white_sdr <= 250);
    CHECK(peak == 255);
    broimage::HeifInfo info;
    Mock m;
    const Bytes file = single(m, {heifw::nclx(9, 18, 9, true)});
    CHECK(broimage::read_heif_info(file.data(), file.size(), info));
    CHECK(info.hdr && info.transfer == 18 && info.mime == "image/avif");
}

void test_grid_alpha_transforms() {
    // A 2x2 grid of 16x16 tiles, output 30 x 20 (the tiles overhang), with
    // an alpha grid, an idat-stored grid item and a tile split in two extents.
    const int colours[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}};
    heifw::File f;
    f.primary = 1;
    heifw::Item grid;
    grid.id = 1;
    grid.type = "grid";
    grid.data = heifw::grid_data(2, 2, 30, 20);
    grid.in_idat = true;
    grid.props = {heifw::ispe(30, 20), heifw::irot(1)};
    f.items.push_back(grid);
    heifw::Ref dimg{"dimg", 1, {}};
    for (int i = 0; i < 4; ++i) {
        Mock m = yuv_of(colours[i][0], colours[i][1], colours[i][2], 0.299, 0.114, 8, true, 6);
        m.layout = 2;
        heifw::Item t = coded(uint32_t(10 + i), m);
        t.hidden = true;
        t.split = i == 2;
        f.items.push_back(t);
        dimg.to.push_back(uint32_t(10 + i));
    }
    f.refs.push_back(dimg);
    // Alpha: a single 30 x 20 monochrome item at 50%.
    Mock a;
    a.w = 30;
    a.h = 20;
    a.layout = 3;
    a.full = true;
    a.y = 128;
    heifw::Item alpha = coded(20, a, {heifw::auxc("urn:mpeg:mpegB:cicp:systems:auxiliary:alpha")});
    alpha.hidden = true;
    f.items.push_back(alpha);
    f.refs.push_back({"auxl", 20, {1}});
    const Bytes file = heifw::write(f);

    broimage::Image img;
    std::string why;
    CHECK(broimage::decode_memory(file.data(), file.size(), img, &why));
    if (!why.empty()) std::fprintf(stderr, "  grid: %s\n", why.c_str());
    CHECK(img.width == 30 && img.height == 20);
    if (img.width == 30 && img.height == 20) {
        CHECK(near(at(img, 2, 2), 255, 0, 0));
        CHECK(near(at(img, 20, 2), 0, 255, 0));
        CHECK(near(at(img, 2, 18), 0, 0, 255));
        CHECK(near(at(img, 29, 19), 255, 255, 255));
        CHECK(at(img, 29, 19)[3] == 128);
    }
    int w = 0, h = 0, c = 0;
    CHECK(broimage::probe_dimensions_memory(file.data(), file.size(), &w, &h, &c));
    CHECK(w == 30 && h == 20 && c == 4);
    // irot 1 = a quarter turn anti-clockwise = EXIF 8.
    CHECK(broimage::read_exif_orientation(file.data(), file.size()) == broimage::ExifOrientation::Rotate90CCW);
    broimage::Image up;
    CHECK(broimage::decode_memory_oriented(file.data(), file.size(), up));
    CHECK(up.width == 20 && up.height == 30);
    // Turned anti-clockwise, the top-right (green) lands top-left.
    if (up.width == 20) CHECK(near(at(up, 2, 2), 0, 255, 0));

    broimage::HeifInfo info;
    CHECK(broimage::read_heif_info(file.data(), file.size(), info));
    CHECK(info.grid && info.grid_rows == 2 && info.grid_columns == 2 && info.has_alpha);
    CHECK(info.codec == "av01" && info.width == 30 && info.height == 20);

    // Bounded decode reads it, and refuses it past a limit.
    broimage::Animation anim;
    broimage::DecodeLimits lim;
    CHECK(broimage::decode_memory_bounded(file.data(), file.size(), lim, anim));
    CHECK(anim.width == 30 && anim.frames.size() == 1);
    lim.max_width = 16;
    CHECK(!broimage::decode_memory_bounded(file.data(), file.size(), lim, anim));
}

void test_orientation_table() {
    // irot then imir (MIAF order) to EXIF orientation.
    struct Case {
        int angle;
        int axis;  // -1: no imir
        broimage::ExifOrientation want;
    };
    using O = broimage::ExifOrientation;
    // imir mode 0 mirrors top to bottom, 1 left to right.
    const Case cases[] = {{0, -1, O::Normal},     {1, -1, O::Rotate90CCW}, {2, -1, O::Rotate180},
                          {3, -1, O::Rotate90CW}, {0, 0, O::FlipV},        {0, 1, O::FlipH},
                          {1, 0, O::Transpose},   {1, 1, O::Transverse},   {2, 0, O::FlipH},
                          {3, 0, O::Transverse},  {3, 1, O::Transpose}};
    for (const Case& k : cases) {
        std::vector<Bytes> props;
        if (k.angle) props.push_back(heifw::irot(k.angle));
        if (k.axis >= 0) props.push_back(heifw::imir(k.axis));
        Mock m;
        const Bytes file = single(m, props);
        CHECK(broimage::read_exif_orientation(file.data(), file.size()) == k.want);
    }
}

void test_clap_exif_thumbs() {
    // 20 x 16 with a centred 10 x 8 clean aperture, an Exif item saying
    // orientation 6 (ignored: HEIF orients by irot / imir), and a thumbnail.
    Mock m = yuv_of(10, 20, 30, 0.299, 0.114, 8, true, 6);
    m.w = 20;
    m.h = 16;
    m.layout = 2;
    heifw::File f;
    f.items.push_back(coded(1, m, {heifw::clap(10, 8, 0, 0), heifw::imir(1)}));
    heifw::Item exif;
    exif.id = 2;
    exif.type = "Exif";
    exif.data = heifw::exif_payload("MockCam", 6);
    exif.hidden = true;
    f.items.push_back(exif);
    f.refs.push_back({"cdsc", 2, {1}});
    Mock tm = yuv_of(250, 250, 0, 0.299, 0.114, 8, true, 6);
    tm.w = 8;
    tm.h = 6;
    tm.layout = 2;
    heifw::Item thumb = coded(3, tm, {heifw::irot(3)});
    thumb.hidden = true;
    f.items.push_back(thumb);
    f.refs.push_back({"thmb", 3, {1}});
    const Bytes file = heifw::write(f);

    broimage::Image img;
    CHECK(broimage::decode_memory(file.data(), file.size(), img));
    CHECK(img.width == 10 && img.height == 8);
    int w = 0, h = 0, c = 0;
    CHECK(broimage::probe_dimensions_memory(file.data(), file.size(), &w, &h, &c));
    CHECK(w == 10 && h == 8);

    broimage::ExifInfo ex;
    CHECK(broimage::read_exif(file.data(), file.size(), ex));
    CHECK(ex.make == "MockCam");
    CHECK(ex.orientation && *ex.orientation == 2);  // imir 1 (left-right), not the Exif tag's 6
    CHECK(broimage::read_exif_orientation(file.data(), file.size()) == broimage::ExifOrientation::FlipH);

    broimage::HeifInfo info;
    CHECK(broimage::read_heif_info(file.data(), file.size(), info));
    CHECK(info.has_exif && info.thumbnails.size() == 1);
    if (!info.thumbnails.empty()) CHECK(info.thumbnails[0].width == 8 && info.thumbnails[0].height == 6);
    broimage::Image t;
    CHECK(broimage::decode_heif_thumbnail(file.data(), file.size(), 4, t));
    CHECK(t.width == 6 && t.height == 8);  // its own irot 3: upright
    if (!t.pixels.empty()) CHECK(near(t.pixels.data(), 250, 250, 0));
    const Bytes plain = single(m);
    CHECK(!broimage::decode_heif_thumbnail(plain.data(), plain.size(), 4, t));
}

void test_failures() {
    Mock m;
    const Bytes file = single(m);
    // Truncated inside mdat: the size still probes, the decode fails cleanly.
    const std::size_t cut = file.size() - 6;
    int w = 0, h = 0, c = 0;
    CHECK(broimage::probe_dimensions_memory(file.data(), cut, &w, &h, &c));
    broimage::Image img;
    std::string why;
    CHECK(!broimage::decode_memory(file.data(), cut, img, &why));
    CHECK(!why.empty());
    // Every prefix fails or succeeds without reading out of bounds.
    for (std::size_t n = 0; n < file.size(); ++n) {
        broimage::Image x;
        broimage::decode_memory(file.data(), n, x);
        broimage::probe_dimensions_memory(file.data(), n, &w, &h, &c);
    }
    // Without an AV1 decoder: a reason, and the size still probes.
    broimage::set_av1_decoder(nullptr);
    CHECK(!broimage::decode_memory(file.data(), file.size(), img, &why));
    CHECK(why.find("AV1 decoder") != std::string::npos);
    CHECK(broimage::probe_dimensions_memory(file.data(), file.size(), &w, &h, &c));
    std::string cwhy;
    CHECK(!broimage::can_decode("image/avif", &cwhy));
    CHECK(!cwhy.empty());
    broimage::set_av1_decoder(mock_decode);
    CHECK(broimage::can_decode("image/avif"));
    CHECK(broimage::can_decode(".avif"));
    CHECK(broimage::can_decode("PNG"));
    CHECK(broimage::can_decode("image/jpeg; charset=x"));
    CHECK(!broimage::can_decode("image/x-unknown"));
    std::string hwhy;
    const bool heic = broimage::can_decode("image/heic", &hwhy);
    CHECK(heic || !hwhy.empty());
    std::printf("  HEIC on this machine: %s%s\n", heic ? "yes" : "no: ", heic ? "" : hwhy.c_str());
    // A grid whose tile count is wrong fails with a reason.
    heifw::File g;
    heifw::Item gi;
    gi.id = 1;
    gi.type = "grid";
    gi.data = heifw::grid_data(2, 2, 32, 32);
    gi.props = {heifw::ispe(32, 32)};
    g.items.push_back(gi);
    g.items.push_back(coded(2, m));
    g.refs.push_back({"dimg", 1, {2}});
    const Bytes bad = heifw::write(g);
    CHECK(!broimage::decode_memory(bad.data(), bad.size(), img, &why));
    CHECK(why.find("tile count") != std::string::npos);
    // An absurd grid size is refused before allocating.
    g.items[0].data = heifw::grid_data(1, 1, 65535, 65535);
    g.refs[0].to = {2};
    const Bytes huge = heifw::write(g);
    CHECK(!broimage::decode_memory(huge.data(), huge.size(), img, &why));
}

void test_sequence_still() {
    // An image sequence with no meta box: its first sample is the still.
    Mock m = yuv_of(0, 128, 255, 0.299, 0.114, 8, true, 6);
    m.layout = 2;
    m.w = 8;
    m.h = 4;
    const Bytes sample = mock(m);
    using namespace heifw;
    Bytes ftyp;
    cc(ftyp, "avis");
    u32(ftyp, 0);
    cc(ftyp, "avis");
    cc(ftyp, "msf1");
    Bytes file = box("ftyp", ftyp);
    const Bytes mdat = box("mdat", sample);
    const uint32_t sample_at = uint32_t(file.size() + 8);
    file.insert(file.end(), mdat.begin(), mdat.end());
    // moov / trak / mdia (hdlr pict) / minf / stbl (stsd av01 + av1C, stsz, stco)
    Bytes entry(6, 0);
    u16(entry, 1);
    Bytes vis(16, 0);
    entry.insert(entry.end(), vis.begin(), vis.end());
    u16(entry, 8);
    u16(entry, 4);
    Bytes rest(50, 0);
    entry.insert(entry.end(), rest.begin(), rest.end());
    const Bytes cfg = av1c();
    entry.insert(entry.end(), cfg.begin(), cfg.end());
    Bytes stsd_body;
    u32(stsd_body, 1);
    const Bytes av01 = box("av01", entry);
    stsd_body.insert(stsd_body.end(), av01.begin(), av01.end());
    Bytes stsz;
    u32(stsz, 0);
    u32(stsz, 1);
    u32(stsz, uint32_t(sample.size()));
    Bytes stco;
    u32(stco, 1);
    u32(stco, sample_at);
    Bytes stbl = fullbox("stsd", 0, 0, stsd_body);
    for (const Bytes& b : {fullbox("stsz", 0, 0, stsz), fullbox("stco", 0, 0, stco)}) stbl.insert(stbl.end(), b.begin(), b.end());
    Bytes hdlr;
    u32(hdlr, 0);
    cc(hdlr, "pict");
    u32(hdlr, 0);
    u32(hdlr, 0);
    u32(hdlr, 0);
    str(hdlr, "");
    Bytes mdia = fullbox("hdlr", 0, 0, hdlr);
    const Bytes minf = box("minf", box("stbl", stbl));
    mdia.insert(mdia.end(), minf.begin(), minf.end());
    const Bytes moov = box("moov", box("trak", box("mdia", mdia)));
    file.insert(file.end(), moov.begin(), moov.end());

    broimage::Image img;
    std::string why;
    CHECK(broimage::decode_memory(file.data(), file.size(), img, &why));
    if (!why.empty()) std::fprintf(stderr, "  sequence: %s\n", why.c_str());
    CHECK(img.width == 8 && img.height == 4);
    if (!img.pixels.empty()) CHECK(near(img.pixels.data(), 0, 128, 255));
    int w = 0, h = 0, c = 0;
    CHECK(broimage::probe_dimensions_memory(file.data(), file.size(), &w, &h, &c));
    CHECK(w == 8 && h == 4);
}

}  // namespace

int main() {
    broimage::set_av1_decoder(mock_decode);
    test_colour();
    test_hdr();
    test_grid_alpha_transforms();
    test_orientation_table();
    test_clap_exif_thumbs();
    test_sequence_still();
    test_failures();
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test_heif: all passed\n");
    return 0;
}
