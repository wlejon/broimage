// HEIC through the operating system's decoder (heif.h). On Windows a HEIC
// is made with WIC's own HEIF encoder (which needs the same HEVC extension
// the decoder does) and decoded back; the test skips, saying why, where
// there is no HEVC codec. A copy with an irot property added checks that
// the OS hands back the stored pixels and the orientation comes from the
// container. BROIMAGE_HEIC_FIXTURE_OUT=<path> keeps the made HEIC;
// BROIMAGE_HEIC_FIXTURE=<path> decodes a given one instead (any platform).
#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "broimage/decode.h"
#include "broimage/heif.h"

#include "heif_os.h"
#include "heif_parse.h"
#include "heif_writer.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#endif

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

namespace {

constexpr int kW = 64, kH = 48;

// Red top-left, green bottom-left, blue on the right: no flip or turn of it
// looks like another.
std::array<uint8_t, 3> pattern(int x, int y) {
    if (x >= kW / 2) return {0, 0, 255};
    return y < kH / 2 ? std::array<uint8_t, 3>{255, 0, 0} : std::array<uint8_t, 3>{0, 255, 0};
}

#ifdef _WIN32
// A kW x kH image, red on the left half and blue on the right, as HEIC.
bool wic_encode(std::vector<uint8_t>& out, std::string& why) {
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    IWICImagingFactory* fac = nullptr;
    IStream* mem = nullptr;
    IWICBitmapEncoder* enc = nullptr;
    IWICBitmapFrameEncode* fr = nullptr;
    IPropertyBag2* bag = nullptr;
    bool ok = false;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&fac));
    if (SUCCEEDED(hr)) hr = CreateStreamOnHGlobal(nullptr, TRUE, &mem);
    if (SUCCEEDED(hr)) hr = fac->CreateEncoder(GUID_ContainerFormatHeif, nullptr, &enc);
    if (FAILED(hr)) why = "no WIC HEIF encoder";
    if (SUCCEEDED(hr)) hr = enc->Initialize(mem, WICBitmapEncoderNoCache);
    if (SUCCEEDED(hr)) hr = enc->CreateNewFrame(&fr, &bag);
    if (SUCCEEDED(hr)) {
        PROPBAG2 opt{};
        opt.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
        VARIANT v;
        VariantInit(&v);
        v.vt = VT_R4;
        v.fltVal = 1.0f;
        bag->Write(1, &opt, &v);
        hr = fr->Initialize(bag);
    }
    if (SUCCEEDED(hr)) hr = fr->SetSize(kW, kH);
    WICPixelFormatGUID fmt = GUID_WICPixelFormat32bppBGRA;
    // The encoder may pick another format (BGR without alpha): WriteSource
    // converts to whatever it chose.
    if (SUCCEEDED(hr)) hr = fr->SetPixelFormat(&fmt);
    if (SUCCEEDED(hr)) {
        std::vector<uint8_t> px(std::size_t(kW) * kH * 4);
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) {
                uint8_t* p = &px[(std::size_t(y) * kW + x) * 4];
                const auto c = pattern(x, y);
                p[0] = c[2];
                p[1] = c[1];
                p[2] = c[0];
                p[3] = 255;
            }
        IWICBitmap* bmp = nullptr;
        hr = fac->CreateBitmapFromMemory(kW, kH, GUID_WICPixelFormat32bppBGRA, kW * 4, UINT(px.size()), px.data(), &bmp);
        if (SUCCEEDED(hr)) hr = fr->WriteSource(bmp, nullptr);
        if (bmp) bmp->Release();
    }
    if (SUCCEEDED(hr)) hr = fr->Commit();
    if (SUCCEEDED(hr)) hr = enc->Commit();
    if (FAILED(hr) && why.empty()) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "WIC could not encode HEIC (HRESULT 0x%08lX)", static_cast<unsigned long>(hr));
        why = buf;
    }
    if (SUCCEEDED(hr)) {
        STATSTG st{};
        LARGE_INTEGER zero{};
        mem->Stat(&st, STATFLAG_NONAME);
        mem->Seek(zero, STREAM_SEEK_SET, nullptr);
        out.resize(std::size_t(st.cbSize.QuadPart));
        ULONG got = 0;
        mem->Read(out.data(), ULONG(out.size()), &got);
        ok = got == out.size() && !out.empty();
    }
    if (bag) bag->Release();
    if (fr) fr->Release();
    if (enc) enc->Release();
    if (mem) mem->Release();
    if (fac) fac->Release();
    return ok;
}
#endif

// The same file with irot (`angle` quarter turns anti-clockwise) and imir
// (`axis`, -1 for none) added to the primary item, rewritten from what the
// reader saw.
bool with_transform(const std::vector<uint8_t>& src, std::vector<uint8_t>& out, int angle, int axis) {
    broimage::detail::HeifFile f;
    if (!broimage::detail::heif_parse(src.data(), src.size(), f) || !f.primary) return false;
    heifw::File w;
    w.major = std::string(f.major_brand.data(), 4);
    w.compat.clear();
    for (const auto& b : f.compatible_brands) w.compat.emplace_back(b.data(), 4);
    w.primary = f.primary;
    for (const auto& it : f.items) {
        heifw::Item wi;
        wi.id = it.id;
        wi.type = std::string(it.type.data(), 4);
        wi.hidden = it.hidden;
        if (!f.item_data(it, wi.data)) return false;
        for (const auto& pr : it.props) {
            const auto& p = f.props[pr.index - 1];
            heifw::Bytes b;
            heifw::u32(b, uint32_t(p.size + 8));
            b.insert(b.end(), p.type.begin(), p.type.end());
            b.insert(b.end(), p.body, p.body + p.size);
            wi.props.push_back(b);
        }
        if (it.id == f.primary && angle) wi.props.push_back(heifw::irot(angle));
        if (it.id == f.primary && axis >= 0) wi.props.push_back(heifw::imir(axis));
        w.items.push_back(wi);
    }
    for (const auto& r : f.refs) w.refs.push_back({std::string(r.type.data(), 4), r.from, r.to});
    out = heifw::write(w);
    return true;
}

bool close_to(const uint8_t* p, int r, int g, int b, int tol) {
    return std::abs(p[0] - r) <= tol && std::abs(p[1] - g) <= tol && std::abs(p[2] - b) <= tol;
}

// `img` is the pattern with orientation `o` applied (quadrant centres, lossy tolerance).
bool shows(const broimage::Image& img, broimage::ExifOrientation o) {
    broimage::Image want;
    want.width = kW;
    want.height = kH;
    want.channels = 4;
    want.pixels.resize(std::size_t(kW) * kH * 4);
    for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x) {
            const auto c = pattern(x, y);
            uint8_t* p = &want.pixels[(std::size_t(y) * kW + x) * 4];
            p[0] = c[0];
            p[1] = c[1];
            p[2] = c[2];
            p[3] = 255;
        }
    broimage::apply_exif_orientation(want, o);
    if (img.width != want.width || img.height != want.height || img.pixels.empty()) return false;
    for (int fy : {1, 3})
        for (int fx : {1, 3}) {
            const int x = img.width * fx / 4, y = img.height * fy / 4;
            const uint8_t* a = img.pixels.data() + (std::size_t(y) * img.width + x) * 4;
            const uint8_t* b = want.pixels.data() + (std::size_t(y) * want.width + x) * 4;
            if (!close_to(a, b[0], b[1], b[2], 32)) return false;
        }
    return true;
}

void check_decode(const std::vector<uint8_t>& heic, bool known_content) {
    broimage::Image img;
    std::string why;
    CHECK(broimage::decode_memory(heic.data(), heic.size(), img, &why));
    if (!why.empty()) std::fprintf(stderr, "  decode: %s\n", why.c_str());
    int w = 0, h = 0, c = 0;
    CHECK(broimage::probe_dimensions_memory(heic.data(), heic.size(), &w, &h, &c));
    CHECK(img.width == w && img.height == h);
    broimage::HeifInfo info;
    CHECK(broimage::read_heif_info(heic.data(), heic.size(), info));
    CHECK(info.mime == "image/heic");
    if (!known_content || img.pixels.empty()) return;
    CHECK(shows(img, broimage::ExifOrientation::Normal));

    // Turned and mirrored by the container: decode gives it as stored,
    // oriented turns it as irot / imir say.
    struct Case {
        int angle, axis;
        broimage::ExifOrientation want;
    };
    using O = broimage::ExifOrientation;
    // imir mode 0 mirrors top to bottom, 1 left to right.
    const Case cases[] = {{1, -1, O::Rotate90CCW}, {3, -1, O::Rotate90CW}, {2, -1, O::Rotate180},
                          {0, 0, O::FlipV},        {0, 1, O::FlipH},       {1, 0, O::Transpose},
                          {1, 1, O::Transverse}};
    for (const Case& k : cases) {
        std::vector<uint8_t> t;
        CHECK(with_transform(heic, t, k.angle, k.axis));
        broimage::Image raw;
        bool oriented = false;
        if (broimage::detail::os_heif_decode(t.data(), t.size(), raw, oriented)) {
            int as = 0;
            for (int o = 1; o <= 8 && !as; ++o)
                if (shows(raw, broimage::ExifOrientation(o))) as = o;
            std::printf("  irot %d imir %d (EXIF %d): the OS returns it as EXIF %d of the stored image\n", k.angle,
                        k.axis, int(k.want), as);
        }
        broimage::Image stored, up;
        CHECK(broimage::decode_memory(t.data(), t.size(), stored, &why));
        CHECK(shows(stored, O::Normal));
        CHECK(broimage::read_exif_orientation(t.data(), t.size()) == k.want);
        CHECK(broimage::decode_memory_oriented(t.data(), t.size(), up));
        CHECK(shows(up, k.want));
    }
}

}  // namespace

int main() {
    std::string why;
    const bool available = broimage::can_decode("image/heic", &why);
    std::printf("HEIC decoder: %s%s\n", available ? "available" : "unavailable: ", available ? "" : why.c_str());
    if (!available) CHECK(!why.empty());

    if (const char* in = std::getenv("BROIMAGE_HEIC_FIXTURE")) {
        std::ifstream f(in, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        CHECK(!bytes.empty());
        if (available) {
            check_decode(bytes, false);
        } else {
            broimage::Image img;
            std::string dwhy;
            CHECK(!broimage::decode_memory(bytes.data(), bytes.size(), img, &dwhy));
            CHECK(dwhy == why || dwhy.find("HEIC") != std::string::npos);
            int w = 0, h = 0, c = 0;
            CHECK(broimage::probe_dimensions_memory(bytes.data(), bytes.size(), &w, &h, &c));
        }
    }

#ifdef _WIN32
    if (!available) {
        std::printf("SKIP: no HEVC decoder here (%s)\n", why.c_str());
        return g_failed ? 1 : 0;
    }
    std::vector<uint8_t> heic;
    std::string ewhy;
    if (!wic_encode(heic, ewhy)) {
        std::printf("SKIP: cannot make a HEIC to decode (%s)\n", ewhy.c_str());
        return g_failed ? 1 : 0;
    }
    if (const char* out = std::getenv("BROIMAGE_HEIC_FIXTURE_OUT")) {
        std::ofstream f(out, std::ios::binary);
        f.write(reinterpret_cast<const char*>(heic.data()), std::streamsize(heic.size()));
        std::printf("wrote %s (%zu bytes)\n", out, heic.size());
    }
    check_decode(heic, true);
#endif
    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test_heif_os: all passed\n");
    return 0;
}
