// HEIC through Windows Imaging Component (heif_os.h). WIC's HEIF decoder
// (CLSID_WICHeifDecoder) reads the container and hands the HEVC bitstream
// to a Media Foundation HEVC decoder, which Windows does not ship: it comes
// with the "HEVC Video Extensions" from the Microsoft Store (or a GPU
// vendor's decoder). Availability is the HEIF decoder plus an HEVC decoder
// MFT, probed once.

#include "heif_os.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <mfapi.h>
#include <mfobjects.h>
#include <mftransform.h>
#include <wincodec.h>

#include <cstdio>
#include <mutex>
#include <string>

namespace broimage::detail {

namespace {

const char* kNeedsExtension =
    "HEIC needs the HEVC Video Extensions from the Microsoft Store (Windows has no HEVC decoder without them)";

template <class T>
struct Com {
    T* p = nullptr;
    ~Com() {
        if (p) p->Release();
    }
    T** put() { return &p; }
    T* operator->() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

// COM for this thread, for the call: an apartment the thread already has
// (either kind) serves as well.
struct ComScope {
    HRESULT hr;
    ComScope() : hr(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
    ~ComScope() {
        if (SUCCEEDED(hr)) CoUninitialize();
    }
    bool usable() const { return SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE; }
};

std::string hr_text(const char* what, HRESULT hr) {
    char buf[160];
    std::snprintf(buf, sizeof buf, "HEIC: %s failed (HRESULT 0x%08lX)", what, static_cast<unsigned long>(hr));
    return buf;
}

bool probe_hevc_mft() {
    MFT_REGISTER_TYPE_INFO in{MFMediaType_Video, MFVideoFormat_HEVC};
    IMFActivate** list = nullptr;
    UINT32 count = 0;
    const HRESULT hr = MFTEnumEx(MFT_CATEGORY_VIDEO_DECODER,
                                 MFT_ENUM_FLAG_SYNCMFT | MFT_ENUM_FLAG_ASYNCMFT | MFT_ENUM_FLAG_HARDWARE |
                                     MFT_ENUM_FLAG_SORTANDFILTER,
                                 &in, nullptr, &list, &count);
    if (FAILED(hr)) return false;
    for (UINT32 i = 0; i < count; ++i) list[i]->Release();
    CoTaskMemFree(list);
    return count > 0;
}

struct Probe {
    bool available = false;
    std::string why;
};

const Probe& probe() {
    static Probe p;
    static std::once_flag once;
    std::call_once(once, [] {
        ComScope com;
        if (!com.usable()) {
            p.why = "HEIC: COM could not be initialised";
            return;
        }
        Com<IWICBitmapDecoder> dec;
        if (FAILED(CoCreateInstance(CLSID_WICHeifDecoder, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(dec.put())))) {
            p.why = "HEIC needs Windows' HEIF Image Extensions (and the HEVC Video Extensions) from the Microsoft Store";
            return;
        }
        if (!probe_hevc_mft()) {
            p.why = kNeedsExtension;
            return;
        }
        p.available = true;
    });
    return p;
}

}  // namespace

bool os_heif_available(std::string* why) {
    const Probe& p = probe();
    if (!p.available && why) *why = p.why;
    return p.available;
}

bool os_heif_decode(const uint8_t* data, std::size_t size, Image& out, bool& oriented, std::string* why) {
    out = Image{};
    oriented = false;
    if (!os_heif_available(why)) return false;
    if (size > 0xFFFFFFFFu) {
        if (why) *why = "HEIC: the file is too large";
        return false;
    }
    ComScope com;
    if (!com.usable()) {
        if (why) *why = "HEIC: COM could not be initialised";
        return false;
    }
    Com<IWICImagingFactory> factory;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(factory.put()));
    if (FAILED(hr)) {
        if (why) *why = hr_text("creating the WIC factory", hr);
        return false;
    }
    Com<IWICStream> stream;
    hr = factory->CreateStream(stream.put());
    if (SUCCEEDED(hr)) hr = stream->InitializeFromMemory(const_cast<BYTE*>(data), DWORD(size));
    if (FAILED(hr)) {
        if (why) *why = hr_text("opening the bytes", hr);
        return false;
    }
    Com<IWICBitmapDecoder> dec;
    hr = CoCreateInstance(CLSID_WICHeifDecoder, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(dec.put()));
    if (SUCCEEDED(hr)) hr = dec->Initialize(stream.p, WICDecodeMetadataCacheOnDemand);
    if (FAILED(hr)) {
        if (why) *why = hr_text("reading the HEIF container", hr);
        return false;
    }
    Com<IWICBitmapFrameDecode> frame;
    hr = dec->GetFrame(0, frame.put());
    if (FAILED(hr)) {
        if (why) *why = hr_text("reading the primary image", hr);
        return false;
    }
    Com<IWICFormatConverter> conv;
    hr = factory->CreateFormatConverter(conv.put());
    if (SUCCEEDED(hr))
        hr = conv->Initialize(frame.p, GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone, nullptr, 0.0,
                              WICBitmapPaletteTypeCustom);
    UINT w = 0, h = 0;
    if (SUCCEEDED(hr)) hr = conv->GetSize(&w, &h);
    if (FAILED(hr) || w == 0 || h == 0) {
        if (why) *why = hr_text("converting the image", hr);
        return false;
    }
    if (uint64_t(w) * h > (uint64_t(1) << 28)) {
        if (why) *why = "HEIC: the image is too large to decode";
        return false;
    }
    out.width = int(w);
    out.height = int(h);
    out.channels = 4;
    out.pixels.resize(std::size_t(w) * h * 4);
    hr = conv->CopyPixels(nullptr, w * 4, UINT(out.pixels.size()), out.pixels.data());
    if (FAILED(hr)) {
        out = Image{};
        // Decoding is where a missing HEVC decoder shows.
        if (why) *why = hr_text("decoding the HEVC image (are the HEVC Video Extensions installed?)", hr);
        return false;
    }
    // WIC's HEIF decoder applies irot / imir itself (seen: a quarter-turned
    // 64 x 48 comes back 48 x 64); the caller turns it back to as stored.
    oriented = true;
    return true;
}

}  // namespace broimage::detail
