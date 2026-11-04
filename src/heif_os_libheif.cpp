// HEIC through the system's libheif (heif_os.h), on Linux and the other
// Unix systems that are not Apple's. libheif is never linked or shipped: it
// is opened at run time with dlopen (libheif.so.1, the soname of every 1.x
// release), so a machine without it reads HEIC's container (size, Exif,
// orientation) and says what to install, and one with it decodes. libheif
// brings its HEVC decoder with it (libde265, built in or a plugin that
// heif_init loads), so availability is the library plus a decoder for HEVC,
// probed once.
//
// BROIMAGE_LIBHEIF names the library to open instead (a path or another
// soname); a name that does not open is the machine without libheif.
//
// Only the stable C API is used, declared here rather than from libheif's
// headers (which a build machine need not have): the context, the primary
// handle, heif_decode_image to interleaved RGBA (8-bit, whatever the coded
// depth), its plane. libheif applies clap, irot and imir; the caller turns
// irot / imir back. The file is untrusted: the context's security limits
// are tightened to broimage's own canvas bound before it reads a byte.

#include "heif_os.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace broimage::detail {

namespace {

// ---- The libheif C API used here (libheif/heif.h, 1.x) -----------------------

struct heif_context;
struct heif_image_handle;
struct heif_image;

struct heif_error {
    int code;  // heif_error_code; 0 is heif_error_Ok
    int subcode;
    const char* message;
};

// heif_security_limits, as far as version 2 (libheif 1.20); `version` says
// how much of it the library has.
struct heif_security_limits {
    uint8_t version;
    uint64_t max_image_size_pixels;
    uint64_t max_number_of_tiles;
    uint32_t max_bayer_pattern_pixels;
    uint32_t max_items;
    uint32_t max_color_profile_size;
    uint64_t max_memory_block_size;
    uint32_t max_components;
    uint32_t max_iloc_extents_per_item;
    uint32_t max_size_entity_group;
    uint32_t max_children_per_box;
    uint64_t max_total_memory;  // version 2
};

constexpr int kHeifCompressionHevc = 1;       // heif_compression_HEVC
constexpr int kHeifColorspaceRgb = 1;         // heif_colorspace_RGB
constexpr int kHeifChromaInterleavedRgba = 11;  // heif_chroma_interleaved_RGBA
constexpr int kHeifChannelInterleaved = 10;   // heif_channel_interleaved

// broimage's canvas bound (decode_heif.cpp): 2^28 pixels, 2^16 a side.
constexpr uint64_t kMaxPixels = uint64_t(1) << 28;
constexpr int kMaxSide = 1 << 16;
constexpr uint64_t kMaxMemory = uint64_t(2) << 30;

struct Lib {
    void* handle = nullptr;
    heif_error (*init)(void*) = nullptr;  // 1.13+
    const char* (*get_version)() = nullptr;
    int (*have_decoder_for_format)(int) = nullptr;
    heif_context* (*context_alloc)() = nullptr;
    void (*context_free)(heif_context*) = nullptr;
    heif_error (*read_from_memory_without_copy)(heif_context*, const void*, std::size_t, const void*) = nullptr;
    heif_error (*get_primary_image_handle)(heif_context*, heif_image_handle**) = nullptr;
    void (*image_handle_release)(const heif_image_handle*) = nullptr;
    int (*image_handle_get_width)(const heif_image_handle*) = nullptr;
    int (*image_handle_get_height)(const heif_image_handle*) = nullptr;
    heif_error (*decode_image)(const heif_image_handle*, heif_image**, int, int, const void*) = nullptr;
    void (*image_release)(const heif_image*) = nullptr;
    int (*image_get_width)(const heif_image*, int) = nullptr;
    int (*image_get_height)(const heif_image*, int) = nullptr;
    const uint8_t* (*image_get_plane_readonly)(const heif_image*, int, int*) = nullptr;
    int (*image_is_premultiplied_alpha)(heif_image*) = nullptr;                      // 1.12+
    heif_security_limits* (*context_get_security_limits)(const heif_context*) = nullptr;  // 1.19+
    void (*context_set_maximum_image_size_limit)(heif_context*, int) = nullptr;      // before 1.19
};

struct Probe {
    Lib lib;
    bool available = false;
    std::string why;
};

const char* kInstall = "install libheif from the system's packages (libheif on Arch and Fedora; libheif1 and "
                       "libheif-plugin-libde265 on Debian and Ubuntu)";

template <class F>
bool sym(void* h, const char* name, F& out) {
    out = reinterpret_cast<F>(dlsym(h, name));
    return out != nullptr;
}

const Probe& probe() {
    static Probe p;
    static std::once_flag once;
    std::call_once(once, [] {
        const char* env = std::getenv("BROIMAGE_LIBHEIF");
        const char* name = env && *env ? env : "libheif.so.1";
        void* h = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            p.why = std::string("HEIC needs libheif (") + name + "), which is not installed: " + kInstall;
            return;
        }
        Lib& l = p.lib;
        l.handle = h;
        const bool core =
            sym(h, "heif_get_version", l.get_version) &&
            sym(h, "heif_have_decoder_for_format", l.have_decoder_for_format) &&
            sym(h, "heif_context_alloc", l.context_alloc) && sym(h, "heif_context_free", l.context_free) &&
            sym(h, "heif_context_read_from_memory_without_copy", l.read_from_memory_without_copy) &&
            sym(h, "heif_context_get_primary_image_handle", l.get_primary_image_handle) &&
            sym(h, "heif_image_handle_release", l.image_handle_release) &&
            sym(h, "heif_image_handle_get_width", l.image_handle_get_width) &&
            sym(h, "heif_image_handle_get_height", l.image_handle_get_height) &&
            sym(h, "heif_decode_image", l.decode_image) && sym(h, "heif_image_release", l.image_release) &&
            sym(h, "heif_image_get_width", l.image_get_width) &&
            sym(h, "heif_image_get_height", l.image_get_height) &&
            sym(h, "heif_image_get_plane_readonly", l.image_get_plane_readonly);
        if (!core) {
            p.why = std::string("HEIC: ") + name + " is not a libheif broimage can use (a symbol is missing): " +
                    kInstall;
            return;
        }
        sym(h, "heif_init", l.init);
        sym(h, "heif_image_is_premultiplied_alpha", l.image_is_premultiplied_alpha);
        sym(h, "heif_context_get_security_limits", l.context_get_security_limits);
        sym(h, "heif_context_set_maximum_image_size_limit", l.context_set_maximum_image_size_limit);
        if (!l.context_get_security_limits && !l.context_set_maximum_image_size_limit) {
            p.why = std::string("HEIC: libheif ") + l.get_version() + " sets no image size limit: " + kInstall;
            return;
        }
        // Loads the decoder plugins (libde265 is one on some distributions).
        // Never matched by heif_deinit: the library stays for the process.
        if (l.init) l.init(nullptr);
        if (!l.have_decoder_for_format(kHeifCompressionHevc)) {
            p.why = std::string("HEIC: libheif ") + l.get_version() +
                    " has no HEVC decoder: install its libde265 plugin (libheif-plugin-libde265)";
            return;
        }
        p.available = true;
    });
    return p;
}

std::string err_text(const char* what, const heif_error& e) {
    std::string s = std::string("HEIC: ") + what;
    if (e.message && *e.message) s += std::string(" (libheif: ") + e.message + ")";
    return s;
}

// The decode's owned libheif objects, released on every return.
struct Session {
    const Lib& l;
    heif_context* ctx = nullptr;
    heif_image_handle* handle = nullptr;
    heif_image* img = nullptr;
    explicit Session(const Lib& lib) : l(lib) {}
    ~Session() {
        if (img) l.image_release(img);
        if (handle) l.image_handle_release(handle);
        if (ctx) l.context_free(ctx);
    }
};

void limit(const Lib& l, heif_context* ctx) {
    if (l.context_get_security_limits) {
        if (heif_security_limits* s = l.context_get_security_limits(ctx); s && s->version >= 1) {
            if (s->max_image_size_pixels == 0 || s->max_image_size_pixels > kMaxPixels)
                s->max_image_size_pixels = kMaxPixels;
            if (s->version >= 2 && (s->max_total_memory == 0 || s->max_total_memory > kMaxMemory))
                s->max_total_memory = kMaxMemory;
            return;
        }
    }
    // The older call bounds the area at its argument squared: 16384^2 = 2^28.
    if (l.context_set_maximum_image_size_limit) l.context_set_maximum_image_size_limit(ctx, 16384);
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
    const Lib& l = probe().lib;
    Session s(l);
    s.ctx = l.context_alloc();
    if (!s.ctx) {
        if (why) *why = "HEIC: out of memory";
        return false;
    }
    limit(l, s.ctx);
    heif_error e = l.read_from_memory_without_copy(s.ctx, data, size, nullptr);
    if (e.code != 0) {
        if (why) *why = err_text("libheif could not read the file", e);
        return false;
    }
    e = l.get_primary_image_handle(s.ctx, &s.handle);
    if (e.code != 0 || !s.handle) {
        if (why) *why = err_text("the file has no primary image", e);
        return false;
    }
    const int hw = l.image_handle_get_width(s.handle), hh = l.image_handle_get_height(s.handle);
    if (hw <= 0 || hh <= 0 || hw > kMaxSide || hh > kMaxSide || uint64_t(hw) * uint64_t(hh) > kMaxPixels) {
        if (why) *why = "HEIC: the image has no size or is too large to decode";
        return false;
    }
    // Default options: clap, irot and imir applied; alpha (when there is
    // one) in the fourth channel, else opaque; 10/12-bit scaled to 8.
    e = l.decode_image(s.handle, &s.img, kHeifColorspaceRgb, kHeifChromaInterleavedRgba, nullptr);
    if (e.code != 0 || !s.img) {
        if (why) *why = err_text("decoding the HEVC image failed", e);
        return false;
    }
    const int w = l.image_get_width(s.img, kHeifChannelInterleaved);
    const int h = l.image_get_height(s.img, kHeifChannelInterleaved);
    int stride = 0;
    const uint8_t* px = l.image_get_plane_readonly(s.img, kHeifChannelInterleaved, &stride);
    if (!px || w <= 0 || h <= 0 || w > kMaxSide || h > kMaxSide || uint64_t(w) * uint64_t(h) > kMaxPixels ||
        stride < w * 4) {
        if (why) *why = "HEIC: libheif returned an image broimage cannot read";
        return false;
    }
    out.width = w;
    out.height = h;
    out.channels = 4;
    out.pixels.resize(std::size_t(w) * std::size_t(h) * 4);
    for (int y = 0; y < h; ++y)
        std::memcpy(out.pixels.data() + std::size_t(y) * std::size_t(w) * 4, px + std::size_t(y) * std::size_t(stride),
                    std::size_t(w) * 4);
    if (l.image_is_premultiplied_alpha && l.image_is_premultiplied_alpha(s.img)) {
        for (std::size_t i = 0; i < out.pixels.size(); i += 4) {
            const unsigned a = out.pixels[i + 3];
            if (a == 0 || a == 255) continue;
            for (int k = 0; k < 3; ++k)
                out.pixels[i + std::size_t(k)] =
                    uint8_t(std::min(255u, (out.pixels[i + std::size_t(k)] * 255u + a / 2) / a));
        }
    }
    oriented = true;
    return true;
}

}  // namespace broimage::detail
