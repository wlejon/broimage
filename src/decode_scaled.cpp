// Scaled decode and the EXIF thumbnail (decode.h).

#include "broimage/decode.h"

#include "broimage/alpha.h"
#include "broimage/exif.h"
#include "broimage/geometric.h"
#include "broimage/heif.h"

#include "decode_cr3.h"
#include "decode_jpeg.h"
#include "file_bytes.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

namespace broimage {

namespace {

bool transposes(ExifOrientation o) { return int(o) >= 5; }

// The size `options` asks for of an upright w x h image: never larger.
void target_size(int w, int h, const ScaledDecodeOptions& o, int& tw, int& th) {
    const double sx = o.width > 0 ? double(o.width) / w : 0;
    const double sy = o.height > 0 ? double(o.height) / h : 0;
    double s = 1;
    if (sx > 0 && sy > 0) s = o.fit == ScaleFit::Contain ? std::min(sx, sy) : std::max(sx, sy);
    else if (sx > 0) s = sx;
    else if (sy > 0) s = sy;
    if (s >= 1) {
        tw = w;
        th = h;
        return;
    }
    tw = std::clamp(int(std::lround(w * s)), 1, w);
    th = std::clamp(int(std::lround(h * s)), 1, h);
}

bool opaque(const Image& img) {
    for (std::size_t i = 3; i < img.pixels.size(); i += 4)
        if (img.pixels[i] != 255) return false;
    return true;
}

// Area-average `img` down to w x h (alpha-aware unless it is opaque).
void resize_to(Image& img, int w, int h) {
    if (img.width == w && img.height == h) return;
    std::vector<uint8_t> dst(std::size_t(w) * h * 4);
    if (opaque(img))
        resize_hwc_u8(img.pixels.data(), img.width, img.height, 4, dst.data(), w, h, Filter::Area);
    else
        resize_rgba8_alpha(img.pixels.data(), img.width, img.height, dst.data(), w, h, Filter::Area);
    img.width = w;
    img.height = h;
    img.pixels = std::move(dst);
}

// Cut the bars a camera pads its thumbnail with to fit 4:3 (or 3:2) off,
// leaving the main image's aspect ratio: `full_w` x `full_h`, as stored.
void crop_to_aspect(Image& img, int full_w, int full_h) {
    if (full_w <= 0 || full_h <= 0 || img.width <= 0 || img.height <= 0) return;
    const double want = double(full_w) / full_h;
    const double have = double(img.width) / img.height;
    if (std::fabs(have - want) <= 0.02 * want) return;
    int x = 0, y = 0, w = img.width, h = img.height;
    if (want > have) {
        h = std::max(1, int(std::lround(img.width / want)));
        y = (img.height - h) / 2;
    } else {
        w = std::max(1, int(std::lround(img.height * want)));
        x = (img.width - w) / 2;
    }
    std::vector<uint8_t> out(std::size_t(w) * h * 4);
    for (int r = 0; r < h; ++r)
        std::memcpy(out.data() + std::size_t(r) * w * 4,
                    img.pixels.data() + (std::size_t(y + r) * img.width + x) * 4, std::size_t(w) * 4);
    img.width = w;
    img.height = h;
    img.pixels = std::move(out);
}

// The EXIF thumbnail, cropped and upright; the main frame's stored size is
// passed in when the caller already read it.
bool exif_thumbnail(const uint8_t* data, std::size_t size, int full_w, int full_h, ExifOrientation orient,
                    Image& out, std::string* why) {
    const uint8_t* jpeg = nullptr;
    std::size_t n = 0;
    if (!find_exif_thumbnail(data, size, jpeg, n)) {
        if (why) *why = "no EXIF thumbnail";
        return false;
    }
    if (!decode_memory(jpeg, n, out, why)) return false;
    crop_to_aspect(out, full_w, full_h);
    apply_exif_orientation(out, orient);
    return true;
}

// A JPEG's EXIF thumbnail when `options` lets it stand in and it is at least
// the target size; finished (resized) into `out`. Reads only the head of the
// file (the APP segments and the frame header), so a caller can try it on
// the first bytes of a file before reading the rest.
bool from_exif_thumbnail(const uint8_t* data, std::size_t size, const ScaledDecodeOptions& options, Image& out,
                         ScaledDecodeInfo& info) {
    int sw = 0, sh = 0, sc = 0;
    if (!options.embedded_thumbnail || !detail::jpeg_probe(data, size, sw, sh, sc)) return false;
    const ExifOrientation orient = read_exif_orientation(data, size);
    const int uw = transposes(orient) ? sh : sw, uh = transposes(orient) ? sw : sh;
    int tw = 0, th = 0;
    target_size(uw, uh, options, tw, th);
    if (tw >= uw && th >= uh) return false;  // the image's own size: not a thumbnail's
    Image img;
    if (!exif_thumbnail(data, size, sw, sh, orient, img, nullptr) || img.width < tw || img.height < th) return false;
    info = ScaledDecodeInfo{};
    info.source_width = uw;
    info.source_height = uh;
    info.decoded_width = img.width;
    info.decoded_height = img.height;
    info.path = ScaledDecodePath::ExifThumbnail;
    resize_to(img, tw, th);
    out = std::move(img);
    return true;
}

} // namespace

bool decode_exif_thumbnail(const uint8_t* data, std::size_t size, Image& out, std::string* error) {
    out = Image{};
    // A CR3's counterpart is its THMB box.
    if (detail::Cr3Parts cr3; detail::cr3_parts(data, size, cr3)) {
        if (!cr3.thumbnail || !decode_memory(cr3.thumbnail.data, cr3.thumbnail.size, out, error)) {
            if (error && !cr3.thumbnail) *error = "no CR3 thumbnail";
            out = Image{};
            out.channels = 0;
            return false;
        }
        crop_to_aspect(out, cr3.full_width, cr3.full_height);
        apply_exif_orientation(out, read_exif_orientation(data, size));
        return true;
    }
    int w = 0, h = 0, c = 0;
    if (!detail::jpeg_probe(data, size, w, h, c)) {
        if (error) *error = "not a JPEG";
        out.channels = 0;
        return false;
    }
    if (!exif_thumbnail(data, size, w, h, read_exif_orientation(data, size), out, error)) {
        out = Image{};
        out.channels = 0;
        return false;
    }
    return true;
}

bool decode_file_exif_thumbnail(const std::string& path, Image& out, std::string* error) {
    std::vector<uint8_t> head;
    if (!detail::read_file_bytes(path, head, std::size_t(1) << 20)) {
        if (error) *error = "can't fopen";
        out = Image{};
        out.channels = 0;
        return false;
    }
    return decode_exif_thumbnail(head.data(), head.size(), out, error);
}

bool decode_memory_scaled(const uint8_t* data, std::size_t size, const ScaledDecodeOptions& options,
                          Image& out, std::string* error, ScaledDecodeInfo* info) {
    ScaledDecodeInfo local;
    ScaledDecodeInfo& in = info ? *info : local;
    in = ScaledDecodeInfo{};
    out = Image{};

    const ExifOrientation orient = read_exif_orientation(data, size);
    int sw = 0, sh = 0, sc = 0;  // stored size
    const bool jpeg = detail::jpeg_probe(data, size, sw, sh, sc);
    if (!jpeg && !probe_dimensions_memory(data, size, &sw, &sh, &sc)) sw = sh = 0;

    auto finish = [&](Image& img, ScaledDecodePath path, int scale) {
        in.path = path;
        in.dct_scale = scale;
        in.decoded_width = img.width;
        in.decoded_height = img.height;
        if (path == ScaledDecodePath::Full || in.source_width <= 0) {  // the decode says exactly
            in.source_width = img.width;
            in.source_height = img.height;
        }
        int tw = 0, th = 0;
        target_size(in.source_width, in.source_height, options, tw, th);
        resize_to(img, std::min(tw, img.width), std::min(th, img.height));
        out = std::move(img);
        return true;
    };

    int tw = 0, th = 0;  // the target, upright
    if (sw > 0 && sh > 0) {
        in.source_width = transposes(orient) ? sh : sw;
        in.source_height = transposes(orient) ? sw : sh;
        target_size(in.source_width, in.source_height, options, tw, th);
    }

    // A CR3: the smallest embedded JPEG that covers the target, scaled as any
    // JPEG is (the thumbnail cropped of its bars first), then upright.
    if (detail::Cr3Parts cr3; tw > 0 && detail::cr3_parts(data, size, cr3)) {
        const detail::Cr3Span candidates[] = {options.embedded_thumbnail ? cr3.thumbnail : detail::Cr3Span{},
                                              cr3.preview, cr3.full};
        const int rw = transposes(orient) ? th : tw, rh = transposes(orient) ? tw : th;  // as stored
        for (const detail::Cr3Span& c : candidates) {
            int cw = 0, chh = 0, cc = 0;
            if (!c || !detail::jpeg_probe(c.data, c.size, cw, chh, cc)) continue;
            const bool last = c.data == cr3.full.data;
            if (!last && (cw < rw || chh < rh)) continue;  // too small, and something bigger may follow
            ScaledDecodeOptions inner;
            if (c.data != cr3.thumbnail.data) {  // the thumbnail is cropped whole, then resized
                inner.width = rw;
                inner.height = rh;
                inner.fit = ScaleFit::Cover;
            }
            inner.embedded_thumbnail = false;
            ScaledDecodeInfo ii;
            Image img;
            if (!decode_memory_scaled(c.data, c.size, inner, img, nullptr, &ii)) continue;
            crop_to_aspect(img, sw, sh);
            if (!last && (img.width < rw || img.height < rh)) continue;
            apply_exif_orientation(img, orient);
            return finish(img, ScaledDecodePath::RawPreview, ii.dct_scale);
        }
    }

    if (jpeg && tw > 0) {
        if (from_exif_thumbnail(data, size, options, out, in)) return true;
        Image img;
        if (detail::jpeg_turbo()) {
            // The target as stored, and the smallest M/8 that covers it.
            const int rw = transposes(orient) ? th : tw;
            const int rh = transposes(orient) ? tw : th;
            int scale = 1;
            while (scale < 8 && (detail::jpeg_scaled_size(sw, scale) < rw || detail::jpeg_scaled_size(sh, scale) < rh))
                ++scale;
            // On a failure the whole decode below tries again, stb_image included.
            if (detail::jpeg_decode(data, size, scale, img, nullptr)) {
                apply_exif_orientation(img, orient);
                return finish(img, scale == 8 ? ScaledDecodePath::Full : ScaledDecodePath::Dct, scale);
            }
        }
    }

    if (options.embedded_thumbnail && tw > 0 && (tw < in.source_width || th < in.source_height) &&
        is_heif(data, size)) {
        Image img;
        if (decode_heif_thumbnail(data, size, std::min(tw, th), img) && img.width >= tw && img.height >= th)
            return finish(img, ScaledDecodePath::HeifThumbnail, 8);
    }

    Image img;
    if (!decode_memory_oriented(data, size, img, error)) {
        out = std::move(img);
        return false;
    }
    return finish(img, ScaledDecodePath::Full, 8);
}

bool decode_file_scaled(const std::string& path, const ScaledDecodeOptions& options, Image& out,
                        std::string* error, ScaledDecodeInfo* info) {
    std::vector<uint8_t> bytes;
    // A JPEG's EXIF block and frame header sit in its first few dozen KB:
    // when its thumbnail will do, the rest of the file (8-10 MB off a camera
    // card) is never read.
    if (options.embedded_thumbnail && detail::read_file_bytes(path, bytes, std::size_t(256) * 1024) &&
        detail::is_jpeg(bytes.data(), bytes.size())) {
        ScaledDecodeInfo local;
        if (from_exif_thumbnail(bytes.data(), bytes.size(), options, out, info ? *info : local)) return true;
    }
    // A CR3's THMB and PRVW are in its first ~0.5 MB, ahead of 20-40 MB of
    // RAW data: when one of them covers the target, that is all that is read.
    if (detail::read_file_bytes(path, bytes, std::size_t(1) << 20) && detail::cr3_sniff(bytes.data(), bytes.size())) {
        ScaledDecodeInfo head;
        Image img;
        if (decode_memory_scaled(bytes.data(), bytes.size(), options, img, nullptr, &head) &&
            head.path == ScaledDecodePath::RawPreview) {
            out = std::move(img);
            if (info) *info = head;
            return true;
        }
    }
    if (!detail::read_file_bytes(path, bytes)) {
        if (error) *error = "can't fopen";
        out = Image{};
        out.channels = 0;
        if (info) *info = ScaledDecodeInfo{};
        return false;
    }
    return decode_memory_scaled(bytes.data(), bytes.size(), options, out, error, info);
}

} // namespace broimage
