// HEIC through ImageIO (heif_os.h). macOS decodes HEIF/HEVC itself (since
// 10.13). CGImageSourceCreateImageAtIndex gives the image as stored — the
// orientation is a property beside it — and it is drawn into an sRGB
// bitmap, so a Display P3 photo is converted by ColorSync on the way.

#include "heif_os.h"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>

#include <algorithm>
#include <mutex>
#include <string>

namespace broimage::detail {

namespace {

struct Probe {
    bool available = false;
    std::string why;
};

const Probe& probe() {
    static Probe p;
    static std::once_flag once;
    std::call_once(once, [] {
        CFArrayRef types = CGImageSourceCopyTypeIdentifiers();
        if (types) {
            const CFIndex n = CFArrayGetCount(types);
            for (CFIndex i = 0; i < n && !p.available; ++i) {
                CFStringRef t = static_cast<CFStringRef>(CFArrayGetValueAtIndex(types, i));
                if (CFStringCompare(t, CFSTR("public.heic"), 0) == kCFCompareEqualTo) p.available = true;
            }
            CFRelease(types);
        }
        if (!p.available) p.why = "HEIC: this macOS's ImageIO has no HEIC decoder";
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
    CFDataRef cf = CFDataCreateWithBytesNoCopy(kCFAllocatorDefault, data, CFIndex(size), kCFAllocatorNull);
    if (!cf) {
        if (why) *why = "HEIC: out of memory";
        return false;
    }
    CGImageSourceRef src = CGImageSourceCreateWithData(cf, nullptr);
    CFRelease(cf);
    if (!src) {
        if (why) *why = "HEIC: ImageIO could not read the file";
        return false;
    }
    const size_t index = CGImageSourceGetPrimaryImageIndex(src);
    CGImageRef img = CGImageSourceCreateImageAtIndex(src, index, nullptr);
    CFRelease(src);
    if (!img) {
        if (why) *why = "HEIC: ImageIO could not decode the image";
        return false;
    }
    const size_t w = CGImageGetWidth(img), h = CGImageGetHeight(img);
    if (w == 0 || h == 0 || uint64_t(w) * h > (uint64_t(1) << 28)) {
        CGImageRelease(img);
        if (why) *why = "HEIC: the image has no size or is too large";
        return false;
    }
    out.width = int(w);
    out.height = int(h);
    out.channels = 4;
    out.pixels.assign(w * h * 4, 0);
    CGColorSpaceRef srgb = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    // Core Graphics draws only premultiplied RGBA; it is undone below.
    CGContextRef ctx = CGBitmapContextCreate(out.pixels.data(), w, h, 8, w * 4, srgb,
                                             uint32_t(kCGImageAlphaPremultipliedLast) | uint32_t(kCGBitmapByteOrder32Big));
    CGColorSpaceRelease(srgb);
    if (!ctx) {
        CGImageRelease(img);
        out = Image{};
        if (why) *why = "HEIC: could not make a bitmap to draw into";
        return false;
    }
    CGContextSetBlendMode(ctx, kCGBlendModeCopy);
    CGContextDrawImage(ctx, CGRectMake(0, 0, CGFloat(w), CGFloat(h)), img);
    CGContextRelease(ctx);
    CGImageRelease(img);
    for (std::size_t i = 0; i < out.pixels.size(); i += 4) {
        const unsigned a = out.pixels[i + 3];
        if (a == 0 || a == 255) continue;
        for (int k = 0; k < 3; ++k)
            out.pixels[i + std::size_t(k)] = uint8_t(std::min(255u, (out.pixels[i + std::size_t(k)] * 255u + a / 2) / a));
    }
    return true;
}

}  // namespace broimage::detail
