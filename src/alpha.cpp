#include "broimage/alpha.h"

#include "broimage/geometric.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace broimage {

void premultiply_alpha_rgba8(const uint8_t* src, uint8_t* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        const uint8_t a = src[i * 4 + 3];
        // (channel * a + 127) / 255 is the standard rounded fixed-point form.
        dst[i * 4 + 0] = static_cast<uint8_t>((src[i * 4 + 0] * a + 127) / 255);
        dst[i * 4 + 1] = static_cast<uint8_t>((src[i * 4 + 1] * a + 127) / 255);
        dst[i * 4 + 2] = static_cast<uint8_t>((src[i * 4 + 2] * a + 127) / 255);
        dst[i * 4 + 3] = a;
    }
}

void unpremultiply_alpha_rgba8(const uint8_t* src, uint8_t* dst, int pixel_count) {
    for (int i = 0; i < pixel_count; ++i) {
        const uint8_t a = src[i * 4 + 3];
        if (a == 0) {
            // RGB is undefined in a fully transparent premultiplied pixel; the
            // canonical un-premul leaves it at 0 rather than dividing by 0.
            dst[i * 4 + 0] = 0;
            dst[i * 4 + 1] = 0;
            dst[i * 4 + 2] = 0;
            dst[i * 4 + 3] = 0;
            continue;
        }
        const int r = (src[i * 4 + 0] * 255 + a / 2) / a;
        const int g = (src[i * 4 + 1] * 255 + a / 2) / a;
        const int b = (src[i * 4 + 2] * 255 + a / 2) / a;
        dst[i * 4 + 0] = static_cast<uint8_t>(std::min(255, r));
        dst[i * 4 + 1] = static_cast<uint8_t>(std::min(255, g));
        dst[i * 4 + 2] = static_cast<uint8_t>(std::min(255, b));
        dst[i * 4 + 3] = a;
    }
}

void resize_rgba8_alpha(const uint8_t* src, int src_w, int src_h,
                        uint8_t*       dst, int dst_w, int dst_h,
                        Filter filter) {
    const std::size_t src_n = static_cast<std::size_t>(src_w) * src_h;
    const std::size_t dst_n = static_cast<std::size_t>(dst_w) * dst_h;
    std::vector<uint8_t> prem_src(src_n * 4);
    premultiply_alpha_rgba8(src, prem_src.data(), static_cast<int>(src_n));
    std::vector<uint8_t> prem_dst(dst_n * 4);
    resize_hwc_u8(prem_src.data(), src_w, src_h, 4,
                  prem_dst.data(), dst_w, dst_h, filter);
    unpremultiply_alpha_rgba8(prem_dst.data(), dst, static_cast<int>(dst_n));
}

void letterbox_rgba8_alpha(const uint8_t* src, int src_w, int src_h,
                           uint8_t*       dst, int dst_w, int dst_h,
                           uint8_t pad_r, uint8_t pad_g, uint8_t pad_b, uint8_t pad_a,
                           Filter filter,
                           int* out_x, int* out_y, int* out_w, int* out_h) {
    const float sx = static_cast<float>(dst_w) / static_cast<float>(src_w);
    const float sy = static_cast<float>(dst_h) / static_cast<float>(src_h);
    const float s = std::min(sx, sy);
    int new_w = std::max(1, static_cast<int>(s * src_w + 0.5f));
    int new_h = std::max(1, static_cast<int>(s * src_h + 0.5f));
    if (new_w > dst_w) new_w = dst_w;
    if (new_h > dst_h) new_h = dst_h;
    const int off_x = (dst_w - new_w) / 2;
    const int off_y = (dst_h - new_h) / 2;

    std::vector<uint8_t> scratch(static_cast<std::size_t>(new_w) * new_h * 4);
    resize_rgba8_alpha(src, src_w, src_h, scratch.data(), new_w, new_h, filter);
    pad_hwc_u8(scratch.data(), new_w, new_h, 4, dst, dst_w, dst_h,
               off_x, off_y, pad_r, pad_g, pad_b, pad_a);

    if (out_x) *out_x = off_x;
    if (out_y) *out_y = off_y;
    if (out_w) *out_w = new_w;
    if (out_h) *out_h = new_h;
}

} // namespace broimage
