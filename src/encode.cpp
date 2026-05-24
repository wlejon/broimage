#include "broimage/encode.h"

#include <stb_image_write.h>

#include <cstdint>
#include <string>
#include <vector>

namespace broimage {

namespace {

struct MemSink {
    std::vector<uint8_t>* out;
};

void mem_write_cb(void* ctx, void* data, int size) {
    auto* sink = static_cast<MemSink*>(ctx);
    const uint8_t* p = static_cast<const uint8_t*>(data);
    sink->out->insert(sink->out->end(), p, p + size);
}

} // namespace

bool encode_png_file(const std::string& path,
                     const uint8_t* pixels, int width, int height, int channels,
                     int stride_bytes) {
    if (stride_bytes <= 0) stride_bytes = width * channels;
    return stbi_write_png(path.c_str(), width, height, channels,
                          pixels, stride_bytes) != 0;
}

bool encode_png_memory(std::vector<uint8_t>& out,
                       const uint8_t* pixels, int width, int height, int channels,
                       int stride_bytes) {
    if (stride_bytes <= 0) stride_bytes = width * channels;
    out.clear();
    MemSink sink{ &out };
    return stbi_write_png_to_func(&mem_write_cb, &sink,
                                  width, height, channels, pixels,
                                  stride_bytes) != 0;
}

bool encode_jpeg_file(const std::string& path,
                      const uint8_t* pixels, int width, int height, int channels,
                      int quality) {
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    return stbi_write_jpg(path.c_str(), width, height, channels,
                          pixels, quality) != 0;
}

bool encode_jpeg_memory(std::vector<uint8_t>& out,
                        const uint8_t* pixels, int width, int height, int channels,
                        int quality) {
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    out.clear();
    MemSink sink{ &out };
    return stbi_write_jpg_to_func(&mem_write_cb, &sink,
                                  width, height, channels, pixels,
                                  quality) != 0;
}

} // namespace broimage
