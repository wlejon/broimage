#include "broimage/decode.h"

#include "broimage/buffer.h"

#include <stb_image.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace broimage {

namespace {

void set_fallback(Image& out) {
    out.width = 1;
    out.height = 1;
    out.channels = 4;
    out.pixels = { 255, 255, 255, 255 };
}

} // namespace

bool decode_file(const std::string& path, Image& out, std::string* error) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_load failed";
        set_fallback(out);
        return false;
    }
    out.width = w;
    out.height = h;
    out.channels = 4;
    out.pixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

bool decode_memory(const uint8_t* data, std::size_t size, Image& out,
                   std::string* error) {
    int w = 0, h = 0, channels = 0;
    unsigned char* pixels = stbi_load_from_memory(data, static_cast<int>(size),
                                                  &w, &h, &channels, 4);
    if (!pixels) {
        if (error) *error = stbi_failure_reason() ? stbi_failure_reason() : "stbi_load_from_memory failed";
        set_fallback(out);
        return false;
    }
    out.width = w;
    out.height = h;
    out.channels = 4;
    out.pixels.assign(pixels, pixels + static_cast<std::size_t>(w) * h * 4);
    stbi_image_free(pixels);
    return true;
}

} // namespace broimage
