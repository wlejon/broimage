#include "broimage/kernel_filter.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            std::fprintf(stderr, "CHECK FAILED at %s:%d: %s\n", __FILE__, __LINE__, #cond);        \
            std::abort();                                                                          \
        }                                                                                          \
    } while (0)

int main() {
    const char* src_bk = R"bk(
import std::image;

@kernel @grid(2)
fn k_filter(g: grid<2>, src: read_store<vec4<f32>, 2>, dst: write_store<vec4<f32>, 2>) {
    let x = g.x();
    let y = g.y();
    let px = src[x, y];
    let lin = image::srgb_to_linear3(vec3<f32>(px.x, px.y, px.z));
    let tm = image::aces_tonemap(lin);
    let out_rgb = image::linear_to_srgb3(tm);
    dst[x, y] = vec4<f32>(out_rgb.x, out_rgb.y, out_rgb.z, px.w);
}
)bk";

    broimage::KernelFilter kf;
    std::string err;
    CHECK(kf.compile(src_bk, "k_filter", &err));
    CHECK(kf.valid());

    broimage::Image in;
    in.width = 4;
    in.height = 4;
    in.channels = 4;
    in.pixels.assign(4 * 4 * 4, 255);

    broimage::Image out;
    CHECK(kf.run_rgba8(in, out, /*use_gpu=*/false, &err));
    CHECK(out.width == 4 && out.height == 4 && out.channels == 4);
    // ACES tonemap compresses pure white (1, 1, 1) below 255 while preserving alpha=255.
    CHECK(out.pixels[0] >= 180 && out.pixels[0] < 245);
    CHECK(out.pixels[3] == 255);
    return 0;
}
