#include <broimage/version.h>
#include "../src/api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

// assert() is compiled out in the Release configuration ctest runs, which
// left every check below inert; this one survives NDEBUG.
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "CHECK failed: " #cond " (line " << __LINE__ << ")" \
                      << std::endl;                                        \
            std::exit(1);                                                  \
        }                                                                  \
    } while (0)

// tests/test_image_api_restored.cpp — the bro.image kernels the bronze port
// dropped (bro's docs/transition-drift.md row E4).
void broimageTestRestoredSurface();

// tests/test_image_api_paths.cpp — setPathResolver: the host's resolver is
// consulted for every filename the file entry points take.
void broimageTestPathResolver();

// tests/test_image_api_ops.cpp — pixel checks for the ops kernels, each
// called with an options object so GC stress can move its buffers.
void broimageTestOpsSurface();

int main() {
    namespace ev = bronze::embed;
    using namespace bronze::eval;

    std::cout << "Installing image API into Bronze realm..." << std::endl;
    broimage::api::installImage();

    // 1. Verify bro and bro.image mounting
    auto g = ev::globalValue("bro");
    CHECK(g.found);
    CHECK(ev::isObject(g.value));

    // Rooted: each getProperty below may allocate and move it.
    ev::Persistent img(ev::getProperty(g.value, "image"));
    CHECK(ev::isObject(img.get()));

    // Verify codec properties
    const char* codecProps[] = {
        "transcodeKTX2", "encodePng", "encodePngFile", "encodeJpeg", "encodeJpegFile"
    };
    for (const char* prop : codecProps) {
        auto p = ev::getProperty(img.get(), prop);
        CHECK(ev::isFunction(p));
        std::cout << "  Found bro.image." << prop << std::endl;
    }

    // Verify ops properties
    const char* opsProps[] = {
        "gradient", "alloc", "lookup", "reduce", "map", "combine", "stencil", "resample",
        "resize", "crop", "centerCrop", "flipHorizontal", "flipVertical", "rotate90", "pad",
        "rgbaToRgb", "rgbToRgba", "rgbaToGray", "rgbToGray", "srgbToLinear", "linearToSrgb",
        "applyGamma", "rgbToHsv", "hsvToRgb", "rgbToHsl", "hslToRgb",
        "normalize", "u8ToF32", "f32ToU8", "nhwcToNchw", "nchwToNhwc"
    };
    for (const char* prop : opsProps) {
        auto p = ev::getProperty(img.get(), prop);
        CHECK(ev::isFunction(p));
    }
    std::cout << "  Verified all image ops functions are mounted." << std::endl;

    // 2. Test operations via Bronze evalScript
    std::cout << "Testing image operations via Bronze evalScript..." << std::endl;

    // Test Alloc + Reduce
    {
        auto r = evalScript(
            "(function() {"
            "  const buf = bro.image.alloc(2, 2, 1, 'float32');"
            "  buf[0] = 5.0; buf[1] = 1.0; buf[2] = 8.0; buf[3] = 3.0;"
            "  const mm = bro.image.reduce(buf, 'minmax');"
            "  return (mm.min === 1.0 && mm.max === 8.0);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: alloc + reduce [PASS]" << std::endl;
    }

    // Test FlipHorizontal
    {
        auto r = evalScript(
            "(function() {"
            "  const src = new Uint8Array([1, 2, 3, 4]);"
            "  const dst = new Uint8Array(4);"
            "  bro.image.flipHorizontal(dst, src, { w: 2, h: 2, channels: 1 });"
            "  return (dst[0] === 2 && dst[1] === 1 && dst[2] === 4 && dst[3] === 3);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: flipHorizontal [PASS]" << std::endl;
    }

    // Test rgbToRgba
    {
        auto r = evalScript(
            "(function() {"
            "  const rgb = new Uint8Array([10, 20, 30]);"
            "  const rgba = new Uint8Array(4);"
            "  bro.image.rgbToRgba(rgba, rgb, 1, 200);"
            "  return (rgba[0] === 10 && rgba[1] === 20 && rgba[2] === 30 && rgba[3] === 200);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: rgbToRgba [PASS]" << std::endl;
    }

    // Test rgbaToGray
    {
        auto r = evalScript(
            "(function() {"
            "  const rgba = new Uint8Array([255, 255, 255, 255]);"
            "  const gray = new Uint8Array(1);"
            "  bro.image.rgbaToGray(gray, rgba, 1);"
            "  return (gray[0] === 255);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: rgbaToGray [PASS]" << std::endl;
    }

    // Test encodePng
    {
        auto r = evalScript(
            "(function() {"
            "  const pixels = new Uint8Array([255, 0, 0, 255]);"
            "  const png = bro.image.encodePng(pixels, 1, 1, 4);"
            "  return (png instanceof Uint8Array && png.length > 8 &&"
            "          png[0] === 0x89 && png[1] === 0x50 && png[2] === 0x4E && png[3] === 0x47);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: encodePng [PASS]" << std::endl;
    }

    // Test encodeJpeg
    {
        auto r = evalScript(
            "(function() {"
            "  const pixels = new Uint8Array([128, 128, 128]);"
            "  const jpg = bro.image.encodeJpeg(pixels, 1, 1, 3, 90);"
            "  return (jpg instanceof Uint8Array && jpg.length > 4 &&"
            "          jpg[0] === 0xFF && jpg[1] === 0xD8);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: encodeJpeg [PASS]" << std::endl;
    }

    // Test Normalize
    {
        auto r = evalScript(
            "(function() {"
            "  const X = new Float32Array([10.0, 30.0]);"
            "  const Y = new Float32Array(2);"
            "  const mean = new Float32Array([10.0]);"
            "  const std = new Float32Array([5.0]);"
            "  bro.image.normalize(Y, X, mean, std, 1, 1, 1, 2);"
            "  return (Math.abs(Y[0] - 0.0) < 1e-4 && Math.abs(Y[1] - 4.0) < 1e-4);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: normalize [PASS]" << std::endl;
    }

    // Test Resize
    {
        auto r = evalScript(
            "(function() {"
            "  const src = new Uint8Array([100, 100, 100, 100]);"
            "  const dst = new Uint8Array(1);"
            "  bro.image.resize(dst, src, { srcW: 2, srcH: 2, dstW: 1, dstH: 1, channels: 1 });"
            "  return (dst[0] === 100);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: resize [PASS]" << std::endl;
    }

    // Test Gradient
    {
        auto r = evalScript(
            "(function() {"
            "  const grad = bro.image.gradient([[0, 0, 0, 0], [1, 255, 255, 255]], 2);"
            "  return (grad instanceof Uint8Array && grad.length === 8 &&"
            "          grad[0] === 0 && grad[4] === 255);"
            "})()"
        );
        CHECK(!r.thrown);
        CHECK(ev::isBool(r.value) && ev::toBool(r.value));
        std::cout << "  eval: gradient [PASS]" << std::endl;
    }

    broimageTestRestoredSurface();
    broimageTestOpsSurface();
    broimageTestPathResolver();

    std::cout << "All broimage Bronze JavaScript API tests passed successfully!" << std::endl;
    return 0;
}
