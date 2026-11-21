#include "host_image_internal.h"

#if BROIMAGE_HAS_BROKERNEL
#include <broimage/kernel_filter.h>

#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>
#endif

namespace broimage::api {

#if BROIMAGE_HAS_BROKERNEL
namespace {

bool parseGpuFlag(Value opts) {
    if (!ev::isObject(opts)) return false;
    Value g = ev::getProperty(opts, "gpu");
    return ev::isBool(g) && ev::toBool(g);
}

bool parseDims(Value wV, Value hV, const char* who, int& outW, int& outH) {
    if (!wV.isNumber() || !hV.isNumber()) {
        ev::throwTypeError(std::string(who) + ": width and height must be numbers");
        return false;
    }
    const double wd = wV.asNumber();
    const double hd = hV.asNumber();
    if (wd < 0.0 || hd < 0.0 || wd > 65536.0 || hd > 65536.0) {
        ev::throwRangeError(std::string(who) + ": invalid image dimensions");
        return false;
    }
    outW = static_cast<int>(wd);
    outH = static_cast<int>(hd);
    return true;
}

Value imageCompileFilter(Value, std::span<const Value> args) {
    if (args.empty() || !ev::isString(args[0])) {
        return ev::throwTypeError("bro.image.compileFilter(bkSource, options?): expects kernel source string");
    }
    ev::Persistent srcVal(args[0]);
    ev::Persistent optsVal(args.size() >= 2 ? args[1] : ev::undefined());

    std::string entry = "k_filter";
    if (ev::isObject(optsVal.get())) {
        Value entryV = ev::getProperty(optsVal.get(), "entry");
        if (ev::isString(entryV)) entry = ev::toUtf8(entryV);
        Value entryNameV = ev::getProperty(optsVal.get(), "entryName");
        if (ev::isString(entryNameV)) entry = ev::toUtf8(entryNameV);
    }

    const std::string src = ev::toUtf8(srcVal.get());
    auto filter = std::make_shared<broimage::KernelFilter>();
    std::string err;
    if (!filter->compile(src, entry, &err)) {
        return ev::throwError("bro.image.compileFilter: " + err);
    }

    ObjectBuilder obj;
    obj.set("entry", entry);

    obj.def("applyRgba8", 4, [filter](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) {
            return ev::throwTypeError("ImageFilter.applyRgba8(srcRgbaU8, width, height, opts?)");
        }
        TypedArrayView srcView;
        if (!unpackTypedArray(a[0], "srcRgbaU8", &srcView)) return ev::undefined();
        int w = 0, h = 0;
        if (!parseDims(a[1], a[2], "ImageFilter.applyRgba8", w, h)) return ev::undefined();
        if (!requireImage(srcView, w, h, 4, 0, 1, "ImageFilter.applyRgba8")) return ev::undefined();
        ev::Persistent optHold(a.size() >= 4 ? a[3] : ev::undefined());
        const bool useGpu = parseGpuFlag(optHold.get());
        if (!resolveViews({&srcView})) return ev::undefined();

        const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h) * 4u;
        broimage::Image srcImg;
        srcImg.width = w;
        srcImg.height = h;
        srcImg.channels = 4;
        srcImg.pixels.resize(total);
        if (total > 0) std::memcpy(srcImg.pixels.data(), srcView.data, total);

        broimage::Image dstImg;
        std::string runErr;
        if (!filter->run_rgba8(srcImg, dstImg, useGpu, &runErr)) {
            return ev::throwError("ImageFilter.applyRgba8: " + runErr);
        }
        return typedArrayFrom(ev::elements::Uint8, dstImg.pixels.data(), dstImg.pixels.size(),
                              static_cast<uint32_t>(dstImg.pixels.size()));
    });

    obj.def("applyRgbaF32", 4, [filter](Value, std::span<const Value> a) -> Value {
        if (a.size() < 3) {
            return ev::throwTypeError("ImageFilter.applyRgbaF32(srcRgbaF32, width, height, opts?)");
        }
        TypedArrayView srcView;
        if (!unpackTypedArray(a[0], "srcRgbaF32", &srcView)) return ev::undefined();
        if (!requireFloat32(srcView, "ImageFilter.applyRgbaF32")) return ev::undefined();
        int w = 0, h = 0;
        if (!parseDims(a[1], a[2], "ImageFilter.applyRgbaF32", w, h)) return ev::undefined();
        if (!requireImage(srcView, w, h, 4, 0, sizeof(float), "ImageFilter.applyRgbaF32")) {
            return ev::undefined();
        }
        ev::Persistent optHold(a.size() >= 4 ? a[3] : ev::undefined());
        const bool useGpu = parseGpuFlag(optHold.get());
        if (!resolveViews({&srcView})) return ev::undefined();

        const size_t totalFloats = static_cast<size_t>(w) * static_cast<size_t>(h) * 4u;
        std::vector<float> srcCopy(totalFloats);
        if (totalFloats > 0) {
            std::memcpy(srcCopy.data(), srcView.data, totalFloats * sizeof(float));
        }
        std::vector<float> dst(totalFloats, 0.0f);
        std::string runErr;
        if (!filter->run_rgba_f32(srcCopy.data(), dst.data(),
                                  static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                                  useGpu, &runErr)) {
            return ev::throwError("ImageFilter.applyRgbaF32: " + runErr);
        }
        return typedArrayFrom(ev::elements::Float32, dst.data(), dst.size() * sizeof(float),
                              static_cast<uint32_t>(dst.size()));
    });

    return obj.build();
}

} // namespace

void installKernelFilterOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("compileFilter", 2, imageCompileFilter);
}
#else
void installKernelFilterOnto(Value) {}
#endif

} // namespace broimage::api
