#include "broimage/kernel_filter.h"

#include <brokernel/brokernel.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace broimage {

struct KernelFilter::Impl {
    brokernel::CompileResult result;
    const brokernel::CompiledKernel* kernel = nullptr;
};

KernelFilter::KernelFilter() = default;
KernelFilter::~KernelFilter() = default;
KernelFilter::KernelFilter(KernelFilter&&) noexcept = default;
KernelFilter& KernelFilter::operator=(KernelFilter&&) noexcept = default;

bool KernelFilter::compile(std::string_view bk_source, std::string_view entry_name, std::string* err) {
    brokernel::Compiler compiler;
    brokernel::CompileOptions opts;
    opts.entry_name = std::string(entry_name);
    opts.compile_cpu = true;
    opts.emit_spirv = true;

    auto res = compiler.compile_source(bk_source, "image_filter.bk", opts);
    if (!res.ok) {
        if (err) *err = res.error_message();
        impl_.reset();
        return false;
    }
    const brokernel::CompiledKernel* k = res.find_kernel(entry_name);
    if (!k) {
        if (err) *err = "broimage::KernelFilter: entry kernel '" + std::string(entry_name) + "' not found";
        impl_.reset();
        return false;
    }
    if (k->grid_dim != 2) {
        if (err) *err = "broimage::KernelFilter: kernel must be @grid(2)";
        impl_.reset();
        return false;
    }
    size_t non_grid = 0;
    for (const auto& p : k->params) {
        if (!brokernel::is_bound_param(p.role)) ++non_grid;
    }
    if (non_grid != 2) {
        if (err) *err = "broimage::KernelFilter: kernel must take (g, src, dst)";
        impl_.reset();
        return false;
    }

    auto next = std::make_unique<Impl>();
    next->result = std::move(res);
    next->kernel = next->result.find_kernel(entry_name);
    impl_ = std::move(next);
    return true;
}

bool KernelFilter::valid() const noexcept {
    return impl_ && impl_->kernel != nullptr;
}

bool KernelFilter::run_rgba_f32(
    const float* src_rgba,
    float* dst_rgba,
    std::uint32_t width,
    std::uint32_t height,
    bool use_gpu,
    std::string* err) const {
    if (!valid()) {
        if (err) *err = "broimage::KernelFilter: kernel not compiled";
        return false;
    }
    if (width == 0 || height == 0) return true;
    if (!src_rgba || !dst_rgba) {
        if (err) *err = "broimage::KernelFilter: null pixel buffer";
        return false;
    }
    const size_t nbytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4 * sizeof(float);
    const brokernel::KernelArg args[2] = {
        brokernel::KernelArg::buffer(src_rgba, nbytes),
        brokernel::KernelArg::buffer(dst_rgba, nbytes),
    };
    const brokernel::GridShape shape(width, height);
    if (use_gpu && brass::gpu::vulkan_available() && !impl_->kernel->spirv.words.empty()) {
        return impl_->kernel->dispatch_gpu(shape, args, nullptr, err);
    }
    return impl_->kernel->dispatch_cpu(shape, args, nullptr, err);
}

bool KernelFilter::run_rgba8(const Image& src, Image& dst, bool use_gpu, std::string* err) const {
    if (src.width <= 0 || src.height <= 0 || src.channels != 4 || src.pixels.size() < size_t(src.width) * src.height * 4) {
        if (err) *err = "broimage::KernelFilter: run_rgba8 requires valid RGBA8 input image";
        return false;
    }
    const uint32_t w = static_cast<uint32_t>(src.width);
    const uint32_t h = static_cast<uint32_t>(src.height);
    const size_t nfloats = static_cast<size_t>(w) * h * 4;
    std::vector<float> f_src(nfloats);
    std::vector<float> f_dst(nfloats, 0.0f);
    for (size_t i = 0; i < nfloats; ++i) {
        f_src[i] = static_cast<float>(src.pixels[i]) * (1.0f / 255.0f);
    }
    if (!run_rgba_f32(f_src.data(), f_dst.data(), w, h, use_gpu, err)) {
        return false;
    }
    dst.width = src.width;
    dst.height = src.height;
    dst.channels = 4;
    dst.pixels.resize(nfloats);
    for (size_t i = 0; i < nfloats; ++i) {
        const float clamped = std::clamp(f_dst[i], 0.0f, 1.0f);
        dst.pixels[i] = static_cast<uint8_t>(std::lround(clamped * 255.0f));
    }
    return true;
}

} // namespace broimage
