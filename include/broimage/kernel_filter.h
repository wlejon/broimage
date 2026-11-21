#pragma once
// Optional brokernel-compiled 2D image filter helper (built when BROIMAGE_WITH_BROKERNEL=ON).

#include "broimage/buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace broimage {

class KernelFilter {
public:
    KernelFilter();
    ~KernelFilter();
    KernelFilter(KernelFilter&&) noexcept;
    KernelFilter& operator=(KernelFilter&&) noexcept;
    KernelFilter(const KernelFilter&) = delete;
    KernelFilter& operator=(const KernelFilter&) = delete;

    /// Compiles a 2D `@kernel @grid(2)` `.bk` kernel taking
    /// `(g: grid<2>, src: read_store<vec4<f32>, 2> | stencil<vec4<f32>, 2, R>, dst: write_store<vec4<f32>, 2>)`.
    bool compile(std::string_view bk_source,
                 std::string_view entry_name = "k_filter",
                 std::string* err = nullptr);

    bool valid() const noexcept;

    /// Runs the filter over tightly-packed RGBA `f32` pixels (`4 * width * height` floats).
    bool run_rgba_f32(const float* src_rgba,
                      float* dst_rgba,
                      std::uint32_t width,
                      std::uint32_t height,
                      bool use_gpu = false,
                      std::string* err = nullptr) const;

    /// Runs the filter on an RGBA8 `Image` (converting to/from linear/normalized [0, 1] floats).
    bool run_rgba8(const Image& src,
                   Image& dst,
                   bool use_gpu = false,
                   std::string* err = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace broimage
