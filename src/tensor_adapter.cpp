#include "broimage/tensor_adapter.h"

#include <brotensor/ops.h>
#include <brotensor/tensor.h>

#include <cstdint>

namespace broimage {

// The public brotensor::image_* entry points dispatch by Device, so both CPU
// tensors and GPU tensors land in the right backend without broimage needing
// to know. These wrappers just keep callers from having to depend on the
// brotensor header directly when all they want is the image preproc.

void image_normalize(const brotensor::Tensor& X,
                     const brotensor::Tensor& mean,
                     const brotensor::Tensor& std_,
                     int N, int C, int H, int W,
                     brotensor::Tensor& Y) {
    brotensor::image_normalize(X, mean, std_, N, C, H, W, Y);
}

void image_u8_to_f32_nhwc_to_nchw(const uint8_t* src,
                                  int N, int H, int W, int C,
                                  float scale, float bias,
                                  brotensor::Tensor& Y) {
    brotensor::image_u8_to_f32_nhwc_to_nchw(src, N, H, W, C, scale, bias, Y);
}

} // namespace broimage
