#pragma once

namespace broimage {

// CLIP (openai/clip-vit-large-patch14) preprocessing constants. RGB channel
// order; applied after `[0,255] -> [0,1]` scaling.
extern const float CLIP_MEAN[3];
extern const float CLIP_STD[3];

// ImageNet (PyTorch torchvision default).
extern const float IMAGENET_MEAN[3];
extern const float IMAGENET_STD[3];

// SAM (Segment Anything) preprocessing — same ImageNet stats, but the
// upstream model expects images padded to 1024x1024 after the normalize.
extern const float SAM_MEAN[3];
extern const float SAM_STD[3];

} // namespace broimage
