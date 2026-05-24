#include "broimage/presets.h"

namespace broimage {

const float CLIP_MEAN[3]     = { 0.48145466f, 0.4578275f,  0.40821073f };
const float CLIP_STD[3]      = { 0.26862954f, 0.26130258f, 0.27577711f };

const float IMAGENET_MEAN[3] = { 0.485f,      0.456f,      0.406f      };
const float IMAGENET_STD[3]  = { 0.229f,      0.224f,      0.225f      };

const float SAM_MEAN[3]      = { 0.485f,      0.456f,      0.406f      };
const float SAM_STD[3]       = { 0.229f,      0.224f,      0.225f      };

} // namespace broimage
