#pragma once

#include "embed/embed.h"

namespace broimage::api {

/// Mounts image codecs (transcodeKTX2, encodePng, encodePngFile, encodeJpeg, encodeJpegFile) onto `bro.image` in Bronze.
void installCodecs();

/// Mounts image operations (geometric, color, preproc) and codecs onto `bro.image` in Bronze.
void installImage();

} // namespace broimage::api

using broimage::api::installImage;
using broimage::api::installCodecs;
