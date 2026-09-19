#pragma once

#include "embed/embed.h"

#include <functional>
#include <string>

namespace broimage::api {

/// Mounts image codecs (transcodeKTX2, encodePng, encodePngFile, encodeJpeg, encodeJpegFile) onto `bro.image` in Bronze.
void installCodecs();

/// Mounts image operations (geometric, color, preproc) and codecs onto `bro.image` in Bronze.
void installImage();

/// How a path handed to a `bro.image` file entry point (`encodePngFile`,
/// `encodeJpegFile`, and the `decodeU16` / `decodeF32` / `decodeOriented` /
/// `readExifOrientation` overloads that take a filename) becomes a filesystem
/// path. Unset, the path is used as given; a host sets its `fs` resolver so a
/// relative path means what it means to the app
/// (bro: `setPathResolver(&brokit::api::resolveAssetPath)` before
/// installImage, the way it does for brotensor and brodiffusion).
/// Process-wide: every realm shares the host's filesystem.
void setPathResolver(std::function<std::string(const std::string&)> resolver);

} // namespace broimage::api

using broimage::api::installImage;
using broimage::api::installCodecs;
