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

// ---- Access -----------------------------------------------------------------
//
// Which files a program may touch is the host's decision, not broimage's:
// every entry point handed a path (rather than bytes) asks the host's one
// access gate, bronze::embed::checkAccess (embed/embed_access.h), with the
// path resolved (setPathResolver), absolute and lexically normal (UTF-8), on
// the JS thread that made the call, before the file is opened. Bytes need no
// asking. With no gate installed every path is allowed and nothing changes.
//
// What each entry point asks, given a path:
///   decodeU16, decodeF32, decodeOriented, decodeScaled, decodeExifThumbnail,
///   readExifOrientation, readExif, readHeifInfo, decodeThumbnail,
///   decodeFrames, openFrames                     the path Read
///   encodePngFile, encodeJpegFile                the path Write
//
// A refusal is bronze::embed::newAccessError, thrown: an Error whose `message`
// is the gate's sentence (named PermissionRevoked for a grant taken back),
// `code` is "EACCES" and `path` the absolute path asked about. Nothing is read
// or written.

} // namespace broimage::api

using broimage::api::installImage;
using broimage::api::installCodecs;
