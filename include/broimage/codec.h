#pragma once

// Decoders a host registers, and animations decoded a frame at a time.
//
// REGISTERED CODECS. broimage reads what stb_image and its own decoders
// (TIFF, ICO, GIF) read. A host that links a decoder broimage does not carry
// (bro registers WebP from libwebp and SVG from Skia) adds it here, once, and
// from then on every entry point — decode_memory and its oriented, file and
// bounded variants, probe_dimensions_memory, open_frames — reads that format
// too, on any thread. Registered codecs are tried before broimage's own,
// in registration order; one that recognises the bytes (`sniff`) but cannot
// decode them lets the built-in decoders try.
//
// FRAMES. open_frames() returns a FrameDecoder: the frames of an animation
// in order, each composed onto the canvas as it shows (disposal, blending
// and transparency applied), one at a time. It holds the canvas and what
// the next frame needs — never every frame — so a long animation can be
// stepped through in bounded memory. A still image is an animation of one
// frame.

#include "broimage/buffer.h"
#include "broimage/decode.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace broimage {

// The bytes an animation is decoded from, shared with whoever else holds
// them: a FrameDecoder keeps its source alive.
using SharedBytes = std::shared_ptr<const std::vector<uint8_t>>;

class FrameDecoder {
public:
    virtual ~FrameDecoder() = default;

    // The canvas every frame is composed onto.
    virtual int width() const = 0;
    virtual int height() const = 0;
    // Frames in one play-through, from the container's own structure.
    virtual int frame_count() const = 0;
    // How many times the animation plays through; 0 = forever. GIF: no
    // NETSCAPE2.0 loop extension plays once, a stored count n plays n + 1
    // times, 0 forever (as browsers read it). WebP: the ANIM loop count.
    virtual int loop_count() const = 0;
    // Each frame's delay in ms as stored (frame_count() entries; 0 =
    // unspecified), from the headers, without decoding. A player clamps
    // tiny ones itself (browsers show <= 10 ms as 100 ms).
    virtual const std::vector<int>& delays_ms() const = 0;

    // The next frame, composed: `out.rgba` is width() * height() * 4
    // straight-alpha RGBA8, `out.delay_ms` its delay. False after the last
    // frame (`error`, if given, left empty) or on a decode failure (`error`
    // says why); a decoder that failed stays failed until rewind().
    virtual bool next(AnimationFrame& out, std::string* error = nullptr) = 0;
    // Back to before the first frame.
    virtual bool rewind() = 0;
    // Frames returned since the last rewind.
    virtual int index() const = 0;
};

struct Codec {
    const char* name = "";  // a second codec of the same name is ignored
    bool (*sniff)(const uint8_t* data, std::size_t size) = nullptr;
    // Dimensions from the header; channels as decode_memory documents.
    bool (*probe)(const uint8_t* data, std::size_t size, int& width, int& height, int& channels) = nullptr;
    // The still image (an animation's first frame), straight RGBA8.
    bool (*decode)(const uint8_t* data, std::size_t size, Image& out, std::string* why) = nullptr;
    // Optional: the frames of an animation. Null: the codec's images are
    // stills and open_frames() wraps `decode`.
    std::unique_ptr<FrameDecoder> (*open_frames)(SharedBytes bytes, std::string* why) = nullptr;
};

// Add a codec (thread-safe; at most 16). Registered codecs keep their
// function pointers for the life of the process.
void register_codec(const Codec& codec);

// The registered codec that recognises `data`, or null.
const Codec* find_registered_codec(const uint8_t* data, std::size_t size);

// The frames of `bytes`: an animated GIF or a registered animated format
// (bro: WebP) frame by frame, any other image as one frame. Null when
// nothing decodes it (`error` says why). Cheap: it reads the headers and
// allocates the canvas; frames decode in next().
std::unique_ptr<FrameDecoder> open_frames(SharedBytes bytes, std::string* error = nullptr);
std::unique_ptr<FrameDecoder> open_frames(const uint8_t* data, std::size_t size,
                                          std::string* error = nullptr);

// Every frame at once, bounded: decoding stops (and `truncated` is set)
// before a frame that would pass `max_frames` or `max_bytes` (RGBA8 of the
// frames returned; 0 = no limit). For animations short enough to hold.
struct DecodedFrames {
    int width = 0;
    int height = 0;
    int frame_count = 0;  // in the file
    int loop_count = 1;   // as FrameDecoder::loop_count
    std::vector<AnimationFrame> frames;
    bool truncated = false;
};
bool decode_frames(const uint8_t* data, std::size_t size, DecodedFrames& out,
                   int max_frames = 0, std::size_t max_bytes = 0, std::string* error = nullptr);

} // namespace broimage
