// Registered codecs and frame decoding (codec.h).

#include "broimage/codec.h"

#include "decode_gif.h"

#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace broimage {

namespace {

// Written once per slot, under the mutex, before the count that publishes
// it: readers take the count (acquire) and read the slots below it without
// a lock, so a decode on any thread costs no more than a loop.
constexpr int kMaxCodecs = 16;
std::array<Codec, kMaxCodecs> g_codecs{};
std::atomic<int> g_count{0};
std::mutex g_register_mu;

// A still image as an animation of one frame.
class StillFrames final : public FrameDecoder {
public:
    StillFrames(SharedBytes bytes, Image first) : bytes_(std::move(bytes)), first_(std::move(first)) {}
    int width() const override { return first_.width; }
    int height() const override { return first_.height; }
    int frame_count() const override { return 1; }
    int loop_count() const override { return 1; }
    const std::vector<int>& delays_ms() const override { return delays_; }
    int index() const override { return index_; }
    bool rewind() override {
        index_ = 0;
        return true;
    }
    bool next(AnimationFrame& out, std::string* error) override {
        if (error) error->clear();
        if (index_ > 0) return false;
        out.rgba = first_.pixels;
        out.delay_ms = 0;
        index_ = 1;
        return true;
    }

private:
    SharedBytes bytes_;
    Image first_;
    std::vector<int> delays_{0};
    int index_ = 0;
};

} // namespace

void register_codec(const Codec& codec) {
    if (!codec.sniff || !codec.decode) return;
    std::lock_guard<std::mutex> lk(g_register_mu);
    const int n = g_count.load(std::memory_order_relaxed);
    if (n >= kMaxCodecs) return;
    for (int i = 0; i < n; ++i)
        if (std::strcmp(g_codecs[std::size_t(i)].name, codec.name ? codec.name : "") == 0) return;
    g_codecs[std::size_t(n)] = codec;
    if (!g_codecs[std::size_t(n)].name) g_codecs[std::size_t(n)].name = "";
    g_count.store(n + 1, std::memory_order_release);
}

const Codec* find_registered_codec(const uint8_t* data, std::size_t size) {
    if (!data || size == 0) return nullptr;
    const int n = g_count.load(std::memory_order_acquire);
    for (int i = 0; i < n; ++i) {
        const Codec& c = g_codecs[std::size_t(i)];
        if (c.sniff(data, size)) return &c;
    }
    return nullptr;
}

std::unique_ptr<FrameDecoder> open_frames(SharedBytes bytes, std::string* error) {
    if (!bytes || bytes->empty()) {
        if (error) *error = "no data";
        return nullptr;
    }
    const uint8_t* d = bytes->data();
    const std::size_t n = bytes->size();
    std::string why;
    if (const Codec* c = find_registered_codec(d, n); c && c->open_frames) {
        if (auto dec = c->open_frames(bytes, &why)) return dec;
    }
    if (detail::is_gif(d, n)) {
        if (auto dec = detail::gif_open(d, n, bytes, &why)) return dec;
    }
    Image first;
    std::string still_why;
    if (!decode_memory(d, n, first, &still_why)) {
        if (error) *error = !why.empty() ? why : still_why;
        return nullptr;
    }
    return std::make_unique<StillFrames>(std::move(bytes), std::move(first));
}

std::unique_ptr<FrameDecoder> open_frames(const uint8_t* data, std::size_t size, std::string* error) {
    if (!data || size == 0) {
        if (error) *error = "no data";
        return nullptr;
    }
    return open_frames(std::make_shared<const std::vector<uint8_t>>(data, data + size), error);
}

bool decode_frames(const uint8_t* data, std::size_t size, DecodedFrames& out, int max_frames,
                   std::size_t max_bytes, std::string* error) {
    out = DecodedFrames{};
    std::unique_ptr<FrameDecoder> dec = open_frames(data, size, error);
    if (!dec) return false;
    out.width = dec->width();
    out.height = dec->height();
    out.frame_count = dec->frame_count();
    out.loop_count = dec->loop_count();
    const std::size_t frame_bytes = std::size_t(out.width) * std::size_t(out.height) * 4;
    std::size_t total = 0;
    for (;;) {
        if ((max_frames > 0 && int(out.frames.size()) >= max_frames) ||
            (max_bytes > 0 && total + frame_bytes > max_bytes)) {
            out.truncated = int(out.frames.size()) < out.frame_count;
            break;
        }
        AnimationFrame f;
        std::string why;
        if (!dec->next(f, &why)) {
            if (!why.empty() && out.frames.empty()) {
                if (error) *error = why;
                out = DecodedFrames{};
                return false;
            }
            // A failure after the first frame ends the animation there, as
            // a browser plays what decoded.
            out.truncated = !why.empty();
            break;
        }
        total += frame_bytes;
        out.frames.push_back(std::move(f));
    }
    return true;
}

} // namespace broimage
