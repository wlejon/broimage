// Registered codecs (codec.h): a format broimage cannot read becomes
// readable by every entry point once a host registers it — decode, probe,
// oriented, bounded, open_frames / decode_frames — on any thread.
#include "broimage/codec.h"
#include "broimage/decode.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

namespace {

// "FAKE" + width + height + frame count: frame k is filled with k * 10.
bool fake_sniff(const uint8_t* d, std::size_t n) { return n >= 7 && std::memcmp(d, "FAKE", 4) == 0; }
bool fake_probe(const uint8_t* d, std::size_t, int& w, int& h, int& c) {
    w = d[4];
    h = d[5];
    c = 4;
    return true;
}
bool fake_decode(const uint8_t* d, std::size_t, broimage::Image& out, std::string*) {
    out.width = d[4];
    out.height = d[5];
    out.channels = 4;
    out.pixels.assign(std::size_t(out.width) * out.height * 4, 0);
    return true;
}

class FakeFrames final : public broimage::FrameDecoder {
public:
    explicit FakeFrames(broimage::SharedBytes b) : b_(std::move(b)) {
        delays_.assign((*b_)[6], 40);
    }
    int width() const override { return (*b_)[4]; }
    int height() const override { return (*b_)[5]; }
    int frame_count() const override { return (*b_)[6]; }
    int loop_count() const override { return 0; }
    const std::vector<int>& delays_ms() const override { return delays_; }
    int index() const override { return i_; }
    bool rewind() override { i_ = 0; return true; }
    bool next(broimage::AnimationFrame& f, std::string* e) override {
        if (e) e->clear();
        if (i_ >= frame_count()) return false;
        f.rgba.assign(std::size_t(width()) * height() * 4, uint8_t(i_ * 10));
        f.delay_ms = 40;
        ++i_;
        return true;
    }

private:
    broimage::SharedBytes b_;
    std::vector<int> delays_;
    int i_ = 0;
};

std::unique_ptr<broimage::FrameDecoder> fake_open(broimage::SharedBytes b, std::string*) {
    return std::make_unique<FakeFrames>(std::move(b));
}

} // namespace

int main() {
    const uint8_t fake[] = {'F', 'A', 'K', 'E', 3, 2, 4};
    broimage::Image img;
    std::string err;

    // Unknown before registration.
    CHECK(!broimage::decode_memory(fake, sizeof fake, img, &err));
    CHECK(broimage::find_registered_codec(fake, sizeof fake) == nullptr);

    // Decoding on other threads while the codec is registered.
    std::atomic<bool> stop{false};
    std::atomic<int> decoded{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t)
        threads.emplace_back([&] {
            while (!stop.load()) {
                broimage::Image i;
                if (broimage::decode_memory(fake, sizeof fake, i)) decoded.fetch_add(1);
            }
        });

    broimage::Codec c;
    c.name = "fake";
    c.sniff = fake_sniff;
    c.probe = fake_probe;
    c.decode = fake_decode;
    c.open_frames = fake_open;
    broimage::register_codec(c);
    // A second codec of the same name is ignored.
    broimage::Codec dup = c;
    dup.probe = [](const uint8_t*, std::size_t, int& w, int& h, int& ch) { w = h = ch = 99; return true; };
    broimage::register_codec(dup);

    while (decoded.load() < 100) std::this_thread::yield();
    stop = true;
    for (auto& t : threads) t.join();

    CHECK(broimage::find_registered_codec(fake, sizeof fake) != nullptr);
    CHECK(broimage::decode_memory(fake, sizeof fake, img, &err) && img.width == 3 && img.height == 2);
    CHECK(broimage::decode_memory_oriented(fake, sizeof fake, img, &err) && img.width == 3);
    int w = 0, h = 0, ch = 0;
    CHECK(broimage::probe_dimensions_memory(fake, sizeof fake, &w, &h, &ch) && w == 3 && h == 2 && ch == 4);

    auto dec = broimage::open_frames(fake, sizeof fake, &err);
    CHECK(dec && dec->frame_count() == 4 && dec->loop_count() == 0);
    broimage::DecodedFrames df;
    CHECK(broimage::decode_frames(fake, sizeof fake, df, 0, 0, &err));
    CHECK(df.frames.size() == 4 && df.frames[3].rgba[0] == 30 && df.loop_count == 0);

    broimage::Animation a;
    CHECK(broimage::decode_memory_bounded(fake, sizeof fake, {}, a, &err));
    CHECK(a.frames.size() == 4 && a.width == 3 && a.frames[2].rgba[5] == 20);
    broimage::DecodeLimits lim;
    lim.max_bytes = 3 * 2 * 4 * 3;  // three frames' worth, four in the file
    CHECK(!broimage::decode_memory_bounded(fake, sizeof fake, lim, a, &err));

    // A still format without frames is one frame.
    const uint8_t png_like[] = {'X'};
    CHECK(broimage::open_frames(png_like, sizeof png_like, &err) == nullptr);

    if (g_failed == 0) std::printf("broimage_test_codec_registry: OK\n");
    return g_failed == 0 ? 0 : 1;
}
