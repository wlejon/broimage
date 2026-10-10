// The streaming GIF decoder (codec.h open_frames, decode_gif.h): every frame
// composed exactly — disposal 1/2/3, transparency, local colour tables,
// interlace, frames off the canvas, a 0 x 0 logical screen — the loop count,
// the delays, rewind, truncation, and the still / bounded entry points that
// share it. Fixtures are written here (gif_writer.h); expectations come from
// a reference compositor that copies whole canvases, nothing like the
// decoder's own bookkeeping.
#include "broimage/codec.h"
#include "broimage/decode.h"

#include "gif_writer.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

namespace {

using Canvas = std::vector<uint8_t>;

// Browsers' model, the slow way: disposal 2 clears the frame's rectangle,
// disposal 3 brings back the whole canvas as it was before the frame.
std::vector<Canvas> reference(int W, int H, const std::vector<uint8_t>& gct, const std::vector<gifw::Frame>& frames) {
    std::vector<Canvas> out;
    Canvas c(std::size_t(W) * H * 4, 0);
    for (const gifw::Frame& f : frames) {
        const Canvas before = c;
        const std::vector<uint8_t>& pal = f.lct.empty() ? gct : f.lct;
        for (int y = 0; y < f.h; ++y)
            for (int x = 0; x < f.w; ++x) {
                const int cx = f.x + x, cy = f.y + y;
                const int i = f.idx[std::size_t(y * f.w + x)];
                if (cx >= W || cy >= H || i == f.transparent) continue;
                uint8_t* p = &c[(std::size_t(cy) * W + cx) * 4];
                p[0] = pal[std::size_t(i) * 3];
                p[1] = pal[std::size_t(i) * 3 + 1];
                p[2] = pal[std::size_t(i) * 3 + 2];
                p[3] = 255;
            }
        out.push_back(c);
        if (f.disposal == 2) {
            for (int y = f.y; y < f.y + f.h && y < H; ++y)
                for (int x = f.x; x < f.x + f.w && x < W; ++x)
                    for (int k = 0; k < 4; ++k) c[(std::size_t(y) * W + x) * 4 + std::size_t(k)] = 0;
        } else if (f.disposal == 3) {
            c = before;
        }
    }
    return out;
}

gifw::Frame solid(int x, int y, int w, int h, uint8_t index) {
    gifw::Frame f;
    f.x = x; f.y = y; f.w = w; f.h = h;
    f.idx.assign(std::size_t(w) * h, index);
    return f;
}

const std::vector<uint8_t> kGct = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};  // red green blue white

} // namespace

int main() {
    const int W = 8, H = 6;
    std::vector<gifw::Frame> frames;
    {
        gifw::Frame f0 = solid(0, 0, W, H, 0);  // red, kept
        f0.disposal = 1;
        f0.delay_cs = 5;
        frames.push_back(f0);
        gifw::Frame f1 = solid(2, 1, 3, 2, 1);  // green, cleared after
        f1.disposal = 2;
        frames.push_back(f1);
        gifw::Frame f2 = solid(1, 0, 3, 3, 0);  // local table, half transparent, restored after
        f2.lct = {255, 255, 0, 0, 255, 255};
        f2.transparent = 1;
        for (std::size_t i = 0; i < f2.idx.size(); ++i) f2.idx[i] = uint8_t(i % 2);
        f2.disposal = 3;
        f2.delay_cs = 20;
        frames.push_back(f2);
        gifw::Frame f3 = solid(4, 0, 4, 6, 0);  // interlaced: each row its own colour
        f3.interlace = true;
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 4; ++x) f3.idx[std::size_t(y * 4 + x)] = uint8_t((y + x) % 4);
        frames.push_back(f3);
        gifw::Frame f4 = solid(6, 4, 5, 5, 3);  // hangs off the canvas
        f4.gce = false;
        frames.push_back(f4);
    }
    const std::vector<uint8_t> gif = gifw::make_gif(W, H, kGct, 2, frames);
    const std::vector<Canvas> want = reference(W, H, kGct, frames);

    // The block walk.
    CHECK(broimage::gif_frame_count(gif.data(), gif.size()) == 5);

    std::string err;
    auto dec = broimage::open_frames(gif.data(), gif.size(), &err);
    CHECK(dec != nullptr);
    if (!dec) return 1;
    CHECK(dec->width() == W && dec->height() == H);
    CHECK(dec->frame_count() == 5);
    CHECK(dec->loop_count() == 3);  // stored 2: plays three times
    CHECK((dec->delays_ms() == std::vector<int>{50, 0, 200, 0, 0}));
    for (int pass = 0; pass < 2; ++pass) {
        for (std::size_t i = 0; i < want.size(); ++i) {
            broimage::AnimationFrame f;
            CHECK(dec->next(f, &err));
            CHECK(f.rgba == want[i]);
            if (f.rgba != want[i]) std::fprintf(stderr, "  frame %zu differs (pass %d)\n", i, pass);
            CHECK(f.delay_ms == dec->delays_ms()[i]);
            CHECK(dec->index() == int(i) + 1);
        }
        broimage::AnimationFrame end;
        CHECK(!dec->next(end, &err) && err.empty());
        CHECK(dec->rewind() && dec->index() == 0);
    }
    // Spot checks against the picture, not the reference: frame 2 shows
    // yellow where its own index was 0 over the cleared hole of frame 1,
    // and frame 3 has frame 2's square restored to red.
    {
        const Canvas& f2 = want[2];
        auto px = [&](const Canvas& c, int x, int y) { return &c[(std::size_t(y) * W + x) * 4]; };
        CHECK(px(f2, 1, 0)[0] == 255 && px(f2, 1, 0)[1] == 255 && px(f2, 1, 0)[2] == 0);  // yellow
        CHECK(px(f2, 3, 1)[3] == 0);   // transparent pixel over frame 1's cleared rectangle
        CHECK(px(f2, 2, 0)[0] == 255 && px(f2, 2, 0)[1] == 0);  // transparent over red: red
        const Canvas& f3 = want[3];
        CHECK(px(f3, 1, 0)[0] == 255 && px(f3, 1, 0)[1] == 0 && px(f3, 1, 0)[3] == 255);  // restored
        CHECK(px(f3, 2, 1)[3] == 0);   // still the hole
    }

    // The still decode is frame 0; the bounded decode is every frame.
    broimage::Image still;
    CHECK(broimage::decode_memory(gif.data(), gif.size(), still, &err));
    CHECK(still.width == W && still.height == H && still.pixels == want[0]);
    int pw = 0, ph = 0, pc = 0;
    CHECK(broimage::probe_dimensions_memory(gif.data(), gif.size(), &pw, &ph, &pc) && pw == W && ph == H);
    broimage::Animation a;
    CHECK(broimage::decode_memory_bounded(gif.data(), gif.size(), {}, a, &err));
    CHECK(a.frames.size() == want.size());
    for (std::size_t i = 0; i < a.frames.size() && i < want.size(); ++i) CHECK(a.frames[i].rgba == want[i]);

    // decode_frames: capped by frames and by bytes.
    broimage::DecodedFrames df;
    CHECK(broimage::decode_frames(gif.data(), gif.size(), df, 2, 0, &err));
    CHECK(df.frames.size() == 2 && df.truncated && df.frame_count == 5 && df.loop_count == 3);
    CHECK(broimage::decode_frames(gif.data(), gif.size(), df, 0, std::size_t(W * H * 4) * 3, &err));
    CHECK(df.frames.size() == 3 && df.truncated);
    CHECK(broimage::decode_frames(gif.data(), gif.size(), df, 0, 0, &err));
    CHECK(df.frames.size() == 5 && !df.truncated && df.frames[4].rgba == want[4]);

    // Loop extension: none plays once, 0 forever.
    {
        auto once = gifw::make_gif(2, 2, kGct, -1, {solid(0, 0, 2, 2, 1), solid(0, 0, 2, 2, 2)});
        auto d1 = broimage::open_frames(once.data(), once.size());
        CHECK(d1 && d1->loop_count() == 1);
        auto ever = gifw::make_gif(2, 2, kGct, 0, {solid(0, 0, 2, 2, 1), solid(0, 0, 2, 2, 2)});
        auto d2 = broimage::open_frames(ever.data(), ever.size());
        CHECK(d2 && d2->loop_count() == 0);
    }

    // A 0 x 0 logical screen takes its frames' extent.
    {
        auto zero = gifw::make_gif(0, 0, kGct, -1, {solid(0, 0, 3, 2, 2), solid(1, 1, 2, 2, 1)});
        auto d = broimage::open_frames(zero.data(), zero.size());
        CHECK(d && d->width() == 3 && d->height() == 3);
    }

    // Many big frames stepped through one at a time; the last one exact.
    {
        const int BW = 300, BH = 200, N = 120;
        std::vector<gifw::Frame> many;
        for (int i = 0; i < N; ++i) {
            gifw::Frame f = solid(0, 0, BW, BH, uint8_t(i % 3));
            f.idx[std::size_t(i)] = 3;
            f.disposal = 1;
            many.push_back(f);
        }
        auto big = gifw::make_gif(BW, BH, kGct, 0, many);
        auto d = broimage::open_frames(big.data(), big.size());
        CHECK(d && d->frame_count() == N);
        broimage::AnimationFrame f;
        int seen = 0;
        while (d && d->next(f)) ++seen;
        CHECK(seen == N);
        const uint8_t* last = &f.rgba[std::size_t(N - 1) * 4];
        CHECK(last[0] == 255 && last[1] == 255 && last[2] == 255);  // white marker
        CHECK(f.rgba[0] == 0 && f.rgba[1] == 0 && f.rgba[2] == 255);  // frame N-1 is blue (index 2)
    }

    // Truncation: a frame whose data the buffer cuts off is an error, and a
    // GIF cut off before any frame opens as nothing.
    {
        std::vector<uint8_t> cut(gif.begin(), gif.end() - 12);
        auto d = broimage::open_frames(cut.data(), cut.size());
        CHECK(d != nullptr);
        broimage::AnimationFrame f;
        int ok = 0;
        std::string why;
        while (d && d->next(f, &why)) ++ok;
        CHECK(ok >= 1 && ok < 5 && !why.empty());
        std::vector<uint8_t> head(gif.begin(), gif.begin() + 30);
        CHECK(broimage::open_frames(head.data(), head.size(), &why) == nullptr);
    }

    if (g_failed == 0) std::printf("broimage_test_decode_gif: OK\n");
    return g_failed == 0 ? 0 : 1;
}
