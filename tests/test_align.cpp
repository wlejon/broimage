// align_similarity recovers a known similarity between a textured image and
// a warped copy of it: sub-pixel, with an exposure change, from a rough start
// (the coarse search), over a mask when the rest of the frame moves another
// way, and translation-only. warp_affine_hwc_u8: identity is exact, fill
// applies outside the source.

#include "broimage/align.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

namespace {

constexpr int W = 320, H = 240;

// Smooth blobs plus a fine texture: what feathers on a branch look like to
// an optimizer, roughly. Deterministic.
std::vector<uint8_t> texture(uint32_t seed) {
    std::vector<uint8_t> img(static_cast<size_t>(W) * H * 3);
    uint32_t s = seed;
    auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return (s >> 8) * (1.0 / 16777216.0); };
    struct Blob { double x, y, r, v[3]; };
    std::vector<Blob> blobs(60);
    for (Blob& b : blobs) {
        b.x = rnd() * W; b.y = rnd() * H; b.r = 6 + rnd() * 30;
        for (double& v : b.v) v = rnd() * 2 - 1;
    }
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            double c[3] = {0.45, 0.4, 0.35};
            for (const Blob& b : blobs) {
                const double d2 = ((x - b.x) * (x - b.x) + (y - b.y) * (y - b.y)) / (b.r * b.r);
                const double g = std::exp(-d2);
                for (int k = 0; k < 3; ++k) c[k] += 0.25 * g * b.v[k];
            }
            const double fine = 0.04 * std::sin(x * 0.9 + y * 0.35) * std::cos(y * 0.7 - x * 0.2);
            for (int k = 0; k < 3; ++k) {
                const double v = std::fmin(std::fmax(c[k] + fine, 0.0), 1.0);
                img[(static_cast<size_t>(y) * W + x) * 3 + k] = static_cast<uint8_t>(v * 255 + 0.5);
            }
        }
    return img;
}

void invert(const double m[6], double r[6]) {
    const double det = m[0] * m[4] - m[1] * m[3];
    r[0] = m[4] / det;  r[1] = -m[1] / det;
    r[3] = -m[3] / det; r[4] = m[0] / det;
    r[2] = -(r[0] * m[2] + r[1] * m[5]);
    r[5] = -(r[3] * m[2] + r[4] * m[5]);
}

// The moving image for a transform T (reference -> moving): mov(q) = ref(T^-1 q).
std::vector<uint8_t> moved(const std::vector<uint8_t>& ref, const broimage::Similarity& t,
                           double gain = 1.0, double bias = 0.0) {
    double m[6], inv[6];
    broimage::similarity_matrix(t, 0.5 * W, 0.5 * H, m);
    invert(m, inv);
    std::vector<uint8_t> out(ref.size());
    broimage::warp_affine_hwc_u8(ref.data(), W, H, 3, out.data(), W, H, inv, broimage::Filter::Bicubic);
    if (gain != 1.0 || bias != 0.0)
        for (uint8_t& v : out) v = static_cast<uint8_t>(std::fmin(std::fmax(v * gain + bias * 255, 0.0), 255.0) + 0.5);
    return out;
}

// A central ellipse: the subject.
std::vector<uint8_t> subjectMask() {
    std::vector<uint8_t> m(static_cast<size_t>(W) * H, 0);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const double dx = (x + 0.5 - W * 0.5) / (W * 0.28), dy = (y + 0.5 - H * 0.5) / (H * 0.3);
            m[static_cast<size_t>(y) * W + x] = dx * dx + dy * dy <= 1.0;
        }
    return m;
}

bool close(const broimage::Similarity& a, const broimage::Similarity& b, double px) {
    return std::fabs(a.tx - b.tx) <= px && std::fabs(a.ty - b.ty) <= px &&
           std::fabs(a.scale - b.scale) <= 2e-3 && std::fabs(a.rotation - b.rotation) <= 2e-3;
}

void print(const char* what, const broimage::AlignResult& r) {
    std::printf("%-28s scale %.5f rot %.5f t (%.3f, %.3f) gain %.3f bias %.3f rms %.4f ncc %.4f cov %.2f it %d%s\n",
                what, r.transform.scale, r.transform.rotation, r.transform.tx, r.transform.ty,
                r.gain, r.bias, r.rms, r.ncc, r.coverage, r.iterations, r.converged ? "" : " (not converged)");
}

} // namespace

int main() {
    const std::vector<uint8_t> ref = texture(7);
    const std::vector<uint8_t> mask = subjectMask();

    // 1. Identity warp is exact.
    {
        const double id[6] = {1, 0, 0, 0, 1, 0};
        std::vector<uint8_t> out(ref.size());
        broimage::warp_affine_hwc_u8(ref.data(), W, H, 3, out.data(), W, H, id, broimage::Filter::Bilinear);
        CHECK(out == ref);
        broimage::warp_affine_hwc_u8(ref.data(), W, H, 3, out.data(), W, H, id, broimage::Filter::Bicubic);
        CHECK(out == ref);
        // Shifted by a whole image width: all fill.
        const double off[6] = {1, 0, 1000, 0, 1, 0};
        const uint8_t fill[3] = {1, 2, 3};
        broimage::warp_affine_hwc_u8(ref.data(), W, H, 3, out.data(), W, H, off, broimage::Filter::Bilinear, fill);
        CHECK(out[0] == 1 && out[1] == 2 && out[2] == 3 && out[out.size() - 1] == 3);
    }

    // 2. A small similarity, sub-pixel, from identity.
    {
        broimage::Similarity t;
        t.scale = 1.03; t.rotation = 0.02; t.tx = 3.4; t.ty = -2.7;
        const auto mov = moved(ref, t);
        const auto r = broimage::align_similarity(ref.data(), mov.data(), W, H, 3, nullptr);
        print("similarity", r);
        CHECK(r.converged);
        CHECK(close(r.transform, t, 0.05));
        CHECK(r.ncc > 0.99);
    }

    // 3. An exposure change between the frames.
    {
        broimage::Similarity t;
        t.scale = 0.98; t.rotation = -0.015; t.tx = -1.8; t.ty = 4.1;
        const auto mov = moved(ref, t, 1.15, -0.05);
        const auto r = broimage::align_similarity(ref.data(), mov.data(), W, H, 3, mask.data());
        print("exposure", r);
        CHECK(close(r.transform, t, 0.08));
        CHECK(std::fabs(1.0 / r.gain - 1.15) < 0.03);   // mov * gain ~ ref, mov = ref * 1.15 - .05
    }

    // 4. A large shift: the coarse search finds it from identity.
    {
        broimage::Similarity t;
        t.tx = 41; t.ty = -29;
        const auto mov = moved(ref, t);
        const auto r = broimage::align_similarity(ref.data(), mov.data(), W, H, 3, mask.data());
        print("large shift", r);
        CHECK(close(r.transform, t, 0.1));
        // Without the search, from identity, it does not reach it...
        broimage::AlignOptions none;
        none.search = 0;
        const auto r0 = broimage::align_similarity(ref.data(), mov.data(), W, H, 3, mask.data(), {}, none);
        print("large shift, no search", r0);
        // ...but from a rough initial estimate (a mask centroid, say) it does.
        broimage::Similarity rough;
        rough.tx = 36; rough.ty = -33;
        const auto r1 = broimage::align_similarity(ref.data(), mov.data(), W, H, 3, mask.data(), rough, none);
        print("large shift, rough init", r1);
        CHECK(close(r1.transform, t, 0.1));
    }

    // 5. The subject moves one way and the background another: the mask
    //    picks the subject's motion.
    {
        broimage::Similarity subject, background;
        subject.tx = 6.2; subject.ty = 1.3; subject.scale = 1.02;
        background.tx = -9; background.ty = 5;
        const auto movS = moved(ref, subject);
        const auto movB = moved(ref, background);
        std::vector<uint8_t> mov(movB);
        // Composite: the subject (the mask region, as it lands after its own
        // motion) over the background moved differently.
        double ms[6];
        broimage::similarity_matrix(subject, 0.5 * W, 0.5 * H, ms);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                // Where does mov pixel (x, y) come from in ref, for the subject?
                double inv[6];
                invert(ms, inv);
                const double px = inv[0] * (x + 0.5) + inv[1] * (y + 0.5) + inv[2];
                const double py = inv[3] * (x + 0.5) + inv[4] * (y + 0.5) + inv[5];
                const int rx = static_cast<int>(px), ry = static_cast<int>(py);
                if (rx < 0 || ry < 0 || rx >= W || ry >= H || !mask[static_cast<size_t>(ry) * W + rx]) continue;
                for (int k = 0; k < 3; ++k)
                    mov[(static_cast<size_t>(y) * W + x) * 3 + k] = movS[(static_cast<size_t>(y) * W + x) * 3 + k];
            }
        // Erode the mask a little: its edge pixels carry the seam.
        std::vector<uint8_t> inner(mask.size(), 0);
        for (int y = 3; y < H - 3; ++y)
            for (int x = 3; x < W - 3; ++x) {
                bool all = true;
                for (int d = -3; d <= 3 && all; ++d)
                    all = mask[static_cast<size_t>(y + d) * W + x] && mask[static_cast<size_t>(y) * W + x + d];
                inner[static_cast<size_t>(y) * W + x] = all;
            }
        const auto r = broimage::align_similarity(ref.data(), mov.data(), W, H, 3, inner.data());
        print("subject over background", r);
        CHECK(close(r.transform, subject, 0.15));
        const auto whole = broimage::align_similarity(ref.data(), mov.data(), W, H, 3, nullptr);
        print("same, no mask", whole);
    }

    // 6. Translation only.
    {
        broimage::Similarity t;
        t.tx = -2.25; t.ty = 1.75;
        const auto mov = moved(ref, t);
        broimage::AlignOptions o;
        o.similarity = false;
        const auto r = broimage::align_similarity(ref.data(), mov.data(), W, H, 3, mask.data(), {}, o);
        print("translation only", r);
        CHECK(close(r.transform, t, 0.05));
        CHECK(r.transform.scale == 1.0 && r.transform.rotation == 0.0);
    }

    // 7. A mask too small to align returns the initial estimate.
    {
        std::vector<uint8_t> tiny(mask.size(), 0);
        tiny[static_cast<size_t>(H / 2) * W + W / 2] = 1;
        broimage::Similarity init;
        init.tx = 5;
        const auto r = broimage::align_similarity(ref.data(), ref.data(), W, H, 3, tiny.data(), init);
        CHECK(!r.converged && r.transform.tx == 5);
    }

    if (g_failed) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test_align: all checks passed\n");
    return 0;
}
