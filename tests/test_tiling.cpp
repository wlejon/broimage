#include "broimage/tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failed++; \
    } \
} while (0)

static bool nearf(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol;
}

// Reconstruct a known HWC image by tiling it into overlapping tiles, copying
// each tile straight out of the source (the "operator" is identity), feathering
// + accumulating, and normalizing. Because every tile carries the SAME source
// value at a given global pixel, the weighted average must reproduce the source
// exactly — this validates the whole split/feather/accumulate/normalize chain.
static void reconstruct(int W, int H, int C, int tile, int step) {
    std::vector<float> src(static_cast<std::size_t>(W) * H * C);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c)
                src[((static_cast<std::size_t>(y) * W) + x) * C + c] =
                    static_cast<float>(x * 2 + y * 3 + c * 7);

    std::vector<float> acc(static_cast<std::size_t>(W) * H * C, 0.0f);
    std::vector<float> wacc(static_cast<std::size_t>(W) * H, 0.0f);
    std::vector<float> tbuf(static_cast<std::size_t>(tile) * tile * C);
    std::vector<float> win(static_cast<std::size_t>(tile) * tile);

    for (int ty = 0; ty < H; ty += step) {
        for (int tx = 0; tx < W; tx += step) {
            const int tw = std::min(tile, W - tx);
            const int th = std::min(tile, H - ty);
            // Copy the tile rect out of the source.
            for (int j = 0; j < th; ++j)
                for (int i = 0; i < tw; ++i)
                    for (int c = 0; c < C; ++c)
                        tbuf[((static_cast<std::size_t>(j) * tw) + i) * C + c] =
                            src[(((static_cast<std::size_t>(ty + j) * W) + (tx + i)) * C) + c];
            // Overlap on an edge only where a neighbour exists (0 on image border).
            const int ov = tile - step;
            const int ov_l = (tx > 0) ? ov : 0;
            const int ov_t = (ty > 0) ? ov : 0;
            const int ov_r = (tx + tw < W) ? ov : 0;
            const int ov_b = (ty + th < H) ? ov : 0;
            broimage::feather_window_f32(win.data(), tw, th, ov_l, ov_r, ov_t, ov_b);
            broimage::accumulate_tile_f32(acc.data(), wacc.data(), W, H, C,
                                          tbuf.data(), tw, th, tx, ty, win.data());
        }
    }
    broimage::normalize_accumulator_f32(acc.data(), wacc.data(), W * H, C);

    bool covered = true, exact = true;
    for (int i = 0; i < W * H; ++i) {
        if (wacc[i] <= 0.0f) covered = false;
        for (int c = 0; c < C; ++c)
            if (!nearf(acc[i * C + c], src[i * C + c], 1e-3f)) exact = false;
    }
    CHECK(covered);
    CHECK(exact);
}

int main() {
    // ----- feather_window_f32 ------------------------------------------------
    {
        // No overlap anywhere -> all weights exactly 1.
        std::vector<float> w(8 * 6);
        broimage::feather_window_f32(w.data(), 8, 6, 0, 0, 0, 0);
        bool all_one = true;
        for (float v : w) if (!nearf(v, 1.0f)) all_one = false;
        CHECK(all_one);
    }
    {
        // Symmetric horizontal ramp (overlap 3 on both sides), no vertical.
        const int tw = 12, th = 1, ov = 3;
        std::vector<float> w(tw * th);
        broimage::feather_window_f32(w.data(), tw, th, ov, ov, 0, 0);
        // Interior pixels (outside both ramps) are 1.
        for (int x = ov; x < tw - ov; ++x) CHECK(nearf(w[x], 1.0f));
        // Leading ramp strictly increases and stays in (0,1].
        for (int x = 0; x < ov; ++x) {
            CHECK(w[x] > 0.0f && w[x] <= 1.0f);
            if (x > 0) CHECK(w[x] > w[x - 1]);
        }
        // Mirror symmetry across the tile.
        for (int x = 0; x < tw; ++x) CHECK(nearf(w[x], w[tw - 1 - x]));
        // (Partition-of-unity across the seam is validated end-to-end by
        // reconstruct() below.)
    }
    {
        // Border behaviour: overlap only on the right edge -> left half full
        // weight, right edge feathered.
        const int tw = 10, th = 1, ov = 4;
        std::vector<float> w(tw * th);
        broimage::feather_window_f32(w.data(), tw, th, 0, ov, 0, 0);
        for (int x = 0; x < tw - ov; ++x) CHECK(nearf(w[x], 1.0f));
        CHECK(w[tw - 1] > 0.0f && w[tw - 1] < w[tw - ov]);
    }
    {
        // Colliding overlaps (ov_l + ov_r > tw) must still produce a valid,
        // positive, finite window.
        const int tw = 6, th = 1;
        std::vector<float> w(tw * th);
        broimage::feather_window_f32(w.data(), tw, th, 5, 5, 0, 0);
        for (float v : w) CHECK(v > 0.0f && std::isfinite(v));
    }

    // ----- accumulate + normalize: single tile = passthrough ----------------
    {
        const int W = 4, H = 3, C = 2;
        std::vector<float> tile(W * H * C);
        for (int i = 0; i < W * H * C; ++i) tile[i] = static_cast<float>(i + 1);
        std::vector<float> acc(W * H * C, 0.0f), wacc(W * H, 0.0f), win(W * H, 1.0f);
        broimage::accumulate_tile_f32(acc.data(), wacc.data(), W, H, C,
                                      tile.data(), W, H, 0, 0, win.data());
        broimage::normalize_accumulator_f32(acc.data(), wacc.data(), W * H, C);
        for (int i = 0; i < W * H * C; ++i) CHECK(nearf(acc[i], tile[i]));
    }

    // ----- uncovered pixels resolve to 0 ------------------------------------
    {
        const int W = 4, H = 1, C = 1;
        std::vector<float> tile(2, 5.0f);   // 2x1 tile of value 5
        std::vector<float> acc(W * H * C, 0.0f), wacc(W * H, 0.0f), win(2, 1.0f);
        // Place the 2-wide tile at x=0; pixels 2,3 are never covered.
        broimage::accumulate_tile_f32(acc.data(), wacc.data(), W, H, C,
                                      tile.data(), 2, 1, 0, 0, win.data());
        broimage::normalize_accumulator_f32(acc.data(), wacc.data(), W * H, C);
        CHECK(nearf(acc[0], 5.0f) && nearf(acc[1], 5.0f));
        CHECK(nearf(acc[2], 0.0f) && nearf(acc[3], 0.0f));
    }

    // ----- out-of-bounds tile placement is clipped --------------------------
    {
        const int W = 3, H = 3, C = 1;
        std::vector<float> tile(4, 9.0f);   // 2x2 tile
        std::vector<float> acc(W * H, 0.0f), wacc(W * H, 0.0f), win(4, 1.0f);
        // Place near the corner so half the tile spills past the right/bottom.
        broimage::accumulate_tile_f32(acc.data(), wacc.data(), W, H, C,
                                      tile.data(), 2, 2, 2, 2, win.data());
        // Only (2,2) lands inside.
        CHECK(nearf(wacc[2 * 3 + 2], 1.0f));
        broimage::normalize_accumulator_f32(acc.data(), wacc.data(), W * H, C);
        CHECK(nearf(acc[2 * 3 + 2], 9.0f));
    }

    // ----- full reconstruction (the real contract) --------------------------
    reconstruct(/*W=*/20, /*H=*/16, /*C=*/1, /*tile=*/8, /*step=*/6);   // overlap 2
    reconstruct(/*W=*/37, /*H=*/29, /*C=*/3, /*tile=*/16, /*step=*/10); // overlap 6, ragged edges
    reconstruct(/*W=*/15, /*H=*/15, /*C=*/1, /*tile=*/15, /*step=*/15); // single tile

    return g_failed == 0 ? 0 : 1;
}
