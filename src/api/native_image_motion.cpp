// bro.image.blockMotion / blockResidual: block motion between two luma
// frames, the hot half of broworkshop's motion oracle (lib/kit/motion.js).
// The kernel is include/broimage/motion.h.

#include "host_image_internal.h"

#include <broimage/motion.h>

#include <cstdint>
#include <span>
#include <string>

namespace broimage::api {

namespace {

// The grid fields of opts: width, height, nx, ny, block?, margin?, radius?,
// threads?.
bool readGrid(Value opts, broimage::BlockMotionGrid* g, int32_t* threads, const char* who) {
    if (!getPropI32(opts, "width", &g->width, 0) || !getPropI32(opts, "height", &g->height, 0) ||
        !getPropI32(opts, "nx", &g->nx, 0) || !getPropI32(opts, "ny", &g->ny, 0) ||
        !getPropI32(opts, "block", &g->block, 8) || !getPropI32(opts, "margin", &g->margin, 16) ||
        !getPropI32(opts, "radius", &g->radius, 4) || !getPropI32(opts, "threads", threads, 0))
        return false;
    if (g->width < 4 || g->height < 4 || g->nx <= 0 || g->ny <= 0 || g->block <= 0 || g->block > 64 ||
        g->margin < 0 || g->radius <= 0 || g->radius > 64) {
        ev::throwRangeError(std::string(who) + ": bad grid (width, height >= 4; nx, ny > 0; block 1..64; radius 1..64)");
        return false;
    }
    if (*threads < 0 || *threads > 64) {
        ev::throwRangeError(std::string(who) + ": threads 0..64");
        return false;
    }
    return true;
}

bool requirePlane(const TypedArrayView& v, size_t n, const char* who) {
    if (v.bytesPerElement != 1) {
        ev::throwTypeError(std::string(who) + ": " + v.name + " must be a Uint8Array / Uint8ClampedArray");
        return false;
    }
    if (v.byteLength < n) {
        ev::throwRangeError(std::string(who) + ": " + v.name + " holds " + std::to_string(v.byteLength) +
                            " bytes, the plane needs " + std::to_string(n));
        return false;
    }
    return true;
}

// blockMotion(fullA, fullB, halfA, halfB, quarterA, quarterB,
//             { width, height, nx, ny, block?, margin?, radius?, lostTexture?, threads?, out? })
//   -> out: Float64Array(6 nx ny): dx, dy, P, D, sigma, status per block
Value imageBlockMotion(Value, std::span<const Value> args) {
    const char* who = "blockMotion";
    if (args.size() < 7 || !ev::isObject(args[6]))
        return ev::throwTypeError("blockMotion(fullA, fullB, halfA, halfB, quarterA, quarterB, { width, height, nx, ny })");
    TypedArrayView fa, fb, ha, hb, qa, qb;
    if (!unpackTypedArray(args[0], "fullA", &fa) || !unpackTypedArray(args[1], "fullB", &fb) ||
        !unpackTypedArray(args[2], "halfA", &ha) || !unpackTypedArray(args[3], "halfB", &hb) ||
        !unpackTypedArray(args[4], "quarterA", &qa) || !unpackTypedArray(args[5], "quarterB", &qb))
        return ev::undefined();
    // Rooted: every option read may allocate and move it (embed.h's pointer contract).
    ev::Persistent optsRoot(args[6]);
    broimage::BlockMotionGrid g;
    int32_t threads = 0;
    double lostTexture = 1.5;
    if (!readGrid(optsRoot.get(), &g, &threads, who) || !getPropF64(optsRoot.get(), "lostTexture", &lostTexture, 1.5))
        return ev::undefined();
    const size_t W = size_t(g.width), H = size_t(g.height);
    const size_t full = W * H, half = (W >> 1) * (H >> 1), quarter = (W >> 2) * (H >> 2);
    if (!requirePlane(fa, full, who) || !requirePlane(fb, full, who) || !requirePlane(ha, half, who) ||
        !requirePlane(hb, half, who) || !requirePlane(qa, quarter, who) || !requirePlane(qb, quarter, who))
        return ev::undefined();
    const size_t n = size_t(6) * g.nx * g.ny;
    ev::Persistent outRoot(ev::getProperty(optsRoot.get(), "out"));
    if (ev::isUndefined(outRoot.get()) || ev::isNull(outRoot.get()))
        outRoot = ev::Persistent(ev::createTypedArray(ev::elements::Float64, static_cast<uint32_t>(n)));
    TypedArrayView out;
    if (!unpackTypedArray(outRoot.get(), "out", &out)) return ev::undefined();
    if (out.kind != ev::elements::Float64 || out.byteLength < n * 8)
        return ev::throwTypeError("blockMotion: out must be a Float64Array of 6 nx ny");
    if (!resolveViews({&fa, &fb, &ha, &hb, &qa, &qb, &out})) return ev::undefined();
    broimage::block_motion_u8(fa.data, fb.data, ha.data, hb.data, qa.data, qb.data, g, lostTexture,
                              reinterpret_cast<double*>(out.data), threads);
    return outRoot.get();
}

// blockResidual(halfA, halfB, motion, d, { width, height, nx, ny, block?, margin?, threads? }) -> d
//   motion: Float32Array(2 nx ny) full-res px, NaN where none; d: Float32Array(nx ny),
//   rewritten where finite
Value imageBlockResidual(Value, std::span<const Value> args) {
    const char* who = "blockResidual";
    if (args.size() < 5 || !ev::isObject(args[4]))
        return ev::throwTypeError("blockResidual(halfA, halfB, motion, d, { width, height, nx, ny })");
    TypedArrayView ha, hb, mv, d;
    if (!unpackTypedArray(args[0], "halfA", &ha) || !unpackTypedArray(args[1], "halfB", &hb) ||
        !unpackTypedArray(args[2], "motion", &mv) || !unpackTypedArray(args[3], "d", &d))
        return ev::undefined();
    ev::Persistent optsRoot(args[4]);
    broimage::BlockMotionGrid g;
    int32_t threads = 0;
    if (!readGrid(optsRoot.get(), &g, &threads, who)) return ev::undefined();
    const size_t half = size_t(g.width >> 1) * size_t(g.height >> 1), nb = size_t(g.nx) * g.ny;
    if (!requirePlane(ha, half, who) || !requirePlane(hb, half, who)) return ev::undefined();
    if (!requireFloat32(mv, who) || !requireFloat32(d, who)) return ev::undefined();
    if (mv.byteLength < nb * 8 || d.byteLength < nb * 4)
        return ev::throwRangeError("blockResidual: motion holds 2 nx ny floats, d nx ny");
    if (!resolveViews({&ha, &hb, &mv, &d})) return ev::undefined();
    broimage::block_residual_multi_u8(ha.data, hb.data, g, reinterpret_cast<const float*>(mv.data),
                                      reinterpret_cast<float*>(d.data), threads);
    return d.root.get();
}

// motionLuma(rgba, { width, height, stride?, full?, half?, quarter?, threads? })
//   -> { full, half, quarter }: Uint8Arrays (the ones given, else new)
Value imageMotionLuma(Value, std::span<const Value> args) {
    const char* who = "motionLuma";
    if (args.size() < 2 || !ev::isObject(args[1]))
        return ev::throwTypeError("motionLuma(rgba, { width, height, stride? })");
    TypedArrayView src;
    if (!unpackTypedArray(args[0], "rgba", &src)) return ev::undefined();
    ev::Persistent optsRoot(args[1]);
    int32_t w = 0, h = 0, stride = 0, threads = 0;
    if (!getPropI32(optsRoot.get(), "width", &w, 0) || !getPropI32(optsRoot.get(), "height", &h, 0) ||
        !getPropI32(optsRoot.get(), "stride", &stride, 0) || !getPropI32(optsRoot.get(), "threads", &threads, 0))
        return ev::undefined();
    if (w <= 0 || h <= 0 || stride < 0 || (stride && stride < w * 4))
        return ev::throwRangeError("motionLuma: width/height must be positive, stride 0 or at least 4 width");
    if (threads < 0 || threads > 64) return ev::throwRangeError("motionLuma: threads 0..64");
    if (!requireImage(src, w, h, 4, stride, 1, who)) return ev::undefined();
    const size_t W = size_t(w), H = size_t(h);
    const size_t sizes[3] = {W * H, (W >> 1) * (H >> 1), (W >> 2) * (H >> 2)};
    const char* names[3] = {"full", "half", "quarter"};
    ev::Persistent roots[3];
    TypedArrayView views[3];
    for (int i = 0; i < 3; ++i) {
        roots[i] = ev::Persistent(ev::getProperty(optsRoot.get(), names[i]));
        if (ev::isUndefined(roots[i].get()) || ev::isNull(roots[i].get()))
            roots[i] = ev::Persistent(ev::createTypedArray(ev::elements::Uint8, static_cast<uint32_t>(sizes[i])));
        if (!unpackTypedArray(roots[i].get(), names[i], &views[i]) || !requirePlane(views[i], sizes[i], who))
            return ev::undefined();
    }
    ObjectBuilder b;
    for (int i = 0; i < 3; ++i) b.set(names[i], roots[i].get());
    Value res = b.build();
    ev::Persistent resRoot(res);
    if (!resolveViews({&src, &views[0], &views[1], &views[2]})) return ev::undefined();
    broimage::motion_luma_u8(src.data, w, h, stride, views[0].data, views[1].data, views[2].data, threads);
    return resRoot.get();
}

} // namespace

void installMotionOnto(Value imageObj) {
    ObjectBuilder b(imageObj);
    b.def("motionLuma", 2, imageMotionLuma);
    b.def("blockMotion", 7, imageBlockMotion);
    b.def("blockResidual", 5, imageBlockResidual);
}

} // namespace broimage::api
