// Pixel-level coverage for the bro.image ops surface (native_image_ops.cpp)
// that test_image_api.cpp only checks for presence.
//
// Every call here hands the native typed arrays AND an options object, so the
// native makes allocating embed calls (property reads) between unpacking its
// buffers and running the kernel. Under BRONZE_GC_STRESS=1 each of those
// collects and moves the buffers, which is what catches a native that runs a
// kernel on the pre-collection address (embed.h, THE POINTER CONTRACT). The
// broimage_test_api_gc_stress ctest entry runs this under exactly that.
//
// Linked into broimage_test_api; called from its main(). Failures exit the
// process: assert() is a no-op in the Release configuration this test runs in.

#include "embed/embed.h"
#include "eval/eval.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
namespace ev = bronze::embed;
} // namespace

void broimageTestOpsSurface() {
    std::cout << "Checking the bro.image ops kernels..." << std::endl;

    const char* script = R"JS(
        const I = bro.image;
        const fail = (m) => { throw new Error(m); };
        const eq = (got, want, what) => {
            const g = Array.from(got).join(","), w = want.join(",");
            if (g !== w) fail(what + " gave [" + g + "], expected [" + w + "]");
        };
        const near = (got, want, what) => {
            for (let i = 0; i < want.length; i++) {
                if (Math.abs(got[i] - want[i]) > 1e-4) {
                    fail(what + " gave [" + Array.from(got).join(",") + "], expected [" +
                         want.join(",") + "]");
                }
            }
        };

        // ── gradient: a 5-element stop carries alpha ────────────────────────
        {
            const g = I.gradient([[0, 0, 0, 0, 0], [1, 255, 255, 255, 255]], 2);
            eq(g, [0, 0, 0, 0, 255, 255, 255, 255], "gradient with alpha");
        }

        // ── lookup: float and integer sources ───────────────────────────────
        {
            const lut = new Uint8Array([10, 20, 30, 40, 50, 60, 70, 80]);
            const dst = new Uint8Array(8);
            I.lookup(dst, new Float32Array([0, 1]), lut, { lo: 0, hi: 1 });
            eq(dst, [10, 20, 30, 40, 50, 60, 70, 80], "lookup f32");
            const dst2 = new Uint8Array(8);
            I.lookup(dst2, new Uint8Array([255, 0]), lut, { lo: 0, hi: 255, edge: "clamp" });
            eq(dst2, [50, 60, 70, 80, 10, 20, 30, 40], "lookup u8");
        }

        // ── reduce: every op, f32 and integer paths ─────────────────────────
        {
            const f = new Float32Array([0.5, 1.5, 2.5, 3.5]);
            if (I.reduce(f, "sum") !== 8) fail("reduce sum");
            if (I.reduce(f, "mean") !== 2) fail("reduce mean");
            eq(I.reduce(f, "histogram", { bins: 2, lo: 0, hi: 4 }), [2, 2], "reduce histogram f32");
            const u = new Uint8Array([9, 1, 7, 3]);
            const mm = I.reduce(u, "minmax", { stride: 1 });
            if (mm.min !== 1 || mm.max !== 9) fail("reduce minmax u8 gave " + mm.min + ".." + mm.max);
            if (I.reduce(u, "sum", { stride: 2 }) !== 16) fail("reduce sum u8 stride 2");
            eq(I.reduce(u, "histogram", { bins: 2, lo: 0, hi: 10 }), [2, 2], "reduce histogram u8");
        }

        // ── map: each op reads its options before running ───────────────────
        {
            const src = new Float32Array([1, 2]);
            const dst = new Float32Array(2);
            I.map(dst, src, { op: "affine", a: 2, b: 1 });
            eq(dst, [3, 5], "map affine");
            I.map(dst, src, { op: "affine", a: 2, b: 1, clamp: [0, 4] });
            eq(dst, [3, 4], "map affine clamp");
            I.map(dst, new Float32Array([-1, 4]), { op: "abs" });
            eq(dst, [1, 4], "map abs");
            I.map(dst, new Float32Array([4, 9]), { op: "sqrt" });
            eq(dst, [2, 3], "map sqrt");
            I.map(dst, new Float32Array([3, 2]), { op: "pow", exp: 2 });
            eq(dst, [9, 4], "map pow");
            let threw = false;
            try { I.map(dst, src, { op: "nope" }); } catch (e) { threw = true; }
            if (!threw) fail("map accepted an unknown op");
        }

        // ── combine ─────────────────────────────────────────────────────────
        {
            const a = new Float32Array([1, 2]), b = new Float32Array([3, 4]);
            const dst = new Float32Array(2);
            I.combine(dst, a, b, { op: "add" });          eq(dst, [4, 6], "combine add");
            I.combine(dst, a, b, { op: "sub" });          eq(dst, [-2, -2], "combine sub");
            I.combine(dst, a, b, { op: "max" });          eq(dst, [3, 4], "combine max");
            I.combine(dst, a, b, { op: "lerp", t: 0.5 }); eq(dst, [2, 3], "combine lerp");
            I.combine(dst, a, b, { op: "wsum", wa: 2, wb: 3 });
            eq(dst, [11, 16], "combine wsum");
            let threw = false;
            try { I.combine(dst, a, b, { op: "nope" }); } catch (e) { threw = true; }
            if (!threw) fail("combine accepted an unknown op");
        }

        // ── stencil: identity kernel, then a divisor ────────────────────────
        {
            const k = new Float32Array([0, 0, 0, 0, 1, 0, 0, 0, 0]);
            const src = new Float32Array([1, 2, 3, 4]);
            const dst = new Float32Array(4);
            I.stencil(dst, src, { data: k, w: 3, h: 3 }, { srcW: 2, srcH: 2, edge: "clamp" });
            eq(dst, [1, 2, 3, 4], "stencil identity");
            I.stencil(dst, src, { data: k, w: 3, h: 3 }, { srcW: 2, srcH: 2, divisor: 2, bias: 1 });
            eq(dst, [1.5, 2, 2.5, 3], "stencil divisor/bias");
        }

        // ── resample ────────────────────────────────────────────────────────
        {
            const dst = new Float32Array(4);
            I.resample(dst, new Float32Array([7]),
                       { srcW: 1, srcH: 1, dstW: 2, dstH: 2, channels: 1, filter: "nearest" });
            eq(dst, [7, 7, 7, 7], "resample nearest");
        }

        // ── geometry: crop / centerCrop / flips / rotate / pad ──────────────
        {
            const src = new Uint8Array([1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255, 4, 4, 4, 255]);
            const one = new Uint8Array(4);
            I.crop(one, src, { srcW: 2, srcH: 2, channels: 4, x: 1, y: 1, w: 1, h: 1 });
            eq(one, [4, 4, 4, 255], "crop");

            const nine = new Uint8Array([0, 1, 2, 3, 4, 5, 6, 7, 8]);
            const mid = new Uint8Array(1);
            I.centerCrop(mid, nine, { srcW: 3, srcH: 3, channels: 1, cropW: 1, cropH: 1 });
            eq(mid, [4], "centerCrop");

            const q = new Uint8Array([1, 2, 3, 4]);
            const f = new Uint8Array(4);
            I.flipVertical(f, q, { w: 2, h: 2, channels: 1 });
            eq(f, [3, 4, 1, 2], "flipVertical");

            const r = new Uint8Array(2);
            I.rotate90(r, new Uint8Array([1, 2]), { srcW: 2, srcH: 1, channels: 1, turns: 1 });
            eq(r, [2, 1], "rotate90");

            const p = new Uint8Array(8);
            I.pad(p, new Uint8Array([9, 9, 9, 9]), {
                srcW: 1, srcH: 1, dstW: 2, dstH: 1, channels: 4, offX: 1, offY: 0,
                padR: 1, padG: 2, padB: 3, padA: 4
            });
            eq(p, [1, 2, 3, 4, 9, 9, 9, 9], "pad");
        }

        // ── color ───────────────────────────────────────────────────────────
        {
            const rgb = new Uint8Array(3);
            I.rgbaToRgb(rgb, new Uint8Array([1, 2, 3, 4]), 1);
            eq(rgb, [1, 2, 3], "rgbaToRgb");
            const gray = new Uint8Array(1);
            I.rgbToGray(gray, new Uint8Array([255, 255, 255]), 1);
            eq(gray, [255], "rgbToGray");

            const lin = new Float32Array(2);
            I.srgbToLinear(lin, new Float32Array([0, 1]), 2);
            near(lin, [0, 1], "srgbToLinear f32");
            const lin8 = new Float32Array(1);
            I.srgbToLinear(lin8, new Uint8Array([255]), 1);
            near(lin8, [1], "srgbToLinear u8");
            const s8 = new Uint8Array(1);
            I.linearToSrgb(s8, new Float32Array([1]), 1);
            eq(s8, [255], "linearToSrgb u8");
            const sf = new Float32Array(1);
            I.linearToSrgb(sf, new Float32Array([1]), 1);
            near(sf, [1], "linearToSrgb f32");

            const g = new Float32Array(1);
            I.applyGamma(g, new Float32Array([4]), 1, 2);
            near(g, [2], "applyGamma");

            const px = new Float32Array([0.8, 0.4, 0.2]);
            const hsv = new Float32Array(3), back = new Float32Array(3);
            I.rgbToHsv(hsv, px, 1);
            I.hsvToRgb(back, hsv, 1);
            near(back, [0.8, 0.4, 0.2], "rgb -> hsv -> rgb");
            const hsl = new Float32Array(3);
            I.rgbToHsl(hsl, px, 1);
            I.hslToRgb(back, hsl, 1);
            near(back, [0.8, 0.4, 0.2], "rgb -> hsl -> rgb");
        }

        // ── layout ──────────────────────────────────────────────────────────
        {
            const y = new Float32Array(2);
            I.u8ToF32(y, new Uint8Array([2, 4]), { n: 1, h: 1, w: 2, c: 1, scale: 0.5, bias: 1 });
            near(y, [2, 3], "u8ToF32");
            const u = new Uint8Array(2);
            I.f32ToU8(u, new Float32Array([3, 4]), { n: 1, c: 1, h: 1, w: 2, scale: 2, bias: 0 });
            eq(u, [6, 8], "f32ToU8");
            const nchw = new Float32Array(4), nhwc = new Float32Array(4);
            I.nhwcToNchw(nchw, new Float32Array([1, 2, 3, 4]), { n: 1, h: 1, w: 2, c: 2 });
            eq(nchw, [1, 3, 2, 4], "nhwcToNchw");
            I.nchwToNhwc(nhwc, nchw, { n: 1, c: 2, h: 1, w: 2 });
            eq(nhwc, [1, 2, 3, 4], "nchwToNhwc");
        }

        // ── validation: every kernel refuses a buffer shorter than its dims ──
        {
            const throwsName = (fn, name, what) => {
                let err = null;
                try { fn(); } catch (e) { err = e; }
                if (!err) fail(what + " did not throw");
                if (err.name !== name) fail(what + " threw " + err.name + ": " + err.message);
            };
            const u8 = (n) => new Uint8Array(n), f32 = (n) => new Float32Array(n);

            // normalize: mean/std shorter than C, and short Y/X.
            const X = f32(6), Y = f32(6);
            throwsName(() => I.normalize(Y, X, [0, 0], [1, 1, 1], 1, 3, 1, 2), "TypeError", "normalize short mean");
            throwsName(() => I.normalize(Y, X, f32(3), f32(2), 1, 3, 1, 2), "TypeError", "normalize short std f32");
            throwsName(() => I.normalize(f32(5), X, [0, 0, 0], [1, 1, 1], 1, 3, 1, 2), "RangeError", "normalize short Y");
            throwsName(() => I.normalize(Y, new Int32Array(6), [0, 0, 0], [1, 1, 1], 1, 3, 1, 2), "TypeError", "normalize int X");
            throwsName(() => I.normalize(Y, X, [0, 0, 0], [1, 1, 1], 1, 0, 1, 2), "RangeError", "normalize C=0");
            I.normalize(Y, X, f32(3), new Float32Array([1, 1, 1]), 1, 3, 1, 2);

            // geometry, untyped and typed spellings, with and without strides.
            throwsName(() => I.resize(u8(15), u8(16), { srcW: 2, srcH: 2, dstW: 2, dstH: 2 }), "RangeError", "resize short dst");
            throwsName(() => I.resize(f32(16), f32(15), { srcW: 2, srcH: 2, dstW: 2, dstH: 2 }), "RangeError", "resize short f32 src");
            throwsName(() => I.resizeU8(u8(16), u8(16), { srcW: 2, srcH: 2, dstW: 2, dstH: 2, srcStride: 12 }), "RangeError", "resizeU8 stride past end");
            throwsName(() => I.resizeU8(u8(16), u8(16), { srcW: 2, srcH: 2, dstW: 2, dstH: 2, srcStride: 4 }), "RangeError", "resizeU8 stride shorter than a row");
            I.resizeU8(u8(16), u8(20), { srcW: 2, srcH: 2, dstW: 2, dstH: 2, srcStride: 12 });
            throwsName(() => I.crop(u8(3), u8(16), { srcW: 2, srcH: 2, x: 0, y: 0, w: 1, h: 1 }), "RangeError", "crop short dst");
            throwsName(() => I.cropU8(u8(4), u8(15), { srcW: 2, srcH: 2, x: 0, y: 0, w: 1, h: 1 }), "RangeError", "cropU8 short src");
            throwsName(() => I.centerCrop(u8(15), u8(64), { srcW: 4, srcH: 4, cropW: 2, cropH: 2 }), "RangeError", "centerCrop short dst");
            throwsName(() => I.flipHorizontal(u8(15), u8(16), { w: 2, h: 2 }), "RangeError", "flipHorizontal short dst");
            throwsName(() => I.flipVerticalU8(u8(16), u8(15), { w: 2, h: 2 }), "RangeError", "flipVerticalU8 short src");
            throwsName(() => I.rotate90(u8(23), u8(24), { srcW: 3, srcH: 2, turns: 1 }), "RangeError", "rotate90 short dst");
            I.rotate90(u8(24), u8(24), { srcW: 3, srcH: 2, turns: 1 });
            throwsName(() => I.pad(u8(35), u8(16), { srcW: 2, srcH: 2, dstW: 3, dstH: 3 }), "RangeError", "pad short dst");
            throwsName(() => I.padU8(u8(36), u8(15), { srcW: 2, srcH: 2, dstW: 3, dstH: 3 }), "RangeError", "padU8 short src");
            throwsName(() => I.letterboxU8(u8(15), u8(16), { srcW: 2, srcH: 2, dstW: 2, dstH: 2 }), "RangeError", "letterboxU8 short dst");
            throwsName(() => I.resizeRgba8Alpha(u8(16), u8(15), { srcW: 2, srcH: 2, dstW: 2, dstH: 2 }), "RangeError", "resizeRgba8Alpha short src");
            throwsName(() => I.resizeF32(f32(4), new Int32Array(4), { srcW: 2, srcH: 2, dstW: 2, dstH: 2 }), "TypeError", "resizeF32 int src");

            // colour converters take their count from the argument, so the
            // buffers are checked against it.
            throwsName(() => I.rgbaToRgb(u8(5), u8(8), 2), "RangeError", "rgbaToRgb short dst");
            throwsName(() => I.rgbToRgba(u8(8), u8(5), 2), "RangeError", "rgbToRgba short src");
            throwsName(() => I.rgbaToGray(u8(1), u8(8), 2), "RangeError", "rgbaToGray short dst");
            throwsName(() => I.rgbToGray(u8(2), u8(5), 2), "RangeError", "rgbToGray short src");
            throwsName(() => I.srgbToLinear(f32(3), f32(4), 4), "RangeError", "srgbToLinear short dst");
            throwsName(() => I.srgbToLinear(f32(4), u8(3), 4), "RangeError", "srgbToLinear u8 short src");
            throwsName(() => I.linearToSrgb(u8(3), f32(4), 4), "RangeError", "linearToSrgb u8 short dst");
            throwsName(() => I.applyGamma(f32(4), f32(4), 5, 2.2), "RangeError", "applyGamma count past end");
            throwsName(() => I.rgbToHsv(f32(6), f32(5), 2), "RangeError", "rgbToHsv short src");
            throwsName(() => I.hslToRgb(f32(5), f32(6), 2), "RangeError", "hslToRgb short dst");
            throwsName(() => I.rgbaToRgb(u8(3), u8(4), 0), "RangeError", "rgbaToRgb zero count");

            // layout
            throwsName(() => I.u8ToF32(f32(4), u8(3), { h: 1, w: 4, c: 1 }), "RangeError", "u8ToF32 short src");
            throwsName(() => I.f32ToU8(u8(3), f32(4), { c: 1, h: 1, w: 4 }), "RangeError", "f32ToU8 short dst");
            throwsName(() => I.nhwcToNchw(f32(4), f32(3), { h: 1, w: 2, c: 2 }), "RangeError", "nhwcToNchw short src");
            throwsName(() => I.nchwToNhwcF32(f32(4), f32(3), { C: 2, H: 1, W: 2 }), "RangeError", "nchwToNhwcF32 short src");
            throwsName(() => I.u8NhwcToF32Nchw(f32(4), u8(3), { H: 1, W: 4, C: 1 }), "RangeError", "u8NhwcToF32Nchw short src");

            // encoders read h rows of w*channels bytes.
            throwsName(() => I.encodePng(u8(15), 2, 2, 4), "RangeError", "encodePng short pixels");
            throwsName(() => I.encodeJpeg(u8(11), 2, 2, 3, 90), "RangeError", "encodeJpeg short pixels");
            throwsName(() => I.encodePng(u8(16), 2, 2, 5), "RangeError", "encodePng 5 channels");
            throwsName(() => I.applyExifOrientation(u8(15), 2, 2, 6), "RangeError", "applyExifOrientation short pixels");

            // options of the wrong type throw instead of silently doing nothing.
            throwsName(() => I.crop(u8(4), u8(16), { srcW: "2", srcH: 2, w: 1, h: 1 }), "TypeError", "crop string srcW");
            throwsName(() => I.crop(u8(4), u8(16), { srcW: 1e12, srcH: 2, w: 1, h: 1 }), "RangeError", "crop huge srcW");
        }

        "SUCCESS";
    )JS";

    auto res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "  ops script threw: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    if (ev::toUtf8(res.value) != "SUCCESS") {
        std::cerr << "  ops script returned: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    std::cout << "  bro.image ops kernels OK." << std::endl;
}
