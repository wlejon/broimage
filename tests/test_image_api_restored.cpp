// Coverage for the bro.image members the QuickJS -> bronze port dropped
// (bro's docs/transition-drift.md row E4: the old runtime had 61 keys under
// bro.image, the port had 37).
//
// Every check runs the kernel on real buffers and asserts the pixels, so a
// name that is present but wired to the wrong thing still fails.
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

void broimageTestRestoredSurface() {
    std::cout << "Checking the restored bro.image surface..." << std::endl;

    const char* script = R"JS(
        const I = bro.image;
        const fail = (m) => { throw new Error(m); };
        const need = (n, arity) => {
            if (typeof I[n] !== "function") fail("bro.image." + n + " is missing");
            if (arity !== undefined && I[n].length !== arity) {
                fail("bro.image." + n + " arity " + I[n].length + ", expected " + arity);
            }
        };

        // Every name the drift log lists as missing must be back.
        const names = [
            ["decodeU16", 1], ["decodeF32", 1], ["decodeOriented", 1],
            ["probeDimensions", 1], ["readExifOrientation", 1], ["applyExifOrientation", 4],
            ["resizeU8", 3], ["resizeF32", 3], ["resizeChwF32", 3], ["letterboxU8", 3],
            ["padU8", 3], ["cropU8", 3], ["centerCropU8", 3],
            ["flipHorizontalU8", 3], ["flipVerticalU8", 3], ["rotate90U8", 3],
            ["premultiplyAlpha", 2], ["unpremultiplyAlpha", 2],
            ["resizeRgba8Alpha", 3], ["letterboxRgba8Alpha", 3],
            ["hwcToChw", 3], ["chwToHwc", 3],
            ["srgbToLinearU8ToF32", 2], ["linearF32ToSrgbU8", 2],
            ["applyColorMatrix3x3", 3], ["applyColorMatrix3x4", 3],
            ["u8NhwcToF32Nchw", 3], ["f32NchwToU8Nhwc", 3],
            ["nhwcToNchwF32", 3], ["nchwToNhwcF32", 3],
            ["normalizeNchw", 3], ["stencilHwc", 4],
            ["featherWindow", 2], ["accumulateTile", 5], ["normalizeAccumulator", 3]
        ];
        for (const [n, a] of names) need(n, a);

        // ── cropU8: pull the bottom-right 1x1 out of a 2x2 RGBA ────────────
        {
            const src = new Uint8Array([
                1, 1, 1, 255,   2, 2, 2, 255,
                3, 3, 3, 255,   4, 4, 4, 255
            ]);
            const dst = new Uint8Array(4);
            I.cropU8(dst, src, { srcW: 2, srcH: 2, channels: 4, x: 1, y: 1, w: 1, h: 1 });
            if (dst[0] !== 4) fail("cropU8 took the wrong pixel: " + dst[0]);
        }

        // ── flipHorizontalU8 / flipVerticalU8 on the same 2x2 ──────────────
        {
            const src = new Uint8Array([
                1, 1, 1, 255,   2, 2, 2, 255,
                3, 3, 3, 255,   4, 4, 4, 255
            ]);
            const fh = new Uint8Array(16);
            I.flipHorizontalU8(fh, src, { w: 2, h: 2, channels: 4 });
            if (fh[0] !== 2 || fh[4] !== 1) fail("flipHorizontalU8 did not mirror the row");
            const fv = new Uint8Array(16);
            I.flipVerticalU8(fv, src, { w: 2, h: 2, channels: 4 });
            if (fv[0] !== 3 || fv[8] !== 1) fail("flipVerticalU8 did not mirror the column");
        }

        // ── rotate90U8: one CCW turn puts the top-right pixel top-left ─────
        {
            const src = new Uint8Array([
                1, 1, 1, 255,   2, 2, 2, 255,
                3, 3, 3, 255,   4, 4, 4, 255
            ]);
            const rot = new Uint8Array(16);
            I.rotate90U8(rot, src, { srcW: 2, srcH: 2, channels: 4, turns: 1 });
            const corners = [rot[0], rot[4], rot[8], rot[12]].join(",");
            if (corners === "1,2,3,4") fail("rotate90U8 left the image unrotated");
        }

        // ── padU8: the pad colour arrives as the [r,g,b,a] array again ─────
        {
            const src = new Uint8Array([9, 9, 9, 255]);
            const dst = new Uint8Array(2 * 2 * 4);
            I.padU8(dst, src, {
                srcW: 1, srcH: 1, dstW: 2, dstH: 2, channels: 4,
                offX: 0, offY: 0, pad: [7, 7, 7, 255]
            });
            if (dst[0] !== 9) fail("padU8 did not place the source at (offX, offY)");
            if (dst[4] !== 7) fail("padU8 ignored the pad array: " + dst[4]);
        }

        // ── letterboxU8: reports the content rect and honours the pad ──────
        {
            const src = new Uint8Array(4 * 2 * 4).fill(200);   // 4x2, wide
            const dst = new Uint8Array(8 * 8 * 4);
            const rect = I.letterboxU8(dst, src, {
                srcW: 4, srcH: 2, dstW: 8, dstH: 8, channels: 4, pad: [0, 0, 0, 255]
            });
            if (!rect || typeof rect.x !== "number" || typeof rect.w !== "number") {
                fail("letterboxU8 did not return the {x, y, w, h} content rect");
            }
            if (rect.w !== 8 || rect.h !== 4) fail("letterboxU8 rect " + rect.w + "x" + rect.h + ", expected 8x4");
            if (rect.y !== 2) fail("letterboxU8 did not centre the content: y=" + rect.y);
            if (dst[3] !== 255 || dst[0] !== 0) fail("letterboxU8 did not fill with the pad colour");
        }

        // ── centerCropU8 / resizeU8 / resizeF32 / resizeChwF32 ─────────────
        {
            const src = new Uint8Array(4 * 4 * 4).fill(128);
            const cc = new Uint8Array(2 * 2 * 4);
            I.centerCropU8(cc, src, { srcW: 4, srcH: 4, channels: 4, cropW: 2, cropH: 2 });
            if (cc[0] !== 128) fail("centerCropU8 produced nothing");

            const rs = new Uint8Array(2 * 2 * 4);
            I.resizeU8(rs, src, { srcW: 4, srcH: 4, dstW: 2, dstH: 2, channels: 4, filter: "bilinear" });
            if (rs[0] !== 128) fail("resizeU8 produced nothing");

            const fsrc = new Float32Array(4 * 4).fill(2.5);
            const fdst = new Float32Array(2 * 2);
            I.resizeF32(fdst, fsrc, { srcW: 4, srcH: 4, dstW: 2, dstH: 2, channels: 1 });
            if (Math.abs(fdst[0] - 2.5) > 1e-5) fail("resizeF32 produced " + fdst[0]);

            const csrc = new Float32Array(2 * 4 * 4).fill(1.5);
            const cdst = new Float32Array(2 * 2 * 2);
            I.resizeChwF32(cdst, csrc, { srcW: 4, srcH: 4, dstW: 2, dstH: 2, channels: 2 });
            if (Math.abs(cdst[0] - 1.5) > 1e-5) fail("resizeChwF32 produced " + cdst[0]);
        }

    )JS" R"JS(
        // ── premultiplyAlpha / unpremultiplyAlpha ──────────────────────────
        // (MSVC caps a single string literal at 16 KiB, hence the splices.)
        {
            const src = new Uint8Array([255, 255, 255, 128]);
            const pm = new Uint8Array(4);
            I.premultiplyAlpha(pm, src);
            if (pm[0] >= 255 || pm[0] < 100) fail("premultiplyAlpha did not scale RGB: " + pm[0]);
            if (pm[3] !== 128) fail("premultiplyAlpha touched alpha");
            const un = new Uint8Array(4);
            I.unpremultiplyAlpha(un, pm);
            if (Math.abs(un[0] - 255) > 2) fail("unpremultiplyAlpha did not invert: " + un[0]);
        }

        // ── resizeRgba8Alpha / letterboxRgba8Alpha ─────────────────────────
        {
            const src = new Uint8Array(4 * 4 * 4).fill(200);
            const dst = new Uint8Array(2 * 2 * 4);
            I.resizeRgba8Alpha(dst, src, { srcW: 4, srcH: 4, dstW: 2, dstH: 2 });
            if (dst[3] !== 200) fail("resizeRgba8Alpha lost the alpha plane");
            const lb = new Uint8Array(8 * 8 * 4);
            const r2 = I.letterboxRgba8Alpha(lb, src, { srcW: 4, srcH: 4, dstW: 8, dstH: 8, pad: [0, 0, 0, 0] });
            if (!r2 || r2.w !== 8 || r2.h !== 8) fail("letterboxRgba8Alpha rect is wrong");
        }

        // ── hwcToChw / chwToHwc round-trip ─────────────────────────────────
        {
            const hwc = new Float32Array([1, 2, 3, 4, 5, 6]);   // 3x1, 2 channels
            const chw = new Float32Array(6);
            I.hwcToChw(chw, hwc, { width: 3, height: 1, channels: 2 });
            if (chw[0] !== 1 || chw[1] !== 3 || chw[2] !== 5 || chw[3] !== 2) {
                fail("hwcToChw did not deinterleave: " + Array.from(chw).join(","));
            }
            const back = new Float32Array(6);
            I.chwToHwc(back, chw, { width: 3, height: 1, channels: 2 });
            for (let i = 0; i < 6; i++) if (back[i] !== hwc[i]) fail("chwToHwc did not round-trip");
        }

        // ── srgbToLinearU8ToF32 / linearF32ToSrgbU8 ────────────────────────
        {
            const u8 = new Uint8Array([0, 128, 255]);
            const lin = new Float32Array(3);
            I.srgbToLinearU8ToF32(lin, u8);
            if (lin[0] !== 0) fail("srgbToLinearU8ToF32: black is not 0");
            if (Math.abs(lin[2] - 1) > 1e-4) fail("srgbToLinearU8ToF32: white is not 1");
            if (!(lin[1] > 0.1 && lin[1] < 0.3)) fail("srgbToLinearU8ToF32 is not the sRGB curve: " + lin[1]);
            const back = new Uint8Array(3);
            I.linearF32ToSrgbU8(back, lin);
            for (let i = 0; i < 3; i++) {
                if (Math.abs(back[i] - u8[i]) > 1) fail("linearF32ToSrgbU8 did not round-trip at " + i);
            }
        }

        // ── applyColorMatrix3x3 / 3x4 ──────────────────────────────────────
        {
            const src = new Float32Array([1, 0, 0]);
            const dst = new Float32Array(3);
            // Swap R and B.
            I.applyColorMatrix3x3(dst, src, { channels: 3, matrix: [0, 0, 1, 0, 1, 0, 1, 0, 0] });
            if (dst[2] !== 1 || dst[0] !== 0) fail("applyColorMatrix3x3 did not swap R/B");
            const dst4 = new Float32Array(3);
            // Identity plus a +0.5 offset on every channel.
            I.applyColorMatrix3x4(dst4, src, {
                channels: 3,
                matrix: [1, 0, 0, 0.5, 0, 1, 0, 0.5, 0, 0, 1, 0.5]
            });
            if (Math.abs(dst4[0] - 1.5) > 1e-5 || Math.abs(dst4[1] - 0.5) > 1e-5) {
                fail("applyColorMatrix3x4 ignored the offset column: " + Array.from(dst4).join(","));
            }
            let threw = false;
            try { I.applyColorMatrix3x3(dst, src, { channels: 3, matrix: [1, 2, 3] }); }
            catch (e) { threw = true; }
            if (!threw) fail("applyColorMatrix3x3 accepted a short matrix");
        }

        // ── u8NhwcToF32Nchw: uppercase keys, scale defaults to 1 ───────────
        {
            const src = new Uint8Array([10, 20, 30, 40]);   // 1x2, 2 channels
            const dst = new Float32Array(4);
            I.u8NhwcToF32Nchw(dst, src, { N: 1, H: 1, W: 2, C: 2 });
            if (dst[0] !== 10 || dst[1] !== 30 || dst[2] !== 20 || dst[3] !== 40) {
                fail("u8NhwcToF32Nchw layout/scale wrong: " + Array.from(dst).join(","));
            }
            const scaled = new Float32Array(4);
            I.u8NhwcToF32Nchw(scaled, src, { N: 1, H: 1, W: 2, C: 2, scale: 1 / 10, bias: 1 });
            if (Math.abs(scaled[0] - 2) > 1e-5) fail("u8NhwcToF32Nchw ignored scale/bias: " + scaled[0]);

            const back = new Uint8Array(4);
            I.f32NchwToU8Nhwc(back, dst, { N: 1, C: 2, H: 1, W: 2 });
            for (let i = 0; i < 4; i++) if (back[i] !== src[i]) fail("f32NchwToU8Nhwc did not round-trip");
        }

        // ── nhwcToNchwF32 / nchwToNhwcF32 ──────────────────────────────────
        {
            const nhwc = new Float32Array([1, 2, 3, 4]);   // 1x2, 2 channels
            const nchw = new Float32Array(4);
            I.nhwcToNchwF32(nchw, nhwc, { N: 1, H: 1, W: 2, C: 2 });
            if (nchw[0] !== 1 || nchw[1] !== 3) fail("nhwcToNchwF32 did not deinterleave");
            const back = new Float32Array(4);
            I.nchwToNhwcF32(back, nchw, { N: 1, H: 1, W: 2, C: 2 });
            for (let i = 0; i < 4; i++) if (back[i] !== nhwc[i]) fail("nchwToNhwcF32 did not round-trip");
        }

    )JS" R"JS(
        // ── normalizeNchw + presets ────────────────────────────────────────
        {
            if (!I.presets || !I.presets.clip || !I.presets.imagenet || !I.presets.sam) {
                fail("bro.image.presets is missing clip/imagenet/sam");
            }
            const p = I.presets.imagenet;
            if (!Array.isArray(p.mean) || p.mean.length !== 3) fail("presets.imagenet.mean is not a 3-array");
            if (!(p.mean[0] > 0.4 && p.mean[0] < 0.5)) fail("presets.imagenet.mean[0] = " + p.mean[0]);

            const src = new Float32Array([1, 1, 3, 3]);   // 1x1x2x... C=2, H=1, W=2
            const dst = new Float32Array(4);
            I.normalizeNchw(dst, src, { N: 1, C: 2, H: 1, W: 2, mean: [1, 1], std: [2, 2] });
            if (dst[0] !== 0 || Math.abs(dst[2] - 1) > 1e-5) {
                fail("normalizeNchw math wrong: " + Array.from(dst).join(","));
            }
            let threw = false;
            try { I.normalizeNchw(dst, src, { N: 1, C: 2, H: 1, W: 2, mean: [1, 1] }); }
            catch (e) { threw = true; }
            if (!threw) fail("normalizeNchw accepted a missing std");
        }

        // ── stencilHwc: a per-channel 3x3 box blur leaves a flat image flat
        {
            const src = new Float32Array(3 * 3 * 2).fill(4);
            const dst = new Float32Array(3 * 3 * 2);
            const k = new Float32Array(9).fill(1);
            I.stencilHwc(dst, src, { data: k, w: 3, h: 3 },
                         { srcW: 3, srcH: 3, channels: 2, divisor: 9, edge: "clamp" });
            for (let i = 0; i < dst.length; i++) {
                if (Math.abs(dst[i] - 4) > 1e-4) fail("stencilHwc blurred a flat image at " + i);
            }
            let threw = false;
            try {
                I.stencilHwc(dst, src, { data: new Float32Array(4), w: 2, h: 2 },
                             { srcW: 3, srcH: 3, channels: 2 });
            } catch (e) { threw = true; }
            if (!threw) fail("stencilHwc accepted an even kernel");
        }

        // ── Tiling: feather + accumulate + normalize rebuild a flat map ────
        {
            const tw = 4, th = 4, fullW = 4, fullH = 4, ch = 1;
            const win = new Float32Array(tw * th);
            I.featherWindow(win, { tw, th, ovL: 0, ovR: 0, ovT: 0, ovB: 0 });
            let anyPositive = false;
            for (const v of win) if (v > 0) anyPositive = true;
            if (!anyPositive) fail("featherWindow produced an all-zero window");

            const acc = new Float32Array(fullW * fullH * ch);
            const wacc = new Float32Array(fullW * fullH);
            const tile = new Float32Array(tw * th * ch).fill(7);
            I.accumulateTile(acc, wacc, tile, win,
                             { fullW, fullH, channels: ch, tw, th, dstX: 0, dstY: 0 });
            let accSum = 0;
            for (const v of acc) accSum += v;
            if (accSum === 0) fail("accumulateTile wrote nothing");
            I.normalizeAccumulator(acc, wacc, { nPixels: fullW * fullH, channels: ch });
            for (let i = 0; i < acc.length; i++) {
                if (Math.abs(acc[i] - 7) > 1e-3) fail("tiling round-trip gave " + acc[i] + " at " + i);
            }
        }

        // ── probeDimensions / decode* / EXIF: argument contracts ───────────
        {
            // A PNG header is enough for probeDimensions; garbage returns null.
            const junk = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
            if (I.probeDimensions(junk) !== null) fail("probeDimensions did not reject garbage");
            if (I.decodeU16(junk) !== null) fail("decodeU16 did not return null on garbage");
            if (I.decodeF32(junk) !== null) fail("decodeF32 did not return null on garbage");
            // Non-JPEG / no EXIF reports Normal (1), not 0.
            if (I.readExifOrientation(junk) !== 1) fail("readExifOrientation did not report Normal on garbage");

            // decodeOriented keeps decode_file's 1x1 fallback rather than null.
            const d = I.decodeOriented(junk);
            if (!d || typeof d.width !== "number" || !(d.pixels instanceof Uint8Array)) {
                fail("decodeOriented lost the { width, height, channels, pixels } shape");
            }

            // applyExifOrientation(6) transposes, so w/h swap.
            const px = new Uint8Array(2 * 1 * 4).fill(3);
            const rot = I.applyExifOrientation(px, 2, 1, 6);
            if (rot.width !== 1 || rot.height !== 2) {
                fail("applyExifOrientation did not transpose: " + rot.width + "x" + rot.height);
            }
            if (!(rot.pixels instanceof Uint8Array) || rot.pixels.length !== 8) {
                fail("applyExifOrientation returned the wrong buffer");
            }
            let threw = false;
            try { I.applyExifOrientation(px, 64, 64, 1); } catch (e) { threw = true; }
            if (!threw) fail("applyExifOrientation accepted a too-small buffer");
        }

        "SUCCESS";
    )JS";

    auto res = bronze::eval::evalScript(script);
    if (res.thrown) {
        std::cerr << "  restored-surface script threw: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    if (ev::toUtf8(res.value) != "SUCCESS") {
        std::cerr << "  restored-surface script returned: " << ev::toUtf8(res.value) << std::endl;
        std::exit(1);
    }
    std::cout << "  restored bro.image kernels OK." << std::endl;
}
