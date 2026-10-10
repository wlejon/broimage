// bro.image.decodeFrames / openFrames: an animated GIF (written by
// gif_writer.h) frame by frame from script — sizes, loop count, delays,
// exact pixels, the caps, stepping, reset and close.

#include "embed/embed.h"
#include "eval/eval.h"

#include "gif_writer.h"

#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ev = bronze::embed;

void fail(const std::string& what) {
    std::cerr << "  frames check failed: " << what << std::endl;
    std::exit(1);
}

} // namespace

void broimageTestFrames() {
    std::cout << "Checking bro.image.decodeFrames / openFrames..." << std::endl;
    // 4 x 2, three frames: red, green over the left half (disposal 1), blue.
    const std::vector<uint8_t> gct = {255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0};
    std::vector<gifw::Frame> frames(3);
    frames[0].w = 4; frames[0].h = 2; frames[0].idx.assign(8, 0); frames[0].delay_cs = 7; frames[0].disposal = 1;
    frames[1].w = 2; frames[1].h = 2; frames[1].idx.assign(4, 1); frames[1].delay_cs = 1; frames[1].disposal = 1;
    frames[2].x = 3; frames[2].w = 1; frames[2].h = 2; frames[2].idx.assign(2, 2); frames[2].delay_cs = 30;
    const std::vector<uint8_t> gif = gifw::make_gif(4, 2, gct, 0, frames);

    ev::Persistent arr(ev::createTypedArray(ev::elements::Uint8, uint32_t(gif.size())));
    ev::fillTypedArray(arr.get(), std::span<const uint8_t>(gif.data(), gif.size()));
    ev::registerGlobal("__gif", arr.get());

    const char* script = R"JS(
        (function() {
            const px = (p, x, y) => Array.from(p.subarray((y * 4 + x) * 4, (y * 4 + x) * 4 + 4)).join(',');
            const all = bro.image.decodeFrames(__gif);
            if (!all || all.width !== 4 || all.height !== 2) return 'size';
            if (all.frameCount !== 3 || all.loopCount !== 0 || all.truncated) return 'counts';
            if (all.frames.map(f => f.delay).join() !== '70,10,300') return 'delays ' + all.frames.map(f => f.delay).join();
            if (px(all.frames[0].pixels, 3, 1) !== '255,0,0,255') return 'f0';
            if (px(all.frames[1].pixels, 1, 1) !== '0,255,0,255' || px(all.frames[1].pixels, 2, 0) !== '255,0,0,255') return 'f1';
            if (px(all.frames[2].pixels, 3, 0) !== '0,0,255,255' || px(all.frames[2].pixels, 0, 0) !== '0,255,0,255') return 'f2';
            const capped = bro.image.decodeFrames(__gif, { maxFrames: 2 });
            if (capped.frames.length !== 2 || !capped.truncated) return 'maxFrames';
            const byBytes = bro.image.decodeFrames(__gif, { maxBytes: 4 * 2 * 4 });
            if (byBytes.frames.length !== 1 || !byBytes.truncated) return 'maxBytes';

            const dec = bro.image.openFrames(__gif);
            if (!dec || dec.frameCount !== 3 || dec.delays.join() !== '70,10,300') return 'open';
            const seen = [];
            for (let f; (f = dec.next()); ) seen.push(f.index + ':' + px(f.pixels, 3, 0));
            if (seen.join(' ') !== '0:255,0,0,255 1:255,0,0,255 2:0,0,255,255') return 'step ' + seen.join(' ');
            if (dec.index !== 3 || dec.next() !== null) return 'end';
            if (!dec.reset() || dec.index !== 0 || dec.next().index !== 0) return 'reset';
            dec.close();
            if (dec.next() !== null) return 'close';
            if (bro.image.openFrames(new Uint8Array([1, 2, 3])) !== null) return 'junk';
            return 'ok';
        })()
    )JS";
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) fail("threw: " + ev::toUtf8(res.value));
    const std::string got = ev::toUtf8(res.value);
    if (got != "ok") fail(got);
    std::cout << "  frames OK" << std::endl;
}
