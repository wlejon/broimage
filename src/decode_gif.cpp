// GIF, a frame at a time (decode_gif.h).

#include "decode_gif.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <utility>

namespace broimage::detail {

namespace {

// A canvas larger than this is refused before it is allocated (a GIF's
// screen is two 16-bit numbers: 65535^2 * 4 bytes would be 17 GB).
constexpr uint64_t kMaxCanvasPixels = uint64_t(1) << 28;

uint16_t le16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }

// Skip a run of data sub-blocks (size byte, that many bytes, ... a zero size).
// False when the buffer ends first.
bool skip_sub_blocks(const uint8_t* d, std::size_t n, std::size_t& p) {
    while (p < n) {
        const uint8_t len = d[p++];
        if (len == 0) return true;
        p += len;
    }
    return false;
}

struct Gce {
    int disposal = 0;
    int delay_ms = 0;
    int transparent = -1;
};

Gce read_gce(const uint8_t* d, std::size_t n, std::size_t p) {
    // p at the block size byte (4).
    Gce g;
    if (p + 5 <= n && d[p] >= 4) {
        const uint8_t flags = d[p + 1];
        g.disposal = (flags >> 2) & 7;
        g.delay_ms = le16(d + p + 2) * 10;
        if (flags & 1) g.transparent = d[p + 4];
    }
    return g;
}

// Reads LZW codes across the sub-blocks of one image.
class CodeReader {
public:
    CodeReader(const uint8_t* d, std::size_t n, std::size_t p) : d_(d), n_(n), p_(p) {}

    // -1: the data ended (block terminator); -2: the buffer ended (truncated).
    int read(int bits) {
        while (nbits_ < bits) {
            if (left_ == 0) {
                if (ended_) return -1;
                if (p_ >= n_) return -2;
                left_ = d_[p_++];
                if (left_ == 0) {
                    ended_ = true;
                    return -1;
                }
            }
            if (p_ >= n_) return -2;
            acc_ |= uint32_t(d_[p_++]) << nbits_;
            nbits_ += 8;
            --left_;
        }
        const int code = int(acc_ & ((1u << bits) - 1));
        acc_ >>= bits;
        nbits_ -= bits;
        return code;
    }

    // Past the rest of the image data. False when the buffer ends first.
    bool finish(std::size_t& p) {
        p_ += left_;
        left_ = 0;
        if (!ended_ && !skip_sub_blocks(d_, n_, p_)) return false;
        if (p_ > n_) return false;
        p = p_;
        return true;
    }

private:
    const uint8_t* d_;
    std::size_t n_;
    std::size_t p_;
    uint32_t acc_ = 0;
    int nbits_ = 0;
    std::size_t left_ = 0;
    bool ended_ = false;
};

// Decode one image's LZW data (p at its minimum code size byte) into `out`
// (up to out.size() indices). `produced` counts the indices written; a
// stream that ends early leaves the rest. False on a truncated buffer or a
// malformed stream; `p` is left after the image's data.
bool lzw_decode(const uint8_t* d, std::size_t n, std::size_t& p, std::vector<uint8_t>& out,
                std::size_t& produced) {
    produced = 0;
    if (p >= n) return false;
    const int min_size = d[p++];
    if (min_size < 1 || min_size > 11) return false;
    CodeReader in(d, n, p);
    const int clear = 1 << min_size;
    const int eoi = clear + 1;
    std::array<uint16_t, 4096> prefix{};
    std::array<uint8_t, 4096> suffix{};
    std::array<uint8_t, 4096> first{};
    std::array<uint8_t, 4097> stack{};
    for (int i = 0; i < clear; ++i) {
        suffix[std::size_t(i)] = uint8_t(i);
        first[std::size_t(i)] = uint8_t(i);
    }
    int size = min_size + 1;
    int next = clear + 2;
    int prev = -1;
    const std::size_t want = out.size();
    bool bad = false;
    for (;;) {
        const int code = in.read(size);
        if (code == -2) return false;
        if (code == -1 || code == eoi) break;
        if (code == clear) {
            size = min_size + 1;
            next = clear + 2;
            prev = -1;
            continue;
        }
        int cur;
        if (prev < 0) {
            if (code >= clear) { bad = true; break; }
            cur = code;
        } else if (code < next) {
            cur = code;
            if (next < 4096) {
                prefix[std::size_t(next)] = uint16_t(prev);
                suffix[std::size_t(next)] = first[std::size_t(code)];
                first[std::size_t(next)] = first[std::size_t(prev)];
                ++next;
            }
        } else if (code == next && next < 4096) {
            prefix[std::size_t(next)] = uint16_t(prev);
            suffix[std::size_t(next)] = first[std::size_t(prev)];
            first[std::size_t(next)] = first[std::size_t(prev)];
            cur = next++;
        } else {
            bad = true;
            break;
        }
        // Emit cur's string.
        std::size_t sp = 0;
        for (int c = cur; ; c = prefix[std::size_t(c)]) {
            stack[sp++] = suffix[std::size_t(c)];
            if (c < clear) break;
        }
        while (sp > 0 && produced < want) out[produced++] = stack[--sp];
        if (next == (1 << size) && size < 12) ++size;
        prev = cur;
    }
    (void)bad;  // a malformed code ends the image like an early end of data
    return in.finish(p);
}

class GifDecoder final : public FrameDecoder {
public:
    GifDecoder(const uint8_t* d, std::size_t n, SharedBytes keep, GifScan scan, std::size_t first_block,
               std::vector<uint8_t> gct)
        : keep_(std::move(keep)), d_(d), n_(n), scan_(std::move(scan)), first_block_(first_block),
          gct_(std::move(gct)) {
        canvas_.assign(std::size_t(scan_.width) * scan_.height * 4, 0);
        p_ = first_block_;
    }

    int width() const override { return scan_.width; }
    int height() const override { return scan_.height; }
    int frame_count() const override { return scan_.frames; }
    int loop_count() const override { return scan_.loop_count; }
    const std::vector<int>& delays_ms() const override { return scan_.delays_ms; }
    int index() const override { return index_; }

    bool rewind() override {
        std::fill(canvas_.begin(), canvas_.end(), uint8_t(0));
        p_ = first_block_;
        index_ = 0;
        failed_ = false;
        have_prev_ = false;
        return true;
    }

    bool next(AnimationFrame& out, std::string* error) override {
        if (error) error->clear();
        if (failed_) return fail(error, "the GIF failed to decode");
        Gce gce;
        while (p_ < n_) {
            const uint8_t block = d_[p_++];
            if (block == 0x3B) break;
            if (block == 0x21) {
                if (p_ >= n_) break;
                const uint8_t label = d_[p_++];
                if (label == 0xF9) gce = read_gce(d_, n_, p_);
                if (!skip_sub_blocks(d_, n_, p_)) return fail(error, "the GIF is truncated");
                continue;
            }
            if (block != 0x2C) break;  // not a block: the stream ends here
            if (!draw_image(gce, error)) return false;
            ++index_;
            out.rgba.assign(canvas_.begin(), canvas_.end());
            out.delay_ms = gce.delay_ms;
            return true;
        }
        p_ = n_;
        return false;  // past the last frame
    }

private:
    bool fail(std::string* error, const char* why) {
        failed_ = true;
        if (error) *error = why;
        return false;
    }

    // The previous frame's disposal, before this one draws.
    void dispose_previous() {
        if (!have_prev_) return;
        have_prev_ = false;
        const int W = scan_.width;
        if (prev_disposal_ == 2) {
            for (int y = prev_y0_; y < prev_y1_; ++y)
                std::memset(&canvas_[(std::size_t(y) * W + prev_x0_) * 4], 0, std::size_t(prev_x1_ - prev_x0_) * 4);
        } else if (prev_disposal_ == 3 && !saved_.empty()) {
            const std::size_t row = std::size_t(prev_x1_ - prev_x0_) * 4;
            for (int y = prev_y0_; y < prev_y1_; ++y)
                std::memcpy(&canvas_[(std::size_t(y) * W + prev_x0_) * 4], &saved_[std::size_t(y - prev_y0_) * row],
                            row);
        }
    }

    bool draw_image(const Gce& gce, std::string* error) {
        if (p_ + 9 > n_) return fail(error, "the GIF is truncated");
        const int fx = le16(d_ + p_), fy = le16(d_ + p_ + 2);
        const int fw = le16(d_ + p_ + 4), fh = le16(d_ + p_ + 6);
        const uint8_t flags = d_[p_ + 8];
        p_ += 9;
        std::vector<uint8_t> lct;
        const std::vector<uint8_t>* palette = &gct_;
        if (flags & 0x80) {
            const std::size_t bytes = 3u * (1u << ((flags & 7) + 1));
            if (p_ + bytes > n_) return fail(error, "the GIF is truncated");
            lct.assign(d_ + p_, d_ + p_ + bytes);
            p_ += bytes;
            palette = &lct;
        }
        const bool interlaced = (flags & 0x40) != 0;

        dispose_previous();

        const int W = scan_.width, H = scan_.height;
        const int x0 = std::min(fx, W), y0 = std::min(fy, H);
        const int x1 = std::min(fx + fw, W), y1 = std::min(fy + fh, H);
        if (gce.disposal == 3) {
            saved_.resize(std::size_t(x1 - x0) * std::size_t(y1 - y0) * 4);
            const std::size_t row = std::size_t(x1 - x0) * 4;
            for (int y = y0; y < y1; ++y)
                std::memcpy(&saved_[std::size_t(y - y0) * row], &canvas_[(std::size_t(y) * W + x0) * 4], row);
        }

        indices_.resize(std::size_t(fw) * fh);
        std::size_t produced = 0;
        if (!lzw_decode(d_, n_, p_, indices_, produced)) return fail(error, "the GIF's image data is truncated");
        if (palette->empty() && produced > 0) return fail(error, "a GIF frame without a colour table");

        // Stream order to rows: interlaced frames come in four passes.
        const std::size_t colours = palette->size() / 3;
        int pass = 0, step = interlaced ? 8 : 1, row = 0;
        for (std::size_t i = 0; i < produced && fw > 0; ) {
            const int y = fy + row;
            if (y < y1) {
                uint8_t* dst = &canvas_[(std::size_t(y) * W) * 4];
                for (int x = 0; x < fw; ++x) {
                    const uint8_t idx = indices_[i + std::size_t(x)];
                    if (i + std::size_t(x) >= produced) break;
                    const int cx = fx + x;
                    if (cx >= x1 || int(idx) == gce.transparent || idx >= colours) continue;
                    uint8_t* px = dst + std::size_t(cx) * 4;
                    px[0] = (*palette)[std::size_t(idx) * 3];
                    px[1] = (*palette)[std::size_t(idx) * 3 + 1];
                    px[2] = (*palette)[std::size_t(idx) * 3 + 2];
                    px[3] = 255;
                }
            }
            i += std::size_t(fw);
            row += step;
            if (interlaced) {
                while (row >= fh && pass < 3) {
                    ++pass;
                    static const int kStart[4] = {0, 4, 2, 1};
                    static const int kStep[4] = {8, 8, 4, 2};
                    row = kStart[pass];
                    step = kStep[pass];
                }
                if (row >= fh) break;
            } else if (row >= fh) {
                break;
            }
        }

        have_prev_ = true;
        prev_disposal_ = gce.disposal;
        prev_x0_ = x0;
        prev_y0_ = y0;
        prev_x1_ = x1;
        prev_y1_ = y1;
        return true;
    }

    SharedBytes keep_;
    const uint8_t* d_;
    std::size_t n_;
    GifScan scan_;
    std::size_t first_block_;
    std::vector<uint8_t> gct_;
    std::vector<uint8_t> canvas_;
    std::vector<uint8_t> saved_;    // under a disposal-3 frame
    std::vector<uint8_t> indices_;  // the frame being drawn
    std::size_t p_ = 0;
    int index_ = 0;
    bool failed_ = false;
    bool have_prev_ = false;
    int prev_disposal_ = 0, prev_x0_ = 0, prev_y0_ = 0, prev_x1_ = 0, prev_y1_ = 0;
};

// Header + logical screen descriptor + global colour table: where the
// blocks start.
bool read_header(const uint8_t* d, std::size_t n, int& w, int& h, std::vector<uint8_t>* gct, std::size_t& p) {
    if (!is_gif(d, n) || n < 13) return false;
    w = le16(d + 6);
    h = le16(d + 8);
    const uint8_t flags = d[10];
    p = 13;
    if (flags & 0x80) {
        const std::size_t bytes = 3u * (1u << ((flags & 7) + 1));
        if (p + bytes > n) return false;
        if (gct) gct->assign(d + p, d + p + bytes);
        p += bytes;
    }
    return true;
}

} // namespace

bool is_gif(const uint8_t* d, std::size_t n) {
    return d && n >= 6 && std::memcmp(d, "GIF8", 4) == 0 && (d[4] == '7' || d[4] == '9') && d[5] == 'a';
}

bool gif_scan(const uint8_t* d, std::size_t n, GifScan& out) {
    out = GifScan{};
    std::size_t p = 0;
    int w = 0, h = 0;
    if (!read_header(d, n, w, h, nullptr, p)) return false;
    int extent_w = 0, extent_h = 0;
    Gce gce;
    while (p < n) {
        const uint8_t block = d[p++];
        if (block == 0x3B) break;
        if (block == 0x21) {
            if (p >= n) break;
            const uint8_t label = d[p++];
            if (label == 0xF9) gce = read_gce(d, n, p);
            if (label == 0xFF && p + 12 <= n && d[p] == 11 &&
                (std::memcmp(d + p + 1, "NETSCAPE2.0", 11) == 0 || std::memcmp(d + p + 1, "ANIMEXTS1.0", 11) == 0)) {
                const std::size_t q = p + 12;
                if (q + 4 <= n && d[q] >= 3 && d[q + 1] == 1) {
                    const int stored = le16(d + q + 2);
                    out.loop_count = stored == 0 ? 0 : stored + 1;
                }
            }
            if (!skip_sub_blocks(d, n, p)) break;
        } else if (block == 0x2C) {
            if (p + 9 > n) break;
            extent_w = std::max(extent_w, le16(d + p) + le16(d + p + 4));
            extent_h = std::max(extent_h, le16(d + p + 2) + le16(d + p + 6));
            const uint8_t flags = d[p + 8];
            p += 9;
            if (flags & 0x80) p += 3u * (1u << ((flags & 7) + 1));
            if (p >= n) break;
            ++p;  // LZW minimum code size
            if (!skip_sub_blocks(d, n, p)) break;
            out.delays_ms.push_back(gce.delay_ms);
            ++out.frames;
            gce = Gce{};
        } else {
            break;
        }
    }
    out.width = w > 0 ? w : extent_w;
    out.height = h > 0 ? h : extent_h;
    return true;
}

std::unique_ptr<FrameDecoder> gif_open(const uint8_t* d, std::size_t n, SharedBytes keep, std::string* why) {
    GifScan scan;
    std::size_t p = 0;
    int w = 0, h = 0;
    std::vector<uint8_t> gct;
    if (!read_header(d, n, w, h, &gct, p) || !gif_scan(d, n, scan)) {
        if (why) *why = "not a GIF";
        return nullptr;
    }
    if (scan.frames <= 0) {
        if (why) *why = "a GIF without frames";
        return nullptr;
    }
    if (scan.width <= 0 || scan.height <= 0 || uint64_t(scan.width) * uint64_t(scan.height) > kMaxCanvasPixels) {
        if (why) *why = "the GIF's canvas size is unusable";
        return nullptr;
    }
    return std::make_unique<GifDecoder>(d, n, std::move(keep), std::move(scan), p, std::move(gct));
}

} // namespace broimage::detail
