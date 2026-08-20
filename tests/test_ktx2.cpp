#include "broimage/ktx2.h"

#include "ktx2_fixtures.h"

#include <cstdio>
#include <cstring>

static int failures = 0;

#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            std::fprintf(stderr, "FAIL: %s\n", msg);            \
            ++failures;                                         \
        }                                                       \
    } while (0)

static void checkRgba8(const unsigned char* bytes, size_t size, const char* label) {
    broimage::Ktx2Image img = broimage::transcode_ktx2(bytes, size);
    CHECK(img.ok(), img.error.c_str());
    if (!img.ok()) return;
    CHECK(img.width == 40 && img.height == 40, label);
    CHECK(!img.mips.empty(), "has at least one mip");
    CHECK(img.format == broimage::Ktx2Format::RGBA8, "format is RGBA8");
    const broimage::Ktx2Level& top = img.mips[0];
    CHECK(top.width == img.width && top.height == img.height, "level 0 is full size");
    CHECK(top.data.size() == static_cast<size_t>(top.width) * top.height * 4,
          "RGBA8 payload is w*h*4");
    // The sample textures are not blank; a transcode that produced all-zero
    // pixels decoded nothing.
    bool nonZero = false;
    for (unsigned char b : top.data) {
        if (b != 0) {
            nonZero = true;
            break;
        }
    }
    CHECK(nonZero, "pixels carry data");
}

int main() {
    checkRgba8(kKtx2Etc1s, sizeof kKtx2Etc1s, "ETC1S 256x256");
    checkRgba8(kKtx2Uastc, sizeof kKtx2Uastc, "UASTC 256x256");

    // BC7 keeps the block layout: ceil(w/4)*ceil(h/4) 16-byte blocks.
    {
        broimage::Ktx2Image img =
            broimage::transcode_ktx2(kKtx2Uastc, sizeof kKtx2Uastc, broimage::Ktx2Format::BC7);
        CHECK(img.ok(), img.error.c_str());
        if (img.ok()) {
            CHECK(img.format == broimage::Ktx2Format::BC7, "BC7 target honored");
            const broimage::Ktx2Level& top = img.mips[0];
            const size_t blocks = ((top.width + 3) / 4) * static_cast<size_t>((top.height + 3) / 4);
            CHECK(top.data.size() == blocks * 16, "BC7 payload is 16 bytes per block");
        }
    }

    // Failure paths answer with an error, not a crash.
    {
        const unsigned char junk[32] = {9, 9, 9, 9};
        CHECK(!broimage::transcode_ktx2(junk, sizeof junk).ok(), "junk refuses");
        CHECK(!broimage::transcode_ktx2(nullptr, 0).ok(), "empty refuses");
    }

    if (failures == 0) std::printf("broimage ktx2: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
