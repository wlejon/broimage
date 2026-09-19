// broimage::api::setPathResolver: every filename `bro.image` hands to an
// encoder or a decoder goes through the host's resolver first, so a relative
// path means what it means to the app rather than to the process CWD.
//
// Mirrors brotensor's resolver: one process-global slot, unset by default.

#include "../src/api/api.h"
#include "embed/embed.h"
#include "eval/eval.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace ev = bronze::embed;

std::filesystem::path g_scratch;
std::vector<std::string> g_asked;

void fail(const std::string& what) {
    std::cerr << "  path-resolver check failed: " << what << std::endl;
    std::exit(1);
}

std::string runScript(const char* label, const std::string& script) {
    ev::CallResult res = bronze::eval::evalScript(script);
    if (res.thrown) fail(std::string(label) + " threw: " + ev::toUtf8(res.value));
    return ev::toUtf8(res.value);
}

// "bvfs" and "bvfs/<file>" land in the scratch dir; anything else is left
// alone, the way a host resolver leaves an absolute path alone.
std::string resolve(const std::string& p) {
    g_asked.push_back(p);
    if (p == "bvfs") return g_scratch.string();
    if (p.rfind("bvfs/", 0) == 0) return (g_scratch / p.substr(5)).string();
    return p;
}

} // namespace

void broimageTestPathResolver() {
    std::cout << "Checking broimage::api::setPathResolver..." << std::endl;

    g_scratch = std::filesystem::temp_directory_path() / "broimage_api_paths";
    std::filesystem::remove_all(g_scratch);
    std::filesystem::create_directories(g_scratch);

    // 1. Unresolved: "bvfs/..." has no such directory under the CWD, so the
    //    encoder cannot write and reports false. This is the behaviour a host
    //    without a resolver keeps.
    if (runScript("unresolved encode", R"JS(
        (function() {
            const px = new Uint8Array(4 * 4 * 4).fill(200);
            return String(bro.image.encodePngFile("bvfs/before.png", px, 4, 4, 4));
        })()
    )JS") != "false") {
        fail("encodePngFile wrote through an unresolved relative path");
    }
    if (!g_asked.empty()) fail("the resolver was consulted before it was installed");

    // 2. Install the resolver. Every path the binding takes now goes through
    //    it — process-wide, not per realm.
    broimage::api::setPathResolver(&resolve);

    if (runScript("resolved encode", R"JS(
        (function() {
            const px = new Uint8Array(4 * 4 * 4).fill(200);
            return String(bro.image.encodePngFile("bvfs/out.png", px, 4, 4, 4));
        })()
    )JS") != "true") {
        fail("encodePngFile failed through the resolver");
    }
    if (g_asked.empty() || g_asked.back() != "bvfs/out.png") {
        fail("encodePngFile did not consult the resolver");
    }
    if (!std::filesystem::exists(g_scratch / "out.png")) {
        fail("encodePngFile did not write to the resolved location");
    }

    if (runScript("resolved jpeg encode", R"JS(
        (function() {
            const px = new Uint8Array(4 * 4 * 3).fill(120);
            return String(bro.image.encodeJpegFile("bvfs/out.jpg", px, 4, 4, 3, 85));
        })()
    )JS") != "true") {
        fail("encodeJpegFile failed through the resolver");
    }
    if (!std::filesystem::exists(g_scratch / "out.jpg")) {
        fail("encodeJpegFile did not write to the resolved location");
    }

    // 3. The decoders read the same way round: the file only exists at the
    //    resolved location, so a decode of the relative name proves it.
    if (runScript("resolved decode", R"JS(
        (function() {
            const d = bro.image.decodeOriented("bvfs/out.png");
            if (!d) return "decodeOriented returned null";
            if (d.width !== 4 || d.height !== 4) return "size " + d.width + "x" + d.height;
            if (bro.image.readExifOrientation("bvfs/out.png") !== 1) return "exif orientation";
            return "SUCCESS";
        })()
    )JS") != "SUCCESS") {
        fail("decodeOriented/readExifOrientation did not resolve the path");
    }

    const size_t askedWithResolver = g_asked.size();
    if (askedWithResolver < 4) fail("not every file entry point consulted the resolver");

    // 4. Clearing the resolver restores the take-it-as-given behaviour, so a
    //    host that never sets one is unaffected.
    broimage::api::setPathResolver(nullptr);
    if (runScript("cleared resolver", R"JS(
        (function() {
            const px = new Uint8Array(4 * 4 * 4).fill(200);
            return String(bro.image.encodePngFile("bvfs/after.png", px, 4, 4, 4));
        })()
    )JS") != "false") {
        fail("encodePngFile still resolved after the resolver was cleared");
    }
    if (g_asked.size() != askedWithResolver) {
        fail("the resolver was still consulted after being cleared");
    }
    if (std::filesystem::exists(g_scratch / "after.png")) {
        fail("a cleared resolver still redirected the write");
    }

    std::filesystem::remove_all(g_scratch);
    std::cout << "  broimage path resolver OK (" << askedWithResolver
              << " resolved paths)." << std::endl;
}
