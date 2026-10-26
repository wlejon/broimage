// bro.image under the host's access gate (embed/embed_access.h; src/api/api.h,
// "Access"): every entry point handed a path asks Read (decoders, probes, EXIF,
// frames) or Write (encodePngFile / encodeJpegFile) about it, a refusal is an
// Error carrying the gate's sentence, code EACCES and the absolute path,
// nothing is read or written, and with the gate removed everything behaves as
// before. Run from test_image_api.cpp's main.

#include "embed/embed.h"
#include "embed/embed_access.h"
#include "eval/eval.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
namespace ev = bronze::embed;

#define GCHECK(cond, msg)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::cerr << "CHECK failed: " #cond " (" << __FILE__ << ":"        \
                      << __LINE__ << "): " << (msg) << std::endl;              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

std::string utf8(const fs::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

// The path as the gate is handed it.
std::string gatePath(const fs::path& p) {
    return utf8(fs::absolute(p).lexically_normal());
}

std::string jsStr(const fs::path& p) {
    std::string out = "'";
    for (char c : utf8(p)) {
        if (c == '\\' || c == '\'') out += '\\';
        out += c;
    }
    return out + "'";
}

std::string evalString(const std::string& src) {
    auto r = bronze::eval::evalScript(src);
    if (r.thrown) return "<thrown>";
    return ev::isString(r.value) ? ev::toUtf8(r.value) : "<not a string>";
}

// Runs `stmt`; "ok" or the refusal as one comparable string.
std::string syncJs(const std::string& stmt) {
    return evalString("(function() { try { " + stmt + "; return 'ok'; } catch (e) {\n"
                      "  return 'denied|' + e.message + '|' + e.code + '|' + e.path; } })()");
}

std::string refusal(const fs::path& p) {
    const std::string path = gatePath(p);
    return "denied|no files access: " + path + "|EACCES|" + path;
}

const char* kPixels = "new Uint8Array(4 * 4 * 4).fill(180)";

}  // namespace

void broimageTestAccessGate() {
    std::cout << "Checking bro.image under the access gate..." << std::endl;

    const fs::path root = fs::temp_directory_path() / "broimage_gate";
    const fs::path ok = root / "ok";
    const fs::path hidden = root / "hidden";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(ok);
    fs::create_directories(hidden);

    // Fixtures, written before any gate is installed.
    std::string r = syncJs("bro.image.encodePngFile(" + jsStr(ok / "a.png") + ", " + kPixels + ", 4, 4, 4);"
                           "bro.image.encodePngFile(" + jsStr(hidden / "a.png") + ", " + kPixels + ", 4, 4, 4)");
    GCHECK(r == "ok", r);

    const std::string hiddenPrefix = gatePath(hidden);
    std::atomic<int> calls{0};
    ev::setAccessGate([&](const ev::AccessRequest& q) -> std::string {
        if (q.kind != ev::AccessKind::Read && q.kind != ev::AccessKind::Write) return "";
        ++calls;
        const std::string path(q.subject);
        return path.compare(0, hiddenPrefix.size(), hiddenPrefix) == 0 ? "no files access: " + path : "";
    });

    // Readers: every path entry point asks Read.
    const fs::path secret = hidden / "a.png";
    for (const char* call : {"decodeU16", "decodeF32", "decodeOriented", "decodeExifThumbnail",
                             "readExifOrientation", "readExif", "readHeifInfo", "decodeFrames",
                             "openFrames"}) {
        r = syncJs(std::string("bro.image.") + call + "(" + jsStr(secret) + ")");
        GCHECK(r == refusal(secret), std::string(call) + ": " + r);
    }
    r = syncJs("bro.image.decodeScaled(" + jsStr(secret) + ", {width: 2, height: 2})");
    GCHECK(r == refusal(secret), r);
    r = syncJs("bro.image.decodeThumbnail(" + jsStr(secret) + ", 2)");
    GCHECK(r == refusal(secret), r);
    r = syncJs("if (bro.image.decodeOriented(" + jsStr(ok / "a.png") + ").width !== 4) throw new Error('decode')");
    GCHECK(r == "ok", r);
    r = syncJs("bro.image.openFrames(" + jsStr(ok / "a.png") + ")");
    GCHECK(r == "ok", r);

    // Writers ask Write; refused, nothing is written.
    r = syncJs("bro.image.encodePngFile(" + jsStr(hidden / "out.png") + ", " + kPixels + ", 4, 4, 4)");
    GCHECK(r == refusal(hidden / "out.png"), r);
    GCHECK(!fs::exists(hidden / "out.png"), "a refused encodePngFile wrote");
    r = syncJs("bro.image.encodeJpegFile(" + jsStr(hidden / "out.jpg") + ", " + kPixels + ", 4, 4, 4, 90)");
    GCHECK(r == refusal(hidden / "out.jpg"), r);
    GCHECK(!fs::exists(hidden / "out.jpg"), "a refused encodeJpegFile wrote");
    r = syncJs("bro.image.encodeJpegFile(" + jsStr(ok / "out.jpg") + ", " + kPixels + ", 4, 4, 4, 90)");
    GCHECK(r == "ok", r);
    GCHECK(fs::exists(ok / "out.jpg"), "an allowed encodeJpegFile did not write");

    // With the gate removed, nothing asks and everything works as before.
    ev::setAccessGate(nullptr);
    const int before = calls.load();
    r = syncJs("if (bro.image.decodeOriented(" + jsStr(secret) + ").width !== 4) throw new Error('decode')");
    GCHECK(r == "ok", r);
    r = syncJs("bro.image.encodePngFile(" + jsStr(hidden / "out.png") + ", " + kPixels + ", 4, 4, 4)");
    GCHECK(r == "ok", r);
    GCHECK(fs::exists(hidden / "out.png"), "an ungated encodePngFile did not write");
    GCHECK(calls.load() == before, "a removed gate was still asked");

    fs::remove_all(root, ec);
    std::cout << "  access gate [PASS]" << std::endl;
}
