#include "api.h"
#include "host_image_internal.h"
#include "embed/embed_access.h"

#include <filesystem>

namespace broimage::api {

namespace {

std::function<std::string(const std::string&)>& pathResolverSlot() {
    static std::function<std::string(const std::string&)> resolver;
    return resolver;
}

std::string resolvePath(const std::string& path) {
    auto& r = pathResolverSlot();
    return r ? r(path) : path;
}

// `resolved` once the host's gate allows `kind` on it; throws the refusal.
std::string gated(std::string resolved, ev::AccessKind kind) {
    if (!ev::accessGateActive()) return resolved;
    namespace fs = std::filesystem;
    fs::path p(std::u8string(resolved.begin(), resolved.end()));
    std::error_code ec;
    fs::path abs = fs::absolute(p, ec);
    if (ec) abs = p;
    const std::u8string u8 = abs.lexically_normal().u8string();
    const std::string subject(u8.begin(), u8.end());
    if (std::string refused = ev::checkAccess(kind, subject); !refused.empty()) {
        ev::throwAccessError(refused, "EACCES", subject);
    }
    return resolved;
}

} // namespace

std::string readPath(const std::string& path) {
    return gated(resolvePath(path), ev::AccessKind::Read);
}

std::string writePath(const std::string& path) {
    return gated(resolvePath(path), ev::AccessKind::Write);
}

void setPathResolver(std::function<std::string(const std::string&)> resolver) {
    pathResolverSlot() = std::move(resolver);
}

// Every Value held across an allocating call below lives in a Persistent
// (embed.h's GC contract): each installer allocates, so a raw `img` handed to
// the second one would name the object's pre-collection address.
Value ensureBroImage() {
    ev::Persistent globalThisVal;
    auto gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        globalThisVal.set(gt.value);
    }

    ev::Persistent broP;
    auto bro = ev::globalValue("bro");
    if (bro.found && ev::isObject(bro.value)) broP.set(bro.value);
    if (!ev::isObject(broP.get()) && ev::isObject(globalThisVal.get())) {
        Value candidate = ev::getProperty(globalThisVal.get(), "bro");
        if (ev::isObject(candidate)) broP.set(candidate);
    }
    if (!ev::isObject(broP.get())) {
        broP.set(ev::createObject());
        ev::registerGlobal("bro", broP.get());
        if (ev::isObject(globalThisVal.get())) {
            globalThisVal.set(ev::setProperty(globalThisVal.get(), "bro", broP.get()));
        }
    }

    ev::Persistent imgP(ev::getProperty(broP.get(), "image"));
    if (!ev::isObject(imgP.get())) {
        imgP.set(ev::createObject());
        broP.set(ev::setProperty(broP.get(), "image", imgP.get()));
    }
    return imgP.get();
}

void installCodecs() {
    ev::Persistent img(ensureBroImage());
    installCodecsOnto(img.get());
    // decode / probe / EXIF belong with the encoders: a host that mounts only
    // the codec half still needs decodeF32 and friends.
    installDecodeOnto(img.get());
    installFramesOnto(img.get());
}

void installImage() {
    ev::Persistent img(ensureBroImage());
    installOpsOnto(img.get());
    installCodecsOnto(img.get());
    installDecodeOnto(img.get());
    installFramesOnto(img.get());
    installGeometryOnto(img.get());
    installAlignOnto(img.get());
    installSharpnessOnto(img.get());
    installCompareOnto(img.get());
    installMotionOnto(img.get());
    installPreprocOnto(img.get());
}

} // namespace broimage::api
