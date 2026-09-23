#include "api.h"
#include "host_image_internal.h"

namespace broimage::api {

namespace {

std::function<std::string(const std::string&)>& pathResolverSlot() {
    static std::function<std::string(const std::string&)> resolver;
    return resolver;
}

} // namespace

std::string resolvePath(const std::string& path) {
    auto& r = pathResolverSlot();
    return r ? r(path) : path;
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
}

void installImage() {
    ev::Persistent img(ensureBroImage());
    installOpsOnto(img.get());
    installCodecsOnto(img.get());
    installDecodeOnto(img.get());
    installGeometryOnto(img.get());
    installPreprocOnto(img.get());
}

} // namespace broimage::api
