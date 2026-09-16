#include "api.h"
#include "host_image_internal.h"

namespace broimage::api {

Value ensureBroImage() {
    Value globalThisVal = ev::undefined();
    auto gt = ev::globalValue("globalThis");
    if (gt.found && ev::isObject(gt.value)) {
        globalThisVal = gt.value;
    }

    Value broVal = ev::globalValue("bro").found ? ev::globalValue("bro").value : ev::undefined();
    if (!ev::isObject(broVal)) {
        if (!ev::isUndefined(globalThisVal)) {
            Value candidate = ev::getProperty(globalThisVal, "bro");
            if (ev::isObject(candidate)) {
                broVal = candidate;
            }
        }
    }
    if (!ev::isObject(broVal)) {
        broVal = ev::createObject();
        ev::registerGlobal("bro", broVal);
        if (!ev::isUndefined(globalThisVal)) {
            ev::setProperty(globalThisVal, "bro", broVal);
        }
    }

    ev::Persistent broP(broVal);
    Value imgVal = ev::getProperty(broP.get(), "image");
    if (!ev::isObject(imgVal)) {
        imgVal = ev::createObject();
        broP.set(ev::setProperty(broP.get(), "image", imgVal));
    }
    return imgVal;
}

void installCodecs() {
    Value img = ensureBroImage();
    installCodecsOnto(img);
}

void installImage() {
    Value img = ensureBroImage();
    installOpsOnto(img);
    installCodecsOnto(img);
}

} // namespace broimage::api
