#pragma once

#include "embed/embed.h"
#include "object_builder.h"
#include "arg_reader.h"
#include "host_class.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace broimage::api {

namespace ev = bronze::embed;
using Value = bronze::Value;
using ElementKind = bronze::ElementKind;

struct TypedArrayView {
    uint8_t* data = nullptr;
    size_t byteLength = 0;
    size_t bytesPerElement = 0;
    ElementKind kind{};
};

inline bool unpackTypedArray(Value val, const char* name, TypedArrayView* out) {
    auto info = ev::typedArrayInfo(val);
    if (!info.data) {
        ev::throwTypeError(std::string(name) + " must be a TypedArray");
        return false;
    }
    out->data = info.data;
    out->byteLength = info.byteLength;
    out->bytesPerElement = info.bytesPerElement;
    out->kind = info.elementKind;
    return true;
}

inline Value typedArrayFrom(ElementKind kind, const void* data, size_t byteLength, uint32_t elementCount) {
    Value arr = ev::createTypedArray(kind, elementCount);
    ev::fillTypedArray(arr, std::span<const uint8_t>(static_cast<const uint8_t*>(data), byteLength));
    return arr;
}

inline bool getPropF64(Value obj, const char* key, double* out, double defVal) {
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) {
        *out = defVal;
        return true;
    }
    if (ev::isNumber(v)) {
        *out = ev::toDouble(v);
        return true;
    }
    return false;
}

inline bool getPropI32(Value obj, const char* key, int32_t* out, int32_t defVal) {
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) {
        *out = defVal;
        return true;
    }
    if (ev::isNumber(v)) {
        *out = static_cast<int32_t>(ev::toDouble(v));
        return true;
    }
    return false;
}

inline bool getPropStr(Value obj, const char* key, std::string* out, const std::string& defVal = "") {
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) {
        *out = defVal;
        return true;
    }
    if (ev::isString(v)) {
        *out = ev::toUtf8(v);
        return true;
    }
    return false;
}

// Read `n` numbers from `obj[key]` (a JS array or a typed array). A missing
// or null property falls back to `defVal` (pass nullptr to make it required).
// Returns false when the property is present but not `n` numbers long.
inline bool getPropFloats(Value obj, const char* key, float* out, int n,
                          const float* defVal) {
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) {
        if (!defVal) return false;
        for (int i = 0; i < n; ++i) out[i] = defVal[i];
        return true;
    }
    if (!ev::isObject(v)) return false;
    auto info = ev::typedArrayInfo(v);
    if (info.data && info.elementKind == ev::elements::Float32) {
        if (info.byteLength < static_cast<size_t>(n) * sizeof(float)) return false;
        const float* src = reinterpret_cast<const float*>(info.data);
        for (int i = 0; i < n; ++i) out[i] = src[i];
        return true;
    }
    Value lenV = ev::getProperty(v, "length");
    if (!ev::isNumber(lenV) || static_cast<int>(ev::toDouble(lenV)) < n) return false;
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<float>(ev::toDouble(ev::getElement(v, static_cast<uint32_t>(i))));
    }
    return true;
}

inline float readScalar(const uint8_t* p, size_t bpe, bool isFloat, bool isSigned) {
    if (isFloat) {
        if (bpe == 4) return *reinterpret_cast<const float*>(p);
        return static_cast<float>(*reinterpret_cast<const double*>(p));
    }
    if (isSigned) {
        if (bpe == 1) return static_cast<float>(*reinterpret_cast<const int8_t*>(p));
        if (bpe == 2) return static_cast<float>(*reinterpret_cast<const int16_t*>(p));
        return static_cast<float>(*reinterpret_cast<const int32_t*>(p));
    }
    if (bpe == 1) return static_cast<float>(*p);
    if (bpe == 2) return static_cast<float>(*reinterpret_cast<const uint16_t*>(p));
    return static_cast<float>(*reinterpret_cast<const uint32_t*>(p));
}

struct ScalarKind { bool isFloat; bool isSigned; };

inline bool probeScalarKind(Value val, ScalarKind* out) {
    auto info = ev::typedArrayInfo(val);
    if (!info.data) return false;
    out->isFloat = (info.elementKind == ev::elements::Float32 || info.elementKind == ev::elements::Float64);
    out->isSigned = (info.elementKind == ev::elements::Int8 || info.elementKind == ev::elements::Int16 || info.elementKind == ev::elements::Int32);
    return true;
}

inline Value hostArrayOf(size_t count, const std::function<Value(size_t)>& make) {
    ev::CallResult parsed = ev::parseJson("[]");
    if (parsed.thrown || !ev::isObject(parsed.value)) {
        return ev::undefined();
    }
    ev::Persistent arr(parsed.value);
    if (count == 0) return arr.get();

    ev::Persistent push(ev::getProperty(arr.get(), "push"));
    if (ev::isFunction(push.get())) {
        for (size_t i = 0; i < count; ++i) {
            Value v = make(i);
            ev::call(push.get(), arr.get(), std::span<const Value>(&v, 1));
        }
    } else {
        for (size_t i = 0; i < count; ++i) {
            Value v = make(i);
            arr.set(ev::setElement(arr.get(), static_cast<uint32_t>(i), v));
        }
    }
    return arr.get();
}

// Path resolution for the file entry points (api.h setPathResolver). Every
// string path a binding hands to a decoder or an encoder goes through this
// first, so a relative path means what it means to the host's app.
std::string resolvePath(const std::string& path);

Value ensureBroImage();
void installCodecsOnto(Value imageObj);
void installOpsOnto(Value imageObj);

// The pre-transition bro.image members the bronze port dropped, split by
// subject so each translation unit stays small: decode / probe / EXIF,
// geometric + alpha, and layout / color-matrix / normalize / tiling.
void installDecodeOnto(Value imageObj);
void installGeometryOnto(Value imageObj);
void installPreprocOnto(Value imageObj);

} // namespace broimage::api
