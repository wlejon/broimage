#pragma once

#include "embed/embed.h"
#include "object_builder.h"
#include "arg_reader.h"
#include "host_class.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <span>
#include <string>
#include <vector>

namespace broimage::api {

namespace ev = bronze::embed;
using Value = bronze::Value;
using ElementKind = bronze::ElementKind;

// A typed-array argument, split into what survives a collection and what does
// not (embed.h, THE POINTER CONTRACT). The shape — byte length, element size,
// kind — is validated once, by unpackTypedArray, and stays true. The bytes
// live in the moving heap: any allocating embed call (a getProperty on an
// options object, a throw, a fromUtf8, a createTypedArray) may relocate them.
// So unpackTypedArray roots the value and leaves `data` null, and a native
// fills `data` with resolveViews AFTER its last allocating call and right
// before the kernel runs. A native that forgets reads a null pointer and
// faults on the first run, rather than reading a stale one only under GC
// pressure.
struct TypedArrayView {
    uint8_t* data = nullptr;
    size_t byteLength = 0;
    size_t bytesPerElement = 0;
    ElementKind kind{};
    const char* name = "";
    ev::Persistent root;
};

inline bool unpackTypedArray(Value val, const char* name, TypedArrayView* out) {
    auto info = ev::typedArrayInfo(val);
    if (!info.data) {
        ev::throwTypeError(std::string(name) + " must be a TypedArray");
        return false;
    }
    out->data = nullptr;
    out->byteLength = info.byteLength;
    out->bytesPerElement = info.bytesPerElement;
    out->kind = info.elementKind;
    out->name = name;
    out->root.set(val);
    return true;
}

// Re-read each view's `data` from its rooted value. Call it once the native
// has made its last allocating embed call, immediately before the kernel, and
// make no allocating call between this and the last use of `data`. Answers
// false (with a TypeError pending) when a view no longer spans the bytes it
// was validated with — an options getter can detach or shrink a buffer.
inline bool resolveViews(std::initializer_list<TypedArrayView*> views) {
    for (TypedArrayView* v : views) {
        auto info = ev::typedArrayInfo(v->root.get());
        if (!info.data || info.byteLength < v->byteLength) {
            for (TypedArrayView* w : views) w->data = nullptr;
            ev::throwTypeError(std::string(v->name) + " was detached or shrunk during the call");
            return false;
        }
        v->data = info.data;
    }
    return true;
}

inline Value typedArrayFrom(ElementKind kind, const void* data, size_t byteLength, uint32_t elementCount) {
    Value arr = ev::createTypedArray(kind, elementCount);
    ev::fillTypedArray(arr, std::span<const uint8_t>(static_cast<const uint8_t*>(data), byteLength));
    return arr;
}

// ---- sizing ------------------------------------------------------------------
//
// Every kernel below trusts its dimensions: it writes w*h*channels elements
// whatever the buffer holds. So a binding checks each view against what the
// kernel will touch BEFORE resolving it — a view that is too small is a
// RangeError, never an out-of-bounds read or write. The kernels index with
// int, so a span past INT32_MAX is refused too.

inline bool requireBytes(const TypedArrayView& v, uint64_t need, const char* who) {
    if (need > static_cast<uint64_t>(INT32_MAX)) {
        ev::throwRangeError(std::string(who) + ": " + v.name + " dimensions are too large");
        return false;
    }
    if (v.byteLength < need) {
        ev::throwRangeError(std::string(who) + ": " + v.name + " is too small (needs " +
                            std::to_string(need) + " bytes, has " +
                            std::to_string(v.byteLength) + ")");
        return false;
    }
    return true;
}

// The bytes an HWC image of `bpe`-byte elements spans with a row pitch of
// `strideBytes` (0 = tightly packed): every row but the last is a full
// stride, the last only its w*channels payload. A negative stride, or one
// shorter than a row, is refused.
inline bool requireImage(const TypedArrayView& v, int w, int h, int channels, int strideBytes,
                         size_t bpe, const char* who) {
    const uint64_t row = static_cast<uint64_t>(w) * static_cast<uint64_t>(channels) * bpe;
    if (strideBytes < 0 || (strideBytes > 0 && static_cast<uint64_t>(strideBytes) < row)) {
        ev::throwRangeError(std::string(who) + ": " + v.name +
                            " stride must be 0 or at least width*channels bytes");
        return false;
    }
    const uint64_t pitch = strideBytes > 0 ? static_cast<uint64_t>(strideBytes) : row;
    return requireBytes(v, pitch * static_cast<uint64_t>(h - 1) + row, who);
}

inline bool requireFloat32(const TypedArrayView& v, const char* who) {
    if (v.kind != ev::elements::Float32) {
        ev::throwTypeError(std::string(who) + ": " + v.name + " must be a Float32Array");
        return false;
    }
    return true;
}

// Read `n` numbers from an array-like argument (a JS array or any typed
// array). False, with a TypeError pending, when it is not one or is shorter
// than `n`. May allocate, so the caller resolves its views afterwards.
inline bool readFloatArray(Value arr, float* out, int n, const std::string& what) {
    ev::Persistent v(arr);
    if (!ev::isObject(v.get())) {
        ev::throwTypeError(what + " must be an array or typed array");
        return false;
    }
    auto info = ev::typedArrayInfo(v.get());
    if (info.data) {
        if (info.elementCount < static_cast<uint32_t>(n)) {
            ev::throwTypeError(what + " must have " + std::to_string(n) + " entries");
            return false;
        }
        if (info.elementKind == ev::elements::Float32) {
            const float* src = reinterpret_cast<const float*>(info.data);
            for (int i = 0; i < n; ++i) out[i] = src[i];
            return true;
        }
    } else {
        Value lenV = ev::getProperty(v.get(), "length");
        if (!ev::isNumber(lenV) || ev::toDouble(lenV) < n) {
            ev::throwTypeError(what + " must have " + std::to_string(n) + " entries");
            return false;
        }
    }
    for (int i = 0; i < n; ++i) {
        Value e = ev::getElement(v.get(), static_cast<uint32_t>(i));
        if (!ev::isNumber(e)) {
            ev::throwTypeError(what + " entries must be numbers");
            return false;
        }
        out[i] = static_cast<float>(ev::toDouble(e));
    }
    return true;
}

// ---- option reads ----------------------------------------------------------
//
// A missing or null key takes the default. A present key of the wrong type
// answers false WITH a TypeError pending, so a caller's bare
// `return ev::undefined()` still throws rather than silently doing nothing.

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
    ev::throwTypeError(std::string(key) + " must be a number");
    return false;
}

// Integers are truncated toward zero; NaN and values outside int32 are a
// RangeError rather than the undefined behaviour of a raw cast.
inline bool getPropI32(Value obj, const char* key, int32_t* out, int32_t defVal) {
    Value v = ev::getProperty(obj, key);
    if (ev::isUndefined(v) || ev::isNull(v)) {
        *out = defVal;
        return true;
    }
    if (!ev::isNumber(v)) {
        ev::throwTypeError(std::string(key) + " must be a number");
        return false;
    }
    const double d = std::trunc(ev::toDouble(v));
    if (!(d >= static_cast<double>(INT32_MIN) && d <= static_cast<double>(INT32_MAX))) {
        ev::throwRangeError(std::string(key) + " is out of range");
        return false;
    }
    *out = static_cast<int32_t>(d);
    return true;
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
    ev::throwTypeError(std::string(key) + " must be a string");
    return false;
}

// A positional count or dimension argument: a finite number in
// (0, INT32_MAX]. False with a RangeError pending otherwise.
inline bool countArg(Value v, const char* who, const char* what, int32_t* out) {
    const double d = ev::isNumber(v) ? std::trunc(ev::toDouble(v)) : 0.0;
    if (!(d > 0 && d <= static_cast<double>(INT32_MAX))) {
        ev::throwRangeError(std::string(who) + ": " + what + " must be a positive integer");
        return false;
    }
    *out = static_cast<int32_t>(d);
    return true;
}

// Read `n` numbers from `obj[key]` (a JS array or a typed array). A missing
// or null property falls back to `defVal` (pass nullptr to make it required).
// Returns false when the property is present but not `n` numbers long.
inline bool getPropFloats(Value obj, const char* key, float* out, int n,
                          const float* defVal) {
    // Rooted: the "length" read and each getElement may allocate and move it.
    ev::Persistent v(ev::getProperty(obj, key));
    if (ev::isUndefined(v.get()) || ev::isNull(v.get())) {
        if (!defVal) return false;
        for (int i = 0; i < n; ++i) out[i] = defVal[i];
        return true;
    }
    if (!ev::isObject(v.get())) return false;
    auto info = ev::typedArrayInfo(v.get());
    if (info.data && info.elementKind == ev::elements::Float32) {
        if (info.byteLength < static_cast<size_t>(n) * sizeof(float)) return false;
        const float* src = reinterpret_cast<const float*>(info.data);
        for (int i = 0; i < n; ++i) out[i] = src[i];
        return true;
    }
    Value lenV = ev::getProperty(v.get(), "length");
    if (!ev::isNumber(lenV) || static_cast<int>(ev::toDouble(lenV)) < n) return false;
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<float>(ev::toDouble(ev::getElement(v.get(), static_cast<uint32_t>(i))));
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
    Value created = ev::makeArray(0);
    if (!ev::isObject(created)) {
        return ev::undefined();
    }
    ev::Persistent arr(created);
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
