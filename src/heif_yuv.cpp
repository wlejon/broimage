// YUV to RGBA8 (heif_yuv.h).
//
// Y'CbCr to R'G'B' follows ITU-T H.273 / ITU-R BT.601, BT.709 and BT.2020:
//   R' = Y' + 2(1 - Kr) Cr,  B' = Y' + 2(1 - Kb) Cb,  G' = (Y' - Kr R' - Kb B') / Kg
// with Y' and C normalised per the range: full uses the whole code range,
// limited puts black at 16 and white at 235 (chroma 16..240), both scaled
// by 2^(bitDepth - 8). Chroma is upsampled bilinearly, with the 4:2:0
// sample position the frame states (centre when unknown).

#include "heif_yuv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <vector>

namespace broimage::detail {

namespace {

struct XY {
    double x, y;
};

struct Primaries {
    XY r, g, b, w;
};

constexpr XY kD65{0.3127, 0.3290};

// H.273 ColourPrimaries; unknown code points read as BT.709.
Primaries primaries_of(int cp) {
    switch (cp) {
        case 4: return {{0.67, 0.33}, {0.21, 0.71}, {0.14, 0.08}, {0.310, 0.316}};
        case 5: return {{0.64, 0.33}, {0.29, 0.60}, {0.15, 0.06}, kD65};
        case 6:
        case 7: return {{0.630, 0.340}, {0.310, 0.595}, {0.155, 0.070}, kD65};
        case 8: return {{0.681, 0.319}, {0.243, 0.692}, {0.145, 0.049}, {0.310, 0.316}};
        case 9: return {{0.708, 0.292}, {0.170, 0.797}, {0.131, 0.046}, kD65};
        case 11: return {{0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}, {0.314, 0.351}};
        case 12: return {{0.680, 0.320}, {0.265, 0.690}, {0.150, 0.060}, kD65};
        case 22: return {{0.630, 0.340}, {0.295, 0.605}, {0.155, 0.077}, kD65};
        default: return {{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}, kD65};
    }
}

using Mat3 = std::array<std::array<double, 3>, 3>;

Mat3 inverse(const Mat3& m) {
    const double a = m[0][0], b = m[0][1], c = m[0][2];
    const double d = m[1][0], e = m[1][1], f = m[1][2];
    const double g = m[2][0], h = m[2][1], i = m[2][2];
    const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const double det = a * A + b * B + c * C;
    Mat3 r{};
    if (std::fabs(det) < 1e-12) return r;
    const double k = 1.0 / det;
    r[0] = {A * k, -(b * i - c * h) * k, (b * f - c * e) * k};
    r[1] = {B * k, (a * i - c * g) * k, -(a * f - c * d) * k};
    r[2] = {C * k, -(a * h - b * g) * k, (a * e - b * d) * k};
    return r;
}

Mat3 multiply(const Mat3& x, const Mat3& y) {
    Mat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) r[i][j] += x[i][k] * y[k][j];
    return r;
}

// Linear RGB to CIE XYZ for a set of primaries (SMPTE RP 177).
Mat3 rgb_to_xyz(const Primaries& p) {
    auto col = [](XY c) { return std::array<double, 3>{c.x / c.y, 1.0, (1 - c.x - c.y) / c.y}; };
    const auto R = col(p.r), G = col(p.g), B = col(p.b), W = col(p.w);
    Mat3 P{};
    for (int i = 0; i < 3; ++i) P[i] = {R[std::size_t(i)], G[std::size_t(i)], B[std::size_t(i)]};
    const Mat3 inv = inverse(P);
    std::array<double, 3> s{};
    for (int i = 0; i < 3; ++i) s[std::size_t(i)] = inv[i][0] * W[0] + inv[i][1] * W[1] + inv[i][2] * W[2];
    Mat3 M{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M[i][j] = P[i][j] * s[std::size_t(j)];
    return M;
}

enum class Kind { YCbCr, Identity, YCgCo };

struct Matrix {
    Kind kind = Kind::YCbCr;
    float kr = 0.299f, kb = 0.114f;
};

// H.273 MatrixCoefficients; unspecified reads as BT.601, as AVIF readers do.
Matrix matrix_of(int mc, int cp) {
    Matrix m;
    switch (mc) {
        case 0: m.kind = Kind::Identity; break;
        case 1: m.kr = 0.2126f; m.kb = 0.0722f; break;
        case 4: m.kr = 0.30f; m.kb = 0.11f; break;
        case 7: m.kr = 0.212f; m.kb = 0.087f; break;
        case 8: m.kind = Kind::YCgCo; break;
        case 9:
        case 10: m.kr = 0.2627f; m.kb = 0.0593f; break;
        case 12:
        case 13: {
            // Chroma-derived: the luma weights of the primaries themselves.
            const Mat3 M = rgb_to_xyz(primaries_of(cp));
            m.kr = float(M[1][0]);
            m.kb = float(M[1][2]);
            break;
        }
        default: break;  // 5, 6 (BT.601) and unspecified
    }
    return m;
}

// ---- Transfer functions -----------------------------------------------------

constexpr float kSdrWhiteNits = 203.0f;  // ITU-R BT.2408 reference white

float pq_to_nits(float e) {
    const double m1 = 2610.0 / 16384, m2 = 2523.0 / 4096 * 128;
    const double c1 = 3424.0 / 4096, c2 = 2413.0 / 4096 * 32, c3 = 2392.0 / 4096 * 32;
    const double p = std::pow(std::max(0.0, double(e)), 1 / m2);
    const double num = std::max(p - c1, 0.0);
    return float(10000.0 * std::pow(num / (c2 - c3 * p), 1 / m1));
}

// HLG inverse OETF (BT.2100): the signal to scene-linear [0, 1].
float hlg_to_scene(float e) {
    const double a = 0.17883277, b = 1 - 4 * a, c = 0.5 - a * std::log(4 * a);
    e = std::clamp(e, 0.0f, 1.0f);
    if (e <= 0.5f) return float(double(e) * e / 3);
    return float((std::exp((e - c) / a) + b) / 12);
}

float srgb_encode(float l) {
    if (l <= 0.0031308f) return 12.92f * l;
    return 1.055f * std::pow(l, 1 / 2.4f) - 0.055f;
}

struct SrgbLut {
    std::array<uint8_t, 16384> v{};
    SrgbLut() {
        for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = uint8_t(std::lround(std::clamp(srgb_encode(float(i) / (v.size() - 1)), 0.0f, 1.0f) * 255));
    }
};

const SrgbLut& srgb_lut() {
    static const SrgbLut lut;
    return lut;
}

struct HdrPath {
    bool pq = true;
    std::vector<float> curve;  // signal (12-bit) to linear: PQ relative to SDR white, HLG scene-linear
    float to709[3][3]{};
    bool convert = false;
    float luma[3] = {0.2627f, 0.6780f, 0.0593f};  // HLG OOTF luminance weights (source primaries)
    float white = 1000.0f / kSdrWhiteNits;        // the peak, relative to SDR white
    float knee = 0.75f;

    float tonemap(float m) const {
        if (m <= knee || white <= 1.0f) return std::min(m, 1.0f);
        const float x = (m - knee) / (1 - knee);
        const float wx = (white - knee) / (1 - knee);
        const float y = x * (1 + x / (wx * wx)) / (1 + x);
        return std::min(knee + (1 - knee) * y, 1.0f);
    }
};

HdrPath make_hdr(const ColorSpec& c) {
    HdrPath h;
    h.pq = c.transfer == 16;
    h.curve.resize(4096);
    for (int i = 0; i < 4096; ++i) {
        const float e = float(i) / 4095;
        h.curve[std::size_t(i)] = h.pq ? pq_to_nits(e) / kSdrWhiteNits : hlg_to_scene(e);
    }
    const float peak = c.max_cll ? float(c.max_cll) : 1000.0f;
    h.white = std::max(peak, kSdrWhiteNits) / kSdrWhiteNits;
    const Primaries src = primaries_of(c.primaries == 2 ? 9 : c.primaries);
    const Mat3 srcM = rgb_to_xyz(src);
    h.luma[0] = float(srcM[1][0]);
    h.luma[1] = float(srcM[1][1]);
    h.luma[2] = float(srcM[1][2]);
    const int cp = c.primaries == 2 ? 9 : c.primaries;
    if (cp != 1) {
        const Mat3 m = multiply(inverse(rgb_to_xyz(primaries_of(1))), srcM);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) h.to709[i][j] = float(m[i][j]);
        h.convert = true;
    }
    return h;
}

// ---- Sampling ---------------------------------------------------------------

template <class T>
inline int sample(const YuvPlanes& f, int plane, int x, int y) {
    const auto* row = reinterpret_cast<const T*>(static_cast<const uint8_t*>(f.planes[plane]) +
                                                 std::ptrdiff_t(y) * f.strides[plane]);
    return int(row[x]);
}

inline int sample(const YuvPlanes& f, int plane, int x, int y) {
    return f.bit_depth > 8 ? sample<uint16_t>(f, plane, x, y) : sample<uint8_t>(f, plane, x, y);
}

// One output row of a chroma plane, upsampled to luma width (raw codes).
void chroma_row(const YuvPlanes& f, int plane, int y, int width, std::vector<float>& out) {
    const int cw = (f.width + (1 << f.chroma_shift_x) - 1) >> f.chroma_shift_x;
    const int chh = (f.height + (1 << f.chroma_shift_y) - 1) >> f.chroma_shift_y;
    // Vertical: two chroma rows and the weight of the second.
    int r0 = y, r1 = y;
    float wy = 0;
    if (f.chroma_shift_y) {
        const int c = y >> 1;
        if (f.chroma_position == 2) {  // colocated with even luma rows
            r0 = c;
            r1 = (y & 1) ? std::min(c + 1, chh - 1) : c;
            wy = (y & 1) ? 0.5f : 0.0f;
        } else {  // centred between two luma rows
            r0 = c;
            r1 = (y & 1) ? std::min(c + 1, chh - 1) : std::max(c - 1, 0);
            wy = 0.25f;
        }
    }
    r0 = std::min(r0, chh - 1);
    r1 = std::min(r1, chh - 1);
    out.resize(std::size_t(width));
    if (!f.chroma_shift_x) {
        for (int x = 0; x < width; ++x)
            out[std::size_t(x)] = (1 - wy) * float(sample(f, plane, x, r0)) + wy * float(sample(f, plane, x, r1));
        return;
    }
    const bool colocated = f.chroma_position == 1 || f.chroma_position == 2;
    auto at = [&](int cx) {
        cx = std::clamp(cx, 0, cw - 1);
        return (1 - wy) * float(sample(f, plane, cx, r0)) + wy * float(sample(f, plane, cx, r1));
    };
    for (int x = 0; x < width; ++x) {
        const int c = x >> 1;
        float v;
        if (colocated) v = (x & 1) ? 0.5f * (at(c) + at(c + 1)) : at(c);
        else v = (x & 1) ? 0.75f * at(c) + 0.25f * at(c + 1) : 0.75f * at(c) + 0.25f * at(c - 1);
        out[std::size_t(x)] = v;
    }
}

} // namespace

bool is_hdr_transfer(int transfer) { return transfer == 16 || transfer == 18; }

void yuv_to_rgba(const YuvPlanes& f, const ColorSpec& c, uint8_t* canvas, int cw, int ch, int x0, int y0) {
    const int w = std::min(f.width, cw - x0);
    const int h = std::min(f.height, ch - y0);
    if (w <= 0 || h <= 0 || x0 < 0 || y0 < 0) return;

    const Matrix m = matrix_of(c.matrix, c.primaries);
    const float maxv = float((1 << f.bit_depth) - 1);
    const float scale = float(1 << (f.bit_depth - 8));
    // Normalise: Y' to [0, 1], chroma to [-0.5, 0.5].
    float y_off, y_mul, c_off, c_mul;
    if (c.full_range) {
        y_off = 0;
        y_mul = 1 / maxv;
        c_off = float(1 << (f.bit_depth - 1));
        c_mul = 1 / maxv;
    } else {
        y_off = 16 * scale;
        y_mul = 1 / (219 * scale);
        c_off = 128 * scale;
        c_mul = 1 / (224 * scale);
    }
    const float kr = m.kr, kb = m.kb, kg = 1 - kr - kb;
    const float cr_r = 2 * (1 - kr), cb_b = 2 * (1 - kb);
    const float cb_g = cb_b * kb / kg, cr_g = cr_r * kr / kg;

    const bool hdr = is_hdr_transfer(c.transfer);
    HdrPath hp;
    if (hdr) hp = make_hdr(c);
    const auto& lut = srgb_lut();
    auto to8 = [](float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255)); };
    auto code12 = [](float v) { return std::size_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 4095)); };

    std::vector<float> urow, vrow;
    for (int y = 0; y < h; ++y) {
        if (!f.monochrome) {
            chroma_row(f, 1, y, w, urow);
            chroma_row(f, 2, y, w, vrow);
        }
        uint8_t* out = canvas + (std::size_t(y0 + y) * std::size_t(cw) + std::size_t(x0)) * 4;
        for (int x = 0; x < w; ++x, out += 4) {
            const float Y = (float(sample(f, 0, x, y)) - y_off) * y_mul;
            float r, g, b;
            if (f.monochrome) {
                r = g = b = Y;
            } else if (m.kind == Kind::Identity) {
                // G, B, R stored as Y, U, V; a limited range scales all three like luma.
                g = Y;
                b = (urow[std::size_t(x)] - y_off) * y_mul;
                r = (vrow[std::size_t(x)] - y_off) * y_mul;
            } else {
                const float cb = (urow[std::size_t(x)] - c_off) * c_mul;
                const float cr = (vrow[std::size_t(x)] - c_off) * c_mul;
                if (m.kind == Kind::YCgCo) {
                    const float t = Y - cb;
                    g = Y + cb;
                    b = t - cr;
                    r = t + cr;
                } else {
                    r = Y + cr_r * cr;
                    g = Y - cb_g * cb - cr_g * cr;
                    b = Y + cb_b * cb;
                }
            }
            if (!hdr) {
                out[0] = to8(r);
                out[1] = to8(g);
                out[2] = to8(b);
                out[3] = 255;
                continue;
            }
            float lr = hp.curve[code12(r)], lg = hp.curve[code12(g)], lb = hp.curve[code12(b)];
            if (!hp.pq) {
                // BT.2100 HLG OOTF at a 1000-nit display (system gamma 1.2),
                // then relative to SDR white.
                const float ys = hp.luma[0] * lr + hp.luma[1] * lg + hp.luma[2] * lb;
                const float k = ys > 0 ? 1000.0f * std::pow(ys, 0.2f) / kSdrWhiteNits : 0.0f;
                lr *= k;
                lg *= k;
                lb *= k;
            }
            if (hp.convert) {
                const float nr = hp.to709[0][0] * lr + hp.to709[0][1] * lg + hp.to709[0][2] * lb;
                const float ng = hp.to709[1][0] * lr + hp.to709[1][1] * lg + hp.to709[1][2] * lb;
                const float nb = hp.to709[2][0] * lr + hp.to709[2][1] * lg + hp.to709[2][2] * lb;
                lr = std::max(nr, 0.0f);
                lg = std::max(ng, 0.0f);
                lb = std::max(nb, 0.0f);
            }
            const float mx = std::max({lr, lg, lb});
            if (mx > 0) {
                const float s = hp.tonemap(mx) / mx;
                lr *= s;
                lg *= s;
                lb *= s;
            }
            auto enc = [&](float v) { return lut.v[std::size_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 16383))]; };
            out[0] = enc(lr);
            out[1] = enc(lg);
            out[2] = enc(lb);
            out[3] = 255;
        }
    }
}

void yuv_to_alpha(const YuvPlanes& f, bool full_range, uint8_t* canvas, int cw, int ch, int x0, int y0) {
    const int w = std::min(f.width, cw - x0);
    const int h = std::min(f.height, ch - y0);
    if (w <= 0 || h <= 0 || x0 < 0 || y0 < 0) return;
    const float maxv = float((1 << f.bit_depth) - 1);
    const float scale = float(1 << (f.bit_depth - 8));
    const float off = full_range ? 0 : 16 * scale;
    const float mul = full_range ? 1 / maxv : 1 / (219 * scale);
    for (int y = 0; y < h; ++y) {
        uint8_t* out = canvas + (std::size_t(y0 + y) * std::size_t(cw) + std::size_t(x0)) * 4 + 3;
        for (int x = 0; x < w; ++x, out += 4) {
            const float a = (float(sample(f, 0, x, y)) - off) * mul;
            *out = uint8_t(std::lround(std::clamp(a, 0.0f, 1.0f) * 255));
        }
    }
}

}  // namespace broimage::detail
