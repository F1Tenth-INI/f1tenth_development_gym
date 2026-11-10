// Branch-free float math for the rollout lanes (Cephes polynomials, about 1e-7 relative).
//
// Inline and free of libm calls, so the compiler vectorizes a loop over lanes with them on any
// CPU (SSE/AVX, NEON) and any C library; glibc's vector libm (libmvec) is not needed. Compiled
// without -ffast-math: the range reductions depend on the order of their operations.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace hwm {

constexpr float PI = 3.14159265358979323846f;
constexpr float PI_2 = 1.57079632679489661923f;
constexpr float PI_4 = 0.78539816339744830962f;

inline float bits_to_float(int32_t i) {
    float f;
    std::memcpy(&f, &i, sizeof f);
    return f;
}

// atan on [0, 1].
inline float atan01(float a) {
    const bool big = a > 0.41421356237f;  // tan(pi / 8)
    const float x = big ? (a - 1.f) / (a + 1.f) : a;
    const float z = x * x;
    const float p = (((8.05374449538e-2f * z - 1.38776856032e-1f) * z + 1.99777106478e-1f) * z - 3.33329491539e-1f) * z * x + x;
    return big ? PI_4 + p : p;
}

inline float atan(float x) {
    const float ax = std::fabs(x);
    const bool inv = ax > 1.f;
    const float r0 = atan01(inv ? 1.f / ax : ax);
    const float r = inv ? PI_2 - r0 : r0;
    return x < 0.f ? -r : r;
}

inline float atan2(float y, float x) {
    const float ax = std::fabs(x), ay = std::fabs(y);
    const float mx = ax > ay ? ax : ay, mn = ax > ay ? ay : ax;
    float r = atan01(mx > 0.f ? mn / mx : 0.f);
    r = ay > ax ? PI_2 - r : r;
    r = x < 0.f ? PI - r : r;
    return y < 0.f ? -r : r;
}

// sin and cos, for |x| up to a few hundred.
inline void sincos(float x, float* s, float* c) {
    const float j = std::floor(x * (2.f / PI) + 0.5f);
    const int q = (int)j;
    const float r = ((x - j * 1.5703125f) - j * 4.837512969970703125e-4f) - j * 7.54978995489188216e-8f;
    const float z = r * r;
    const float sp = ((-1.9515295891e-4f * z + 8.3321608736e-3f) * z - 1.6666654611e-1f) * z * r + r;
    const float cp = ((2.443315711809948e-5f * z - 1.388731625493765e-3f) * z + 4.166664568298827e-2f) * z * z - 0.5f * z + 1.f;
    const int quadrant = q & 3;
    const float sv = (quadrant & 1) ? cp : sp;
    const float cv = (quadrant & 1) ? sp : cp;
    *s = (quadrant & 2) ? -sv : sv;
    *c = ((quadrant == 1) | (quadrant == 2)) ? -cv : cv;
}

inline float sin(float x) {
    float s, c;
    sincos(x, &s, &c);
    return s;
}

inline float cos(float x) {
    float s, c;
    sincos(x, &s, &c);
    return c;
}

inline float tan(float x) {
    float s, c;
    sincos(x, &s, &c);
    return s / c;
}

inline float exp(float x) {
    x = x < -87.f ? -87.f : (x > 88.f ? 88.f : x);
    const float fx = std::floor(x * 1.44269504088896341f + 0.5f);
    const float r = (x - fx * 0.693359375f) - fx * -2.12194440e-4f;
    const float z = r * r;
    const float p = ((((((1.9875691500e-4f * r + 1.3981999507e-3f) * r + 8.3334519073e-3f) * r + 4.1665795894e-2f) * r +
                       1.6666665459e-1f) * r + 5.0000001201e-1f) * z + r) + 1.f;
    return p * bits_to_float(((int32_t)fx + 127) << 23);
}

inline float tanh(float x) {
    const float e = hwm::exp(2.f * std::fabs(x));
    const float t = 1.f - 2.f / (e + 1.f);
    return x < 0.f ? -t : t;
}

inline float sigmoid(float x) { return 1.f / (1.f + hwm::exp(-x)); }

}  // namespace hwm
