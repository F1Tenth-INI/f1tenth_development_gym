// Pacejka rollouts of the HW planner (HWM_BACKEND = "cpp").
//
// The same model as sim/f110_sim/envs/dynamic_model_pacejka_jax.py (ODE:pacejka, ODE:ks and
// ODE:ks_pacejka, acceleration control) and the same control step as
// HirarchicalPlanner._pacejka_scan_rollout: the command issued CONTROL_DELAY earlier is applied,
// denormalized, its translational part clipped, then TIMESTEP_CONTROL / TIMESTEP_SIM substeps.
//
// The samples are stepped in blocks of LANES, one lane per sample, so the compiler can vectorize
// the step over the samples; hw_math.h has the vectorizable sin/cos/atan/exp. The
// rewrites are exact in real arithmetic: cos and sin of a wrapped angle from the unwrapped one,
// cos(atan(t)) = 1 / sqrt(1 + t^2), cos(atan2(y, x)) = x / hypot(x, y).

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "hw_kernels.h"
#include "hw_math.h"
#include "hw_pool.h"

namespace {

using hwk::D;
constexpr int LANES = 16;

struct Car {
    float mu, lf, lr, h, m, Iz, g, Bf, Cf, Df, Ef, Br, Cr, Dr, Er;
    float servo_p, s_min, s_max, sv_min, sv_max, a_min, a_max, v_min, v_max, v_switch;
    float steer_diff_low, c_rr, v_dead, curve_res, brake_mult, accel_mult, steer_mult;
};

inline float clampf(float x, float lo, float hi) { return std::min(std::max(x, lo), hi); }
inline float signf(float x) { return x > 0.f ? 1.f : (x < 0.f ? -1.f : 0.f); }

// One substep of LANES samples, in place. MODE: 0 pacejka, 1 ks, 2 ks_pacejka. HAS_E: a Pacejka E
// coefficient is not 0. LAST: also the outputs that are not inputs of the next substep (cos, sin, slip).
// Branch free, so the lane loop vectorizes.
template <int MODE, bool HAS_E, bool LAST>
void substep(const Car& c, float dt_sub,
             float* __restrict pd, float* __restrict vx, float* __restrict vy, float* __restrict psi,
             float* __restrict cs, float* __restrict sn, float* __restrict sx, float* __restrict sy,
             float* __restrict slip, float* __restrict delta,
             const float* __restrict steer_cmd, const float* __restrict trans_cmd) {
    const float mug = c.mu * c.g;
    const float lwb = c.lf + c.lr;
    const float lr_over_lwb = c.lr / lwb;
    {
#pragma omp simd
        for (int l = 0; l < LANES; ++l) {
            const float p_d = pd[l], v_x = vx[l], v_y = vy[l], ps = psi[l], s_x = sx[l], s_y = sy[l], dl = delta[l];

            // Servo and steering limits.
            const float desired = steer_cmd[l] * c.steer_mult;
            const float diff = desired - dl;
            float ddot = std::fabs(diff) > c.steer_diff_low ? diff * c.servo_p : 0.f;
            ddot = clampf(ddot, c.sv_min, c.sv_max);
            const bool at_min = (dl <= c.s_min) & (ddot <= 0.f);
            const bool at_max = (dl >= c.s_max) & (ddot >= 0.f);
            ddot = (at_min | at_max) ? 0.f : ddot;
            ddot = clampf(ddot, c.sv_min, c.sv_max);

            // Longitudinal: brake/accel multipliers, acceleration limits, rolling resistance.
            float a = trans_cmd[l];
            const float th = hwm::tanh(25.f * a);
            const float sig_neg = 0.5f * (1.f - th), sig_pos = 0.5f * (1.f + th);
            const float neutral = std::max(0.f, 1.f - sig_neg - sig_pos);
            a = a * (sig_neg * c.brake_mult + sig_pos * c.accel_mult + neutral);
            const float vel = v_x == 0.f ? 1.0e-4f : v_x;
            const float pos_limit = vel > c.v_switch ? c.a_max * c.v_switch / vel : c.a_max;
            a = (((v_x < c.v_min) & (a < 0.f)) | ((v_x > c.v_max) & (a > 0.f))) ? 0.f : a;
            a = clampf(a, c.a_min, pos_limit);
            a = clampf(a, -mug, mug);
            const float smooth_sign = v_x / std::sqrt(v_x * v_x + c.v_dead * c.v_dead);
            a = a - c.c_rr * c.g * smooth_sign;
            a = clampf(a, -mug, mug);

            float sd, cd;
            hwm::sincos(dl, &sd, &cd);
            const float new_delta = clampf(dl + dt_sub * ddot, c.s_min, c.s_max);

            // Dynamic single-track model with Pacejka tyres and a friction ellipse.
            const float vxs = std::max(std::fabs(v_x), 1.0e-3f);
            const float alpha_f = -hwm::atan((v_y + p_d * c.lf) / vxs) + signf(v_x) * dl;
            const float alpha_r = -hwm::atan((v_y - p_d * c.lr) / vxs);
            const float fzf = c.m * (-a * c.h + c.g * c.lr) / (c.lr + c.lf);
            const float fzr = c.m * (a * c.h + c.g * c.lf) / (c.lr + c.lf);
            const float bf = c.Bf * alpha_f, br = c.Br * alpha_r;
            const float inner_f = HAS_E ? bf - c.Ef * (bf - hwm::atan(bf)) : bf;
            const float inner_r = HAS_E ? br - c.Er * (br - hwm::atan(br)) : br;
            float fyf = c.mu * fzf * c.Df * hwm::sin(c.Cf * hwm::atan(inner_f));
            float fyr = c.mu * fzr * c.Dr * hwm::sin(c.Cr * hwm::atan(inner_r));
            const float ratio = a / mug;
            const float lat = std::sqrt(std::max(1.f - ratio * ratio, 1.0e-4f));
            fyf *= lat;
            fyr *= lat;
            float a_p = a - c.curve_res * std::sqrt(fyf * fyf + fyr * fyr) / c.m * smooth_sign;
            a_p = clampf(a_p, -mug, mug);
            float sp, cp;
            hwm::sincos(ps, &sp, &cp);
            const float p_sx = s_x + dt_sub * (v_x * cp - v_y * sp);
            const float p_sy = s_y + dt_sub * (v_x * sp + v_y * cp);
            const float p_vx = v_x + dt_sub * (a_p - fyf * sd / c.m + v_y * p_d);
            const float p_vy = v_y + dt_sub * ((fyr + fyf * cd) / c.m - v_x * p_d);
            const float p_psi = ps + dt_sub * p_d;
            const float p_pd = p_d + dt_sub * ((-c.lr * fyr + c.lf * fyf * cd) / c.Iz);

            // Kinematic bicycle at the CoG.
            const float v = (v_x < 0.f ? -1.f : 1.f) * std::sqrt(v_x * v_x + v_y * v_y);
            const float td = hwm::tan(dl);
            const float t1 = td * lr_over_lwb;
            const float cb = 1.f / std::sqrt(1.f + t1 * t1), sb = t1 * cb;
            const float k_sx = s_x + dt_sub * v * (cp * cb - sp * sb);
            const float k_sy = s_y + dt_sub * v * (sp * cb + cp * sb);
            const float k_psi = ps + dt_sub * (v * cb * td / lwb);
            const float v2 = v + dt_sub * a;
            const float td2 = hwm::tan(new_delta);
            const float t2 = td2 * lr_over_lwb;
            const float cb2 = 1.f / std::sqrt(1.f + t2 * t2);
            const float k_vx = v2 * cb2, k_vy = v2 * t2 * cb2;
            const float k_pd = k_vx * td2 / lwb;

            // Blend: kinematic only when slow and not sliding.
            float w;
            if (MODE == 0) {
                w = 1.f;
            } else if (MODE == 1) {
                w = 0.f;
            } else {
                const float speed = std::sqrt(v_x * v_x + v_y * v_y);
                const float slide_f = (v_y + c.lf * p_d) * cd - v_x * sd;
                const float slide_r = v_y - c.lr * p_d;
                const float slide = std::max(std::fabs(slide_f), std::fabs(slide_r));
                const float w_speed = hwm::sigmoid(4.8f * (speed - 1.75f));
                const float w_slide = hwm::sigmoid(10.f * (slide - 0.5f));
                w = 1.f - (1.f - w_speed) * (1.f - w_slide);
            }
            const float u = 1.f - w;
            float sk, ck, spp, cpp;
            hwm::sincos(k_psi, &sk, &ck);
            hwm::sincos(p_psi, &spp, &cpp);
            const float sin_yaw = u * sk + w * spp;
            const float cos_yaw = u * ck + w * cpp;
            const float r = std::sqrt(sin_yaw * sin_yaw + cos_yaw * cos_yaw);
            pd[l] = u * k_pd + w * p_pd;
            vx[l] = u * k_vx + w * p_vx;
            vy[l] = u * k_vy + w * p_vy;
            psi[l] = hwm::atan2(sin_yaw, cos_yaw);
            sx[l] = u * k_sx + w * p_sx;
            sy[l] = u * k_sy + w * p_sy;
            delta[l] = new_delta;
            if (LAST) {
                cs[l] = cos_yaw / r;
                sn[l] = sin_yaw / r;
                const float k_slip = hwm::atan(k_vy / (k_vx < 1.0e-3f ? 1.0e-3f : k_vx));
                const float p_slip = hwm::atan(p_vy / (p_vx < 1.0e-3f ? 1.0e-3f : p_vx));
                slip[l] = u * k_slip + w * p_slip;
            }
        }
    }
}

template <int MODE, bool HAS_E>
void control_step_t(const Car& c, float dt_sub, int substeps, float* const* st, const float* steer, const float* trans) {
    for (int sub = 0; sub < substeps - 1; ++sub)
        substep<MODE, HAS_E, false>(c, dt_sub, st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[8], st[9], steer, trans);
    substep<MODE, HAS_E, true>(c, dt_sub, st[0], st[1], st[2], st[3], st[4], st[5], st[6], st[7], st[8], st[9], steer, trans);
}

// One control step of LANES samples, in place. mode: 0 pacejka, 1 ks, 2 ks_pacejka.
void control_step(const Car& c, int mode, float dt_sub, int substeps, float* const* st, const float* steer,
                  const float* trans) {
    const bool has_e = c.Ef != 0.f || c.Er != 0.f;
    switch (mode * 2 + (has_e ? 1 : 0)) {
        case 0: control_step_t<0, false>(c, dt_sub, substeps, st, steer, trans); break;
        case 1: control_step_t<0, true>(c, dt_sub, substeps, st, steer, trans); break;
        case 2: control_step_t<1, false>(c, dt_sub, substeps, st, steer, trans); break;
        case 3: control_step_t<1, true>(c, dt_sub, substeps, st, steer, trans); break;
        case 4: control_step_t<2, false>(c, dt_sub, substeps, st, steer, trans); break;
        default: control_step_t<2, true>(c, dt_sub, substeps, st, steer, trans); break;
    }
}

struct RolloutArgs {
    Car car;
    int mode, substeps, batch, horizon, delay;
    float dt_sub, denorm0, denorm1, cap;
    const float* s0;       // (B, D)
    const float* actions;  // (B, H, 2)
    const float* pending;  // (B, delay, 2)
    float* out;            // (B, H + 1, D)
};

void rollout_block(const RolloutArgs& a, int lo) {
    const int n = std::min(LANES, a.batch - lo);
    float st[D][LANES];
    float steer[LANES], trans[LANES];
    for (int l = 0; l < LANES; ++l) {
        const int k = lo + std::min(l, n - 1);  // padding lanes repeat the last sample
        for (int i = 0; i < D; ++i) st[i][l] = a.s0[k * D + i];
    }
    for (int l = 0; l < n; ++l)
        for (int i = 0; i < D; ++i) a.out[((size_t)(lo + l) * (a.horizon + 1)) * D + i] = st[i][l];
    for (int t = 0; t < a.horizon; ++t) {
        for (int l = 0; l < LANES; ++l) {
            const int k = lo + std::min(l, n - 1);
            const float* cmd = t < a.delay ? a.pending + ((size_t)k * a.delay + t) * 2
                                           : a.actions + ((size_t)k * a.horizon + (t - a.delay)) * 2;
            steer[l] = cmd[0] * a.denorm0;
            trans[l] = clampf(cmd[1] * a.denorm1, -a.cap, a.cap);
        }
        float* lanes[D];
        for (int i = 0; i < D; ++i) lanes[i] = st[i];
        control_step(a.car, a.mode, a.dt_sub, a.substeps, lanes, steer, trans);
        for (int l = 0; l < n; ++l) {
            float* o = a.out + ((size_t)(lo + l) * (a.horizon + 1) + t + 1) * D;
            for (int i = 0; i < D; ++i) o[i] = st[i][l];
        }
    }
}

}  // namespace

namespace hwk {

void rollout(const CarModel& m, int batch, int horizon, int delay, const float* s0, const float* actions,
             const float* pending, float* out) {
    RolloutArgs a;
    static_assert(sizeof(Car) == 32 * sizeof(float), "Car layout");
    std::copy(m.car, m.car + 32, reinterpret_cast<float*>(&a.car));
    a.mode = m.mode;
    a.substeps = m.substeps;
    a.dt_sub = m.dt / (float)m.substeps;
    a.batch = batch;
    a.horizon = horizon;
    a.delay = delay;
    a.denorm0 = m.denorm0;
    a.denorm1 = m.denorm1;
    a.cap = m.cap;
    a.s0 = s0;
    a.actions = actions;
    a.pending = pending;
    a.out = out;
    const int blocks = (batch + LANES - 1) / LANES;
    hw::pool().run(blocks, [&](int b) { rollout_block(a, b * LANES); });
}

}  // namespace hwk

extern "C" {

// car: VehicleParameters.to_np_array() (32 floats). mode: 0 pacejka, 1 ks, 2 ks_pacejka.
void hw_rollout(const float* car, int mode, float dt, int substeps, float denorm0, float denorm1, float cap,
                int batch, int horizon, int delay, const float* s0, const float* actions, const float* pending,
                float* out) {
    hwk::CarModel m;
    std::copy(car, car + 32, m.car);
    m.mode = mode;
    m.dt = dt;
    m.substeps = substeps;
    m.denorm0 = denorm0;
    m.denorm1 = denorm1;
    m.cap = cap;
    hwk::rollout(m, batch, horizon, delay, s0, actions, pending, out);
}

}  // extern "C"
