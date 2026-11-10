// Per-rollout drive costs and late-crash freeze of the HW planner (HWM_BACKEND = "cpp").
//
// The same terms as TrainingLite/hwm/drive_costs_jax.py: waypoint j takes the closest rollout state no
// earlier waypoint took, the matched-waypoint cost, the first body-corner crash, the corner clearance and
// the depth into the crash margin. float32 like the JAX code, the body corners and the map lookup in
// float64. Built without -ffast-math (inf and NaN are part of the rules).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "hw_kernels.h"
#include "hw_pool.h"

namespace hw {
Pool& pool() {
    static Pool p;
    return p;
}
}  // namespace hw

namespace {

using hwk::D;
using hwk::PD;
using hwk::TH;
using hwk::VX;
using hwk::VY;
using hwk::X;
using hwk::Y;

struct Map {
    bool blank = false;
    std::vector<float> grid;  // wall distance [m], row 0 = lowest y
    int h = 0, w = 0;
    double ox = 0, oy = 0, oc = 1, os = 0, res = 1;
    double cx[4] = {0, 0, 0, 0}, cy[4] = {0, 0, 0, 0};
} g_map;

// Wall distance under a world point, and whether it is on the map (WallGeometry._grid_cells).
inline float wall_at(double x, double y, bool* inside) {
    const Map& m = g_map;
    const double xt = x - m.ox, yt = y - m.oy;
    const double xr = xt * m.oc + yt * m.os;
    const double yr = -xt * m.os + yt * m.oc;
    *inside = xr >= 0 && xr < m.w * m.res && yr >= 0 && yr < m.h * m.res;
    double col = std::floor(xr / m.res), row = std::floor(yr / m.res);
    col = std::isnan(col) ? 0.0 : std::min(std::max(col, 0.0), (double)(m.w - 1));
    row = std::isnan(row) ? 0.0 : std::min(std::max(row, 0.0), (double)(m.h - 1));
    return m.grid[(size_t)row * m.w + (size_t)col];
}

// Smallest wall distance of the four body corners (0 off the map), and whether one is within ``border``.
inline float corner_clearance(const float* s, float border, bool* crashed) {
    if (g_map.blank) {
        *crashed = false;
        return std::numeric_limits<float>::infinity();
    }
    const double px = s[X], py = s[Y], th = s[TH];
    const double c = std::cos(th), sn = std::sin(th);
    float least = std::numeric_limits<float>::infinity();
    bool hit = false;
    for (int i = 0; i < 4; ++i) {
        const double x = px + c * g_map.cx[i] - sn * g_map.cy[i];
        const double y = py + sn * g_map.cx[i] + c * g_map.cy[i];
        bool inside;
        const float d = wall_at(x, y, &inside);
        hit |= !inside || d < border;
        least = std::min(least, inside ? d : 0.f);
    }
    *crashed = hit;
    return least;
}

struct Consts {
    float target_speed_factor, overspeed_weight, arrival_weight, border, clearance, clearance_span;
    int crash_end;
};

void costs_one(const float* states, int steps, const float* match_xy, const float* rows, int count, const Consts& c,
               float* drive_out, uint8_t* active_out, int32_t* late_out, float* closeness_out, float* depth_out,
               std::vector<float>& dist, std::vector<uint8_t>& taken) {
    const float inf = std::numeric_limits<float>::infinity();
    // Squared distances, rounded like drive_costs_jax.squared_distances (no fused multiply-add).
    dist.assign((size_t)steps * count, inf);
    for (int t = 0; t < steps; ++t) {
        const float x = states[t * D + X], y = states[t * D + Y];
        for (int j = 0; j < count; ++j) {
            const float dx = x - match_xy[2 * j], dy = y - match_xy[2 * j + 1];
            const float a = dx * dx, b = dy * dy;
            dist[(size_t)t * count + j] = a + b;
        }
    }
    taken.assign(steps, 0);
    float distance = 0.f, heading = 0.f, drift = 0.f, shortfall = 0.f, overspeed = 0.f;
    float fast = -inf;
    int valid_count = 0;
    for (int j = 0; j < count; ++j) {
        int best = 0;
        float best_d = inf;
        for (int t = 0; t < steps; ++t) {
            const float d = taken[t] ? inf : dist[(size_t)t * count + j];
            if (d < best_d) {  // first minimum, like argmin
                best_d = d;
                best = t;
            }
        }
        if (!std::isfinite(best_d)) continue;
        taken[best] = 1;
        ++valid_count;
        const float* s = states + best * D;
        const float* r = rows + 5 * j;
        const float px = s[X] - r[0], py = s[Y] - r[1];
        distance += std::sqrt(px * px + py * py);
        const float e = s[TH] - r[2];
        heading += std::fabs(std::atan2(std::sin(e), std::cos(e)));
        const float target = r[3] * c.target_speed_factor;
        shortfall += std::max(target - s[VX], 0.f);
        overspeed += std::max(s[VX] - target, 0.f);
        drift += std::fabs(s[VY] - r[4]);
        fast = std::max(fast, (float)best);
    }
    const float counts = (float)std::max(valid_count, 1);
    float drive = (10.f * distance + 2.f * heading + shortfall + overspeed * c.overspeed_weight + 0.25f * drift +
                   c.arrival_weight * fast) /
                  counts;
    float min_speed = inf;
    for (int t = 1; t < steps; ++t) min_speed = std::min(min_speed, states[t * D + VX]);
    if (min_speed < 0.f) drive += 100.f * std::fabs(min_speed);

    int survived = steps;
    float closeness = 0.f, depth = 0.f;
    for (int t = 0; t < steps; ++t) {
        bool crashed;
        const float least = corner_clearance(states + t * D, c.border, &crashed);
        if (crashed && survived == steps) survived = t;
        closeness += std::min(std::max((c.clearance - least) / c.clearance_span, 0.f), 1.f);
        depth += std::min(std::max((c.border - least) / c.border, 0.f), 1.f);
    }
    *drive_out = valid_count > 0 ? drive : 0.f;
    *active_out = valid_count > 0;
    *late_out = std::max(c.crash_end - survived, 0);
    *closeness_out = closeness;
    *depth_out = depth;
}

}  // namespace

namespace hwk {

void set_map(const float* grid, int h, int w, double ox, double oy, double oc, double os, double res,
             const double* cx, const double* cy) {
    g_map.blank = false;
    g_map.grid.assign(grid, grid + (size_t)h * w);
    g_map.h = h;
    g_map.w = w;
    g_map.ox = ox;
    g_map.oy = oy;
    g_map.oc = oc;
    g_map.os = os;
    g_map.res = res;
    for (int i = 0; i < 4; ++i) {
        g_map.cx[i] = cx[i];
        g_map.cy[i] = cy[i];
    }
}

void set_blank_map() { g_map.blank = true; }
bool map_blank() { return g_map.blank; }
int map_width() { return g_map.w; }

void map_cell(double x, double y, int* row, int* col, bool* inside) {
    const Map& m = g_map;
    const double xt = x - m.ox, yt = y - m.oy;
    const double xr = xt * m.oc + yt * m.os;
    const double yr = -xt * m.os + yt * m.oc;
    *inside = xr >= 0 && xr < m.w * m.res && yr >= 0 && yr < m.h * m.res;
    double c = std::floor(xr / m.res), r = std::floor(yr / m.res);
    c = std::isnan(c) ? 0.0 : std::min(std::max(c, 0.0), (double)(m.w - 1));
    r = std::isnan(r) ? 0.0 : std::min(std::max(r, 0.0), (double)(m.h - 1));
    *row = (int)r;
    *col = (int)c;
}

float wall_distance(double x, double y) {
    if (g_map.blank) return std::numeric_limits<float>::infinity();
    bool inside;
    const float d = wall_at(x, y, &inside);
    return inside ? d : 0.f;
}

float corner_clearance(const float* s, float border, bool* crashed) { return ::corner_clearance(s, border, crashed); }

void freeze_late(float* traj, int batch, int steps, int end, float border) {
    if (end >= steps || g_map.blank) return;
    hw::pool().run((batch + 31) / 32, [&](int b) {
        const int hi = std::min(batch, (b + 1) * 32);
        for (int k = b * 32; k < hi; ++k) {
            float* s = traj + (size_t)k * steps * D;
            int first = steps;
            for (int t = 0; t < steps; ++t) {
                bool crashed;
                ::corner_clearance(s + t * D, border, &crashed);
                if (crashed) {
                    first = t;
                    break;
                }
            }
            if (first < end || first >= steps) continue;
            float frozen[D];
            std::copy(s + (first - 1) * D, s + first * D, frozen);
            frozen[VX] = frozen[VY] = frozen[PD] = 0.f;
            for (int t = first; t < steps; ++t) std::copy(frozen, frozen + D, s + t * D);
        }
    });
}

void drive_costs(const float* states, int batch, int steps, const float* match_xy, const float* rows, int count,
                 const float* consts, int crash_end, float* drive, uint8_t* active, int32_t* late, float* closeness,
                 float* depth) {
    Consts c{consts[0], consts[1], consts[2], consts[3], consts[4], consts[5], crash_end};
    const int chunk = 16;
    hw::pool().run((batch + chunk - 1) / chunk, [&](int b) {
        thread_local std::vector<float> dist;
        thread_local std::vector<uint8_t> taken;
        const int hi = std::min(batch, (b + 1) * chunk);
        for (int k = b * chunk; k < hi; ++k)
            costs_one(states + (size_t)k * steps * D, steps, match_xy, rows, count, c, drive + k, active + k,
                      late + k, closeness + k, depth + k, dist, taken);
    });
}

}  // namespace hwk

extern "C" {

void hw_set_threads(int threads) { hw::pool().resize(threads); }

int hw_threads() { return hw::pool().threads(); }

// grid: (h, w) wall distance, origin and resolution as WallGeometry, corners (4,) in the body frame.
void hw_set_map(const float* grid, int h, int w, double ox, double oy, double oc, double os, double res,
                const double* cx, const double* cy) {
    hwk::set_map(grid, h, w, ox, oy, oc, os, res, cx, cy);
}

void hw_set_blank_map() { hwk::set_blank_map(); }

void hw_freeze_late(float* traj, int batch, int steps, int end, float border) {
    hwk::freeze_late(traj, batch, steps, end, border);
}

void hw_drive_costs(const float* states, int batch, int steps, const float* match_xy, const float* rows, int count,
                    const float* consts, int crash_end, float* drive, uint8_t* active, int32_t* late,
                    float* closeness, float* depth) {
    hwk::drive_costs(states, batch, steps, match_xy, rows, count, consts, crash_end, drive, active, late, closeness,
                     depth);
}

// Correlated samples around ``nominal`` (H, 2), clipped (HirarchicalPlanner._sample_mpc_actions):
// a_0 = nominal_0 + std * eps_0, a_t = rho a_{t-1} + (1 - rho) nominal_t + sqrt(1 - rho^2) std eps_t.
void hw_correlated_samples(const float* nominal, const float* eps, int batch, int horizon, float std0, float std1,
                           float rho, float lo1, float hi1, float* out) {
    const float fresh = std::sqrt(1.f - rho * rho);
    const float stdv[2] = {std0, std1};
    hw::pool().run((batch + 63) / 64, [&](int b) {
        const int hi = std::min(batch, (b + 1) * 64);
        for (int k = b * 64; k < hi; ++k) {
            const float* e = eps + (size_t)k * horizon * 2;
            float* o = out + (size_t)k * horizon * 2;
            for (int i = 0; i < 2; ++i) o[i] = nominal[i] + e[i] * stdv[i];
            if (rho > 0.f) {
                for (int t = 1; t < horizon; ++t)
                    for (int i = 0; i < 2; ++i)
                        o[2 * t + i] = o[2 * (t - 1) + i] * rho + (1.f - rho) * nominal[2 * t + i] +
                                       fresh * stdv[i] * e[2 * t + i];
            } else {
                for (int t = 1; t < horizon; ++t)
                    for (int i = 0; i < 2; ++i) o[2 * t + i] = nominal[2 * t + i] + e[2 * t + i] * stdv[i];
            }
            for (int t = 0; t < horizon; ++t) {
                o[2 * t] = std::min(std::max(o[2 * t], -1.f), 1.f);
                o[2 * t + 1] = std::min(std::max(o[2 * t + 1], lo1), hi1);
            }
        }
    });
}

}  // extern "C"
