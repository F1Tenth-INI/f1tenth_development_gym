// The HW planner (TrainingLite/hwm/hirarchical_planner.py) in C++: HWM_BACKEND = "cpp".
//
// One control step is ``hwp_step``: the same select_action as the Python planner (recovery and stall,
// waypoint-seeded sampling, fallback samples, the window update of the waypoint table, the late-crash
// freeze, matched-waypoint and recovery costs, softmax, safe mean, filling empty waypoints, the next
// seed), then the command is recorded for the control delay. ``hwp_lap`` and ``hwp_episode_done`` are
// report_lap_time and the episode end (permanent freeze, best-lap reset, revert). Function and variable
// names follow the Python code so the two can be read side by side.
//
// The waypoint table lives here; Python reads it through zero-copy views (hwp_table, ...).

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hw_kernels.h"
#include "hw_pool.h"

namespace {

using namespace hwk;

constexpr float INF = std::numeric_limits<float>::infinity();
constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

// Waypoint table columns (hirarchical_planner.py WP_*_IDX).
constexpr int NC = 14;
constexpr int WX = 1, WY = 2, WHEAD = 3, WSPEED = 5, WDRIFT = 10, WSTEER = 11, WYAW = 12, WSLIP = 13;

inline float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }

// ---------------------------------------------------------------------------------------------
// Configuration: "KEY v1 v2 ..." lines, values are numbers.
// ---------------------------------------------------------------------------------------------
class Config {
public:
    explicit Config(const char* text) {
        std::istringstream in(text);
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream ls(line);
            std::string key;
            if (!(ls >> key)) continue;
            std::vector<double> values;
            double v;
            while (ls >> v) values.push_back(v);
            values_[key] = values;
        }
    }
    double num(const std::string& k) const {
        const auto& v = list(k);
        if (v.size() != 1) throw std::runtime_error("hw_planner: " + k + " must be one number");
        return v[0];
    }
    const std::vector<double>& list(const std::string& k) const {
        auto it = values_.find(k);
        if (it == values_.end()) throw std::runtime_error("hw_planner: missing setting " + k);
        return it->second;
    }

private:
    std::map<std::string, std::vector<double>> values_;
};

struct Settings {
    int H, H_free, K, delay, ignore_steps, track_window, span, lag, downstream, w_samples, driven_memory,
        w_history, fragile_ahead, freeze_laps, best_lap_reset, revert_ahead, revert_behind, stall_reverse_steps,
        stall_backtrack_wps, crash_horizon;
    bool track_field, fallback, stall_on_blocked, safe_mean, verbose;
    float rho, temperature, noise[2], denorm[2], cap, brake_limit, accel_limit, brake_multiplier, dt;
    float match_share, target_speed_factor, overspeed_w, arrival_w, border, clearance, ramp_end, wall_w, depth_w;
    float seed_speed_gain, creep_speed, creep_accel, back_off_speed, blocked_distance;
    float detour_start, detour_span, update_distance, recovery_heading_error, off_track_grace;
    float stall_distance, stall_time, ema_gamma, reward_drop;
    float w_noise, w_corr, w_tol, w_margin, w_min_margin, w_error[5], w_error_cost, line_reach, fill_min_speed,
        w_max_slip, fragile_share, fragile_memory, fragile_gain;
    struct {
        float window, distance, heading, steer_in, steer_in_max, speed_min, gate, noise_scale, speed_min_w,
            speed_max, speed_max_w, reverse, slip, yaw_rate;
    } rec;
    std::vector<float> fallback_steer_scales, shift_steps, w_steer_scales;
    uint64_t seed;

    explicit Settings(const Config& c) {
        auto i = [&](const char* k) { return (int)std::lround(c.num(k)); };
        auto f = [&](const char* k) { return (float)c.num(k); };
        auto fl = [&](const char* k) {
            std::vector<float> out;
            for (double v : c.list(k)) out.push_back((float)v);
            return out;
        };
        H = i("H"); H_free = i("H_FREE"); K = i("K"); delay = i("DELAY"); ignore_steps = i("IGNORE_STEPS");
        track_window = i("TRACK_WINDOW"); span = std::max(1, i("SPAN")); lag = i("LAG");
        downstream = std::max(1, i("DOWNSTREAM")); w_samples = i("W_SAMPLES"); driven_memory = i("DRIVEN_MEMORY");
        w_history = i("W_HISTORY"); fragile_ahead = i("FRAGILE_AHEAD"); freeze_laps = i("FREEZE_LAPS");
        best_lap_reset = i("BEST_LAP_RESET"); revert_ahead = std::max(0, i("REVERT_AHEAD"));
        revert_behind = std::max(0, i("REVERT_BEHIND")); stall_reverse_steps = i("STALL_REVERSE_STEPS");
        stall_backtrack_wps = i("STALL_BACKTRACK_WPS"); crash_horizon = i("CRASH_HORIZON");
        track_field = i("TRACK_FIELD") != 0; fallback = i("FALLBACK") != 0; stall_on_blocked = i("STALL_ON_BLOCKED") != 0;
        safe_mean = i("SAFE_MEAN") != 0; verbose = i("VERBOSE") != 0;
        rho = std::min(std::max(f("RHO"), 0.f), 0.999f); temperature = f("TEMPERATURE");
        auto n = fl("DRIVE_NOISE"); noise[0] = n[0]; noise[1] = n[1];
        auto d = fl("DENORM"); denorm[0] = d[0]; denorm[1] = d[1];
        cap = f("CAP"); brake_limit = f("BRAKE_LIMIT"); accel_limit = f("ACCEL_LIMIT");
        brake_multiplier = f("BRAKE_MULTIPLIER"); dt = f("DT");
        match_share = f("MATCH_SHARE"); target_speed_factor = f("TARGET_SPEED_FACTOR"); overspeed_w = f("OVERSPEED_W");
        arrival_w = f("ARRIVAL_W"); border = f("BORDER"); clearance = f("CLEARANCE"); ramp_end = f("RAMP_END");
        wall_w = f("WALL_W"); depth_w = f("DEPTH_W"); seed_speed_gain = f("SEED_SPEED_GAIN");
        creep_speed = f("CREEP_SPEED"); creep_accel = f("CREEP_ACCEL"); back_off_speed = f("BACK_OFF_SPEED");
        blocked_distance = f("BLOCKED_DISTANCE"); detour_start = f("DETOUR_START"); detour_span = f("DETOUR_SPAN");
        update_distance = f("UPDATE_DISTANCE"); recovery_heading_error = f("RECOVERY_HEADING_ERROR");
        off_track_grace = f("OFF_TRACK_GRACE"); stall_distance = f("STALL_DISTANCE"); stall_time = f("STALL_TIME");
        ema_gamma = f("EMA_GAMMA"); reward_drop = f("REWARD_DROP");
        w_noise = f("W_NOISE"); w_corr = f("W_CORR"); w_tol = f("W_TOL"); w_margin = f("W_MARGIN");
        w_min_margin = f("W_MIN_MARGIN");
        auto we = fl("W_ERROR"); for (int k = 0; k < 5; ++k) w_error[k] = we.at(k);
        w_error_cost = f("W_ERROR_COST"); line_reach = f("LINE_REACH"); fill_min_speed = f("FILL_MIN_SPEED");
        w_max_slip = f("W_MAX_SLIP"); fragile_share = f("FRAGILE_SHARE"); fragile_memory = f("FRAGILE_MEMORY");
        fragile_gain = f("FRAGILE_GAIN");
        rec.window = f("REC_WINDOW"); rec.distance = f("REC_DISTANCE"); rec.heading = f("REC_HEADING");
        rec.steer_in = f("REC_STEER_IN"); rec.steer_in_max = f("REC_STEER_IN_MAX"); rec.speed_min = f("REC_SPEED_MIN");
        rec.gate = f("REC_GATE"); rec.noise_scale = f("REC_NOISE_SCALE"); rec.speed_min_w = f("REC_SPEED_MIN_W");
        rec.speed_max = f("REC_SPEED_MAX"); rec.speed_max_w = f("REC_SPEED_MAX_W"); rec.reverse = f("REC_REVERSE");
        rec.slip = f("REC_SLIP"); rec.yaw_rate = f("REC_YAW_RATE");
        fallback_steer_scales = fl("FALLBACK_STEER_SCALES"); shift_steps = fl("SHIFT_STEPS");
        w_steer_scales = fl("W_STEER_SCALES");
        seed = (uint64_t)c.num("SEED");
    }
};

// ---------------------------------------------------------------------------------------------
// Random numbers: counter based, so the draws do not depend on the thread count.
// ---------------------------------------------------------------------------------------------
inline uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct Rng {
    uint64_t s[4];
    Rng(uint64_t seed, uint64_t stream, uint64_t block) {
        uint64_t x = seed ^ (stream * 0xD1B54A32D192ED03ull) ^ (block * 0x8CB92BA72F3D8DD7ull);
        for (auto& v : s) v = splitmix(x);
    }
    uint64_t next() {  // xoshiro256**
        const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    double uniform() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); }  // (0, 1)
    void normal2(float* a, float* b) {  // Box-Muller
        const double r = std::sqrt(-2.0 * std::log(uniform())), t = 6.283185307179586 * uniform();
        *a = (float)(r * std::cos(t));
        *b = (float)(r * std::sin(t));
    }
};

using Clock = std::chrono::steady_clock;
inline double since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// ---------------------------------------------------------------------------------------------
// Planner
// ---------------------------------------------------------------------------------------------
struct Snapshot {
    std::vector<float> waypoints, actions, pending;
    std::vector<uint8_t> known, vss;
};

struct HistEntry {
    long step;
    std::array<float, NC> row;
    std::vector<float> actions, pending;
    bool known;
};

struct Window {
    int w;
    std::vector<float> start;    // (D,)
    std::vector<float> pending;  // (delay, 2)
    std::vector<float> samples;  // (Kw, H, 2)
    int count;
};

enum Section { S_RECOVERY, S_SAMPLES, S_ROLLOUT, S_COSTS, S_PLAN, S_WINDOW, S_TABLE, S_TOTAL, S_COUNT };

class Planner {
public:
    Planner(const Config& c, const CarModel& model, int n, const float* table, const float* anchor,
            const float* tangent, const float* ortho, const float* track_length, const int32_t* track_label,
            const float* track_direction, int grid_h, int grid_w)
        : S(c), model_(model), n_(n) {
        waypoints_.assign(table, table + (size_t)n * NC);
        anchor_.assign(anchor, anchor + 2 * n);
        tangent_.assign(tangent, tangent + 2 * n);
        ortho_.assign(ortho, ortho + 2 * n);
        if (S.track_field) {
            const size_t cells = (size_t)grid_h * grid_w;
            track_length_.assign(track_length, track_length + cells);
            track_label_.assign(track_label, track_label + cells);
            track_direction_.assign(track_direction, track_direction + 2 * cells);
        }
        wp_actions_.assign((size_t)n * S.H * 2, 0.f);
        wp_known_.assign(n, 0);
        wp_pending_.assign((size_t)n * S.delay * 2, 0.f);
        wp_hist_.assign(n, {});
        wp_fragile_at_.assign(n, -1);
        wp_off_track_count_.assign(n, 0.f);
        wp_vss_.assign(n, 0);
        wp_off_track_marked_.assign(n, 0);
        stats_.fill(0.0);
        reset();
    }

    // ------------------------------------------------------------------ lifecycle
    void reset() {
        control_index_ = 0;
        episode_timesteps_ = 0;
        action_buffer_.assign((size_t)S.H * 2, 0.f);
        mean_state_seed_.clear();
        fill_previous_state_.reset();
        driven_valid_ = false;
        reset_stall();
    }

    void reset_episode_state() {
        ++episode_id_;
        issued_.clear();
        control_index_ = 0;
        episode_timesteps_ = 0;
        action_buffer_.assign((size_t)S.H * 2, 0.f);
        reset_stall();
    }

    void episode_done(bool crash) {
        if (crash) ++grid_crashes_;
        revert_nearby_waypoints();
        reset_episode_state();
    }

    // One control step: select_action, then the command goes into the control-delay queue.
    // out: [angular control, translational control, network steer, network translational].
    void step(const float* state, float* out) {
        const auto t0 = Clock::now();
        float action[2];
        select_action(state, action);
        const float steering = action[0] * S.denorm[0];
        const float accel = std::min(std::max(action[1] * S.denorm[1], -S.cap), S.cap);
        out[0] = steering;
        out[1] = accel;
        out[2] = action[0];
        out[3] = action[1];
        issued_.push_back({action[0], action[1]});
        while ((int)issued_.size() > S.delay) issued_.pop_front();
        ++control_index_;
        ++total_timesteps_;
        ++episode_timesteps_;
        stats_[S_TOTAL] += since(t0);
        ++stat_steps_;
    }

    void lap(float lap_time) {
        ++completed_laps_;
        const long previous = lap_end_step_;
        lap_end_step_ = total_timesteps_;
        if (previous >= 0) last_lap_steps_ = lap_end_step_ - previous;
        update_permanent_freeze(lap_time);
        update_best_lap_table(lap_time);
        int known = 0;
        double speed = 0.0;
        for (int i = 0; i < n_; ++i)
            if (waypoint_known(i)) {
                ++known;
                speed += waypoints_[(size_t)i * NC + WSPEED];
            }
        log("[HirarchicalPlanner/cpp] Lap: %d/%d window updates accepted, %d/%d waypoints stored, mean stored speed %.2f m/s",
            window_accepted_, window_tried_, known, n_, known ? speed / known : 0.0);
        window_accepted_ = window_tried_ = 0;
    }

    // ------------------------------------------------------------------ views
    float* table() { return waypoints_.data(); }
    float* actions_table() { return wp_actions_.data(); }
    uint8_t* known_table() { return wp_known_.data(); }
    float* off_track_counts() { return wp_off_track_count_.data(); }
    bool frozen() const { return permanently_frozen_; }
    int crashes() const { return grid_crashes_; }
    void stats(double* out) const {
        for (int i = 0; i < S_COUNT; ++i) out[i] = stat_steps_ ? stats_[i] / stat_steps_ : 0.0;
        out[S_COUNT] = (double)stat_steps_;
    }
    void clear_stats() {
        stats_.fill(0.0);
        stat_steps_ = 0;
    }
    bool recover_mode() const { return recover_mode_; }

private:
    const Settings S;
    CarModel model_;
    const int n_;

    // Waypoint table and its bookkeeping.
    std::vector<float> waypoints_, anchor_, tangent_, ortho_, wp_actions_, wp_pending_, wp_off_track_count_;
    std::vector<uint8_t> wp_known_, wp_vss_, wp_off_track_marked_;
    std::vector<long> wp_fragile_at_;
    std::vector<std::vector<HistEntry>> wp_hist_;
    std::vector<float> track_length_, track_direction_;
    std::vector<int32_t> track_label_;

    // Control state.
    long control_index_ = 0, total_timesteps_ = 0, episode_timesteps_ = 0, episode_id_ = 0;
    std::deque<std::array<float, 2>> issued_;
    std::vector<float> action_buffer_, mean_state_seed_;
    std::optional<std::array<float, D>> fill_previous_state_;
    bool off_track_ = false, recover_mode_ = false, reward_drop_warned_ = false;
    double reward_ema_ = 1.0;
    int car_waypoint_ = -1;

    // Stall.
    bool seen_on_track_ = false, was_off_track_ = true, stalled_ = false;
    int stall_phase_ = 0, reverse_left_ = 0, backtrack_wp_ = -1;
    float stall_anchor_[2] = {0, 0};
    std::deque<std::array<float, 2>> pose_history_;

    // Driven commands and line crossings (_record_driven).
    bool driven_valid_ = false;
    long driven_step_ = -1, driven_first_ = 0;
    std::deque<std::array<float, 2>> driven_actions_;
    std::optional<std::array<float, D>> driven_previous_;
    std::vector<float> driven_cross_;

    // Laps, freeze, best-lap table.
    int completed_laps_ = 0, window_accepted_ = 0, window_tried_ = 0, grid_crashes_ = 0;
    long lap_end_step_ = -1, last_lap_steps_ = -1;
    bool permanently_frozen_ = false;
    std::optional<Snapshot> freeze_lap_start_, freeze_best_table_, lap_start_table_, best_lap_table_, driven_table_;
    std::optional<float> freeze_best_lap_, best_lap_time_;
    int freeze_laps_since_best_ = 0, laps_since_best_ = 0, driven_last_ = -1;

    // Work buffers.
    std::vector<float> actions_, traj_, s0_, pending_, eps_, total_, rewards_, crash_, weights_;
    std::vector<float> win_traj_;
    uint64_t draw_counter_ = 0;
    std::array<double, S_COUNT> stats_;
    long stat_steps_ = 0;

    template <typename... A>
    void log(const char* fmt, A... args) const {
        std::printf(fmt, args...);
        std::printf("\n");
        std::fflush(stdout);
    }

    // ------------------------------------------------------------------ table helpers
    float* row(int i) { return &waypoints_[(size_t)i * NC]; }
    const float* row(int i) const { return &waypoints_[(size_t)i * NC]; }
    float wx(int i) const { return waypoints_[(size_t)i * NC + WX]; }
    float wy(int i) const { return waypoints_[(size_t)i * NC + WY]; }
    int mod(long i) const { return (int)(((i % n_) + n_) % n_); }
    bool waypoint_known(int i) const { return wp_known_[i] && wp_vss_[i]; }

    Snapshot snapshot() const { return {waypoints_, wp_actions_, wp_pending_, wp_known_, wp_vss_}; }

    // Copied into the live arrays, which Python views without copying.
    void restore_snapshot(const Snapshot& s) {
        std::copy(s.waypoints.begin(), s.waypoints.end(), waypoints_.begin());
        std::copy(s.actions.begin(), s.actions.end(), wp_actions_.begin());
        std::copy(s.known.begin(), s.known.end(), wp_known_.begin());
        std::copy(s.vss.begin(), s.vss.end(), wp_vss_.begin());
        std::copy(s.pending.begin(), s.pending.end(), wp_pending_.begin());
        for (auto& h : wp_hist_) h.clear();
    }

    std::vector<float> issued_actions() const {  // (delay, 2), oldest first, zeros before the first command
        std::vector<float> p((size_t)S.delay * 2, 0.f);
        const int k = (int)issued_.size();
        for (int i = 0; i < k; ++i) {
            p[(size_t)(S.delay - k + i) * 2] = issued_[i][0];
            p[(size_t)(S.delay - k + i) * 2 + 1] = issued_[i][1];
        }
        return p;
    }

    static std::vector<float> fit_horizon(const std::vector<float>& seq, int horizon) {
        const int n = (int)seq.size() / 2;
        std::vector<float> out((size_t)horizon * 2);
        for (int t = 0; t < horizon; ++t) {
            const int s = std::min(t, n - 1);
            out[2 * t] = seq[2 * s];
            out[2 * t + 1] = seq[2 * s + 1];
        }
        return out;
    }

    // ------------------------------------------------------------------ track field
    void track_field_at(float x, float y, float* length, int* label, float* dir) const {
        int r, c;
        bool inside;
        map_cell(x, y, &r, &c, &inside);
        const size_t cell = (size_t)r * map_width() + c;
        *length = track_length_[cell];
        *label = track_label_[cell];
        if (dir) {
            dir[0] = track_direction_[2 * cell];
            dir[1] = track_direction_[2 * cell + 1];
        }
    }

    // Which waypoints a position may be matched to; true for all without a track field.
    void matchable(float x, float y, std::vector<uint8_t>& allowed) const {
        allowed.assign(n_, 1);
        if (!S.track_field) return;
        float length;
        int label;
        track_field_at(x, y, &length, &label, nullptr);
        if (label < 0) return;
        for (int i = 0; i < n_; ++i) {
            const int gap = mod(i - label);
            allowed[i] = std::min(gap, n_ - gap) <= S.track_window;
        }
    }

    float detour(float x, float y) const {
        if (!S.track_field) return 0.f;
        float path;
        int label;
        track_field_at(x, y, &path, &label, nullptr);
        if (label < 0) return 0.f;
        const int l = std::max(label, 0);
        return path - std::hypot(x - anchor_[2 * l], y - anchor_[2 * l + 1]);
    }

    // Segment of the raceline closest to (x, y): start index and progress along it.
    void segment(float x, float y, const std::vector<uint8_t>* allowed, int* start, float* progress) const {
        float best = INF;
        int bi = 0;
        float bp = 0.f;
        for (int i = 0; i < n_; ++i) {
            const int j = (i + 1) % n_;
            const float sx = wx(j) - wx(i), sy = wy(j) - wy(i);
            const float l2 = std::max(sx * sx + sy * sy, 1e-12f);
            float p = ((x - wx(i)) * sx + (y - wy(i)) * sy) / l2;
            p = std::min(std::max(p, 0.f), 1.f);
            const float px = wx(i) + p * sx - x, py = wy(i) + p * sy - y;
            float d2 = px * px + py * py;
            if (allowed && !(*allowed)[i]) d2 = INF;
            if (d2 < best) {
                best = d2;
                bi = i;
                bp = p;
            }
        }
        *start = bi;
        *progress = bp;
    }

    int nearest_waypoint_index(const float* state) const {
        std::vector<uint8_t> allowed;
        matchable(state[X], state[Y], allowed);
        int idx;
        float along;
        segment(state[X], state[Y], &allowed, &idx, &along);
        long ahead = along <= 0.f ? idx : (along >= 1.f ? idx + 2 : idx + 1);
        return mod(ahead + S.ignore_steps);
    }

    float raceline_distance(const float* state) const {
        std::vector<uint8_t> allowed;
        matchable(state[X], state[Y], allowed);
        int start;
        float p;
        segment(state[X], state[Y], &allowed, &start, &p);
        const int end = (start + 1) % n_;
        const float px = wx(start) + p * (wx(end) - wx(start)), py = wy(start) + p * (wy(end) - wy(start));
        const float distance = std::hypot(state[X] - px, state[Y] - py);
        if (detour(state[X], state[Y]) > S.detour_start) {
            float path;
            int label;
            track_field_at(state[X], state[Y], &path, &label, nullptr);
            return std::max(distance, path);
        }
        return distance;
    }

    // ------------------------------------------------------------------ recovery and stall
    void reset_stall() {
        seen_on_track_ = false;
        was_off_track_ = true;
        stall_phase_ = 0;
        reverse_left_ = 0;
        backtrack_wp_ = -1;
        stalled_ = false;
        pose_history_.clear();
    }

    void update_recover_mode(bool off) { recover_mode_ = off || reward_drop_warned_ || stall_phase_ == 1; }

    bool off_track_affects_waypoints() const {
        if (S.off_track_grace <= 0.f) return true;
        return (float)episode_timesteps_ * S.dt >= S.off_track_grace;
    }

    void start_backtrack(float x, float y, int car_wp, const char* reason) {
        pose_history_.clear();
        if (S.stall_reverse_steps > 0) {
            stall_phase_ = 1;
            reverse_left_ = S.stall_reverse_steps;
            backtrack_wp_ = mod(car_wp - S.stall_backtrack_wps);
            log("[HirarchicalPlanner/cpp] Warning: %s. Driving to the waypoint %d behind it.", reason,
                S.stall_backtrack_wps);
        } else {
            stall_phase_ = 2;
            stall_anchor_[0] = x;
            stall_anchor_[1] = y;
        }
        stalled_ = stall_phase_ == 2;
    }

    void update_stall(float x, float y, int car_wp) {
        if (stall_phase_ == 1) {
            const bool arrived = std::hypot(x - wx(backtrack_wp_), y - wy(backtrack_wp_)) < 0.5f;
            if (reverse_left_ <= 0 || arrived) {
                stall_phase_ = 2;
                stall_anchor_[0] = x;
                stall_anchor_[1] = y;
            }
        } else if (stall_phase_ == 2) {
            if (std::hypot(x - stall_anchor_[0], y - stall_anchor_[1]) >= S.stall_distance) {
                stall_phase_ = 0;
                pose_history_.clear();
            }
        } else {
            const int steps = std::max(1, (int)std::lround(S.stall_time / S.dt));
            pose_history_.push_back({x, y});
            while ((int)pose_history_.size() > steps + 1) pose_history_.pop_front();
            if ((int)pose_history_.size() > steps) {
                const auto& a = pose_history_.front();
                const auto& b = pose_history_.back();
                if (std::hypot(b[0] - a[0], b[1] - a[1]) < S.stall_distance)
                    start_backtrack(x, y, car_wp, "car barely moved");
            }
        }
        stalled_ = stall_phase_ == 2;
    }

    void revert_window_updates() {
        if (last_lap_steps_ < 0) return;
        const long since_step = total_timesteps_ - last_lap_steps_;
        for (int off = -S.revert_behind; off <= S.revert_ahead; ++off) {
            const int idx = mod(car_waypoint_ + off);
            auto& h = wp_hist_[idx];
            std::optional<HistEntry> entry;
            while (!h.empty() && h.back().step > since_step) {
                entry = std::move(h.back());
                h.pop_back();
            }
            if (!entry) continue;
            std::copy(entry->row.begin(), entry->row.end(), row(idx));
            std::copy(entry->actions.begin(), entry->actions.end(), &wp_actions_[(size_t)idx * S.H * 2]);
            wp_known_[idx] = entry->known;
            std::copy(entry->pending.begin(), entry->pending.end(), &wp_pending_[(size_t)idx * S.delay * 2]);
        }
    }

    void revert_nearby_waypoints() {
        if (car_waypoint_ < 0 || permanently_frozen_ || !seen_on_track_) return;
        revert_window_updates();
    }

    void note_off_track_pass(bool reward_drop) {
        if (car_waypoint_ < 0 || !seen_on_track_) return;
        for (int i = 0; i < n_; ++i) {
            const int gap = mod(i - car_waypoint_);
            if (std::min(gap, n_ - gap) > 10) wp_off_track_marked_[i] = 0;
        }
        const bool off = off_track_ && off_track_affects_waypoints();
        if ((off || reward_drop) && !wp_off_track_marked_[car_waypoint_]) {
            wp_off_track_marked_[car_waypoint_] = 1;
            wp_off_track_count_[car_waypoint_] += 1.f;
        }
    }

    void capture_driven_table() {
        if (car_waypoint_ < 0) return;
        if (!driven_table_) {
            driven_table_ = snapshot();
            driven_last_ = car_waypoint_;
            return;
        }
        const int ahead = mod(car_waypoint_ - driven_last_);
        if (ahead == 0) return;
        std::vector<int> idx;
        if (ahead <= n_ / 4)
            for (int k = 0; k < ahead; ++k) idx.push_back(mod(driven_last_ + 1 + k));
        else
            idx.push_back(car_waypoint_);
        driven_last_ = car_waypoint_;
        Snapshot& t = *driven_table_;
        const size_t ha = (size_t)S.H * 2, pa = (size_t)S.delay * 2;
        for (int i : idx) {
            std::copy(row(i), row(i) + NC, &t.waypoints[(size_t)i * NC]);
            std::copy(&wp_actions_[i * ha], &wp_actions_[i * ha] + ha, &t.actions[i * ha]);
            std::copy(&wp_pending_[i * pa], &wp_pending_[i * pa] + pa, &t.pending[i * pa]);
            t.known[i] = wp_known_[i];
            t.vss[i] = wp_vss_[i];
        }
    }

    void refresh_recovery(const float* state) {
        const float x = state[X], y = state[Y];
        const float line_distance = raceline_distance(state);
        std::vector<uint8_t> allowed;
        matchable(x, y, allowed);
        float best = INF;
        int bi = 0;
        for (int i = 0; i < n_; ++i) {
            const float dx = wx(i) - x, dy = wy(i) - y;
            const float d = allowed[i] ? std::sqrt(dx * dx + dy * dy) : INF;
            if (d < best) {
                best = d;
                bi = i;
            }
        }
        car_waypoint_ = bi;
        capture_driven_table();
        bool on_track = line_distance <= S.update_distance;
        if (on_track && S.recovery_heading_error > 0.f) {
            const double e = (double)state[TH] - (double)row(car_waypoint_)[WHEAD];
            on_track = std::fabs(std::atan2(std::sin(e), std::cos(e))) <= S.recovery_heading_error;
        }
        if (on_track) seen_on_track_ = true;
        update_stall(x, y, car_waypoint_);
        off_track_ = !on_track || stalled_;
        const bool affects = off_track_ && off_track_affects_waypoints();
        if (affects && !was_off_track_) revert_nearby_waypoints();
        was_off_track_ = affects;
        update_recover_mode(off_track_);
        note_off_track_pass(false);
    }

    // ------------------------------------------------------------------ sampling
    std::optional<std::vector<float>> interpolated_waypoint_actions(const float* state) const {
        bool any = false;
        for (int i = 0; i < n_ && !any; ++i) any = wp_known_[i];
        if (!any) return std::nullopt;
        std::vector<uint8_t> allowed;
        matchable(state[X], state[Y], allowed);
        int start;
        float alpha;
        segment(state[X], state[Y], &allowed, &start, &alpha);
        const int end = (start + 1) % n_;
        const bool hs = wp_known_[start], he = wp_known_[end];
        if (!hs && !he) return std::nullopt;
        const size_t ha = (size_t)S.H * 2;
        const float* a = &wp_actions_[start * ha];
        const float* b = &wp_actions_[end * ha];
        if (hs != he) return std::vector<float>(hs ? a : b, (hs ? a : b) + ha);
        std::vector<float> out(ha);
        for (size_t k = 0; k < ha; ++k) out[k] = (1.f - alpha) * a[k] + alpha * b[k];
        return out;
    }

    // Correlated samples around nominal (H, 2) into out (count, H, 2) (_sample_mpc_actions).
    void sample_actions(const std::vector<float>& nominal, int count, float noise_scale, float correlation, float* out) {
        const int H = (int)nominal.size() / 2;
        const float rho = std::min(std::max(correlation, 0.f), 0.999f);
        float stdv[2] = {noise_scale * S.noise[0] / S.denorm[0], noise_scale * S.noise[1] / S.denorm[1]};
        if (off_track_ || stall_phase_ == 1) {
            stdv[0] *= S.rec.noise_scale;
            stdv[1] *= S.rec.noise_scale;
        }
        const float fresh = std::sqrt(1.f - rho * rho);
        const uint64_t stream = ++draw_counter_;
        const int per = 64;
        hw::pool().run((count + per - 1) / per, [&](int b) {
            Rng rng(S.seed, stream, (uint64_t)b);
            const int hi = std::min(count, (b + 1) * per);
            std::vector<float> e((size_t)H * 2 + 1);
            for (int k = b * per; k < hi; ++k) {
                for (int i = 0; i < H * 2; i += 2) rng.normal2(&e[i], &e[i + 1]);
                float* o = out + (size_t)k * H * 2;
                for (int i = 0; i < 2; ++i) o[i] = nominal[i] + e[i] * stdv[i];
                for (int t = 1; t < H; ++t)
                    for (int i = 0; i < 2; ++i)
                        o[2 * t + i] = rho > 0.f ? o[2 * (t - 1) + i] * rho + (1.f - rho) * nominal[2 * t + i] +
                                                       fresh * stdv[i] * e[2 * t + i]
                                                 : nominal[2 * t + i] + e[2 * t + i] * stdv[i];
                for (int t = 0; t < H; ++t) {
                    o[2 * t] = std::min(std::max(o[2 * t], -1.f), 1.f);
                    o[2 * t + 1] = std::min(std::max(o[2 * t + 1], S.brake_limit), S.accel_limit);
                }
            }
        });
    }

    void with_fallback_samples(int count, const std::vector<float>& nominal, float speed) {
        const int H = (int)nominal.size() / 2;
        action_buffer_ = fit_horizon(action_buffer_, H);
        std::vector<std::vector<float>> fixed{action_buffer_};
        for (float scale : S.fallback_steer_scales) {
            std::vector<float> s(nominal);
            for (int t = 0; t < H; ++t) s[2 * t] = nominal[2 * t] * scale;
            fixed.push_back(s);
        }
        if (speed < S.creep_speed) {
            const int steps = (int)std::ceil((S.creep_speed - speed) / (S.creep_accel * S.dt));
            std::vector<float> throttle(H, 0.f);
            for (int t = 0; t < std::min(steps, H); ++t) throttle[t] = S.creep_accel / S.denorm[1];
            for (float steer : {-1.f, -0.5f, 0.5f, 1.f}) {
                std::vector<float> s((size_t)H * 2);
                for (int t = 0; t < H; ++t) {
                    s[2 * t] = steer;
                    s[2 * t + 1] = throttle[t];
                }
                fixed.push_back(s);
            }
            const float braking = S.creep_accel * S.brake_multiplier;
            const int rsteps = (int)std::ceil((std::max(speed, 0.f) + S.back_off_speed) / (braking * S.dt));
            std::vector<float> reverse(H, 0.f);
            for (int t = 0; t < std::min(rsteps, H); ++t) reverse[t] = -S.creep_accel / S.denorm[1];
            for (float steer : {-1.f, 1.f}) {
                std::vector<float> s((size_t)H * 2);
                for (int t = 0; t < H; ++t) {
                    s[2 * t] = steer;
                    s[2 * t + 1] = reverse[t];
                }
                fixed.push_back(s);
            }
        }
        const int f = (int)fixed.size();
        if (f >= count) return;
        for (int i = 0; i < f; ++i) std::copy(fixed[i].begin(), fixed[i].end(), &actions_[(size_t)(count - f + i) * H * 2]);
    }

    // ------------------------------------------------------------------ waypoint states and lines
    void waypoint_state(int i, float* s) const {
        std::fill(s, s + D, 0.f);
        const float* w = row(i);
        s[X] = w[WX];
        s[Y] = w[WY];
        s[TH] = w[WHEAD];
        s[SIN] = std::sin(w[WHEAD]);
        s[COS] = std::cos(w[WHEAD]);
        s[VX] = w[WSPEED];
        s[VY] = w[WDRIFT];
        s[STEER] = w[WSTEER];
        s[PD] = w[WYAW];
        s[SLIP] = w[WSLIP];
    }

    void write_waypoint_state(int i, const float* s) {
        const float ax = anchor_[2 * i], ay = anchor_[2 * i + 1], ox = ortho_[2 * i], oy = ortho_[2 * i + 1];
        const float along = (s[X] - ax) * ox + (s[Y] - ay) * oy;
        float* w = row(i);
        w[WX] = ax + along * ox;
        w[WY] = ay + along * oy;
        w[WHEAD] = s[TH];
        w[WSPEED] = std::max(s[VX], 0.f);
        w[WDRIFT] = s[VY];
        w[WSTEER] = s[STEER];
        w[WYAW] = s[PD];
        w[WSLIP] = s[SLIP];
        wp_vss_[i] = 1;
    }

    // First crossing of waypoint idx's line by the path states (T, D): time in steps (NaN if none) and the
    // interpolated state (_line_crossings for one rollout and one waypoint).
    float line_crossing(const float* states, int T, int idx, float* at) const {
        const float ax = anchor_[2 * idx], ay = anchor_[2 * idx + 1];
        const float tx = tangent_[2 * idx], ty = tangent_[2 * idx + 1];
        const float ox = ortho_[2 * idx], oy = ortho_[2 * idx + 1];
        const float stored = (wx(idx) - ax) * ox + (wy(idx) - ay) * oy;
        int k = -1;
        float a0 = 0, a1 = 0;
        float prev_along = 0;
        bool prev_near = false;
        for (int t = 0; t < T; ++t) {
            const float rx = states[t * D + X] - ax, ry = states[t * D + Y] - ay;
            const float along = rx * tx + ry * ty;
            const bool near = std::fabs((rx * ox + ry * oy) - stored) < S.line_reach;
            if (t > 0 && prev_along < 0.f && along >= 0.f && prev_near && near) {
                k = t - 1;
                a0 = prev_along;
                a1 = along;
                break;
            }
            prev_along = along;
            prev_near = near;
        }
        if (k < 0) {
            if (at) std::copy(states, states + D, at);  // k = 0 like argmax of no crossing; not used
            return NaN;
        }
        const float frac = std::min(std::max(-a0 / std::max(a1 - a0, 1e-9f), 0.f), 1.f);
        if (at) {
            for (int i = 0; i < D; ++i) at[i] = states[k * D + i] + frac * (states[(k + 1) * D + i] - states[k * D + i]);
            at[TH] = std::atan2(at[SIN], at[COS]);
        }
        return (float)k + frac;
    }

    static void timeline_at(const std::vector<float>& values, float start, const float* times, int q, float* out) {
        const int last = (int)values.size() / 2 - 1;
        for (int i = 0; i < q; ++i) {
            const float pos = std::min(std::max(times[i] - start, 0.f), (float)last);
            const int lo = std::min((int)std::floor(pos), last);
            const int hi = std::min(lo + 1, last);
            const float frac = pos - (float)lo;
            for (int c = 0; c < 2; ++c) out[2 * i + c] = values[2 * lo + c] * (1.f - frac) + values[2 * hi + c] * frac;
        }
    }

    // ------------------------------------------------------------------ window update
    std::optional<std::vector<float>> driven_candidate(int w, int horizon) const {
        if (!driven_valid_) return std::nullopt;
        const float crossed = driven_cross_[w];
        if (std::isnan(crossed) || crossed < (float)driven_first_ || crossed + horizon - 1 > (float)driven_step_)
            return std::nullopt;
        std::vector<float> timeline;
        for (const auto& a : driven_actions_) {
            timeline.push_back(a[0]);
            timeline.push_back(a[1]);
        }
        std::vector<float> times(horizon), out((size_t)horizon * 2);
        for (int i = 0; i < horizon; ++i) times[i] = crossed + (float)i;
        timeline_at(timeline, (float)driven_first_, times.data(), horizon, out.data());
        return out;
    }

    std::vector<std::vector<float>> shifted_sequences(const float* stored) const {
        std::vector<std::vector<float>> out;
        const int H = S.H;
        for (int channel : {1, 0})
            for (float shift : S.shift_steps) {
                std::vector<float> seq(stored, stored + H * 2);
                for (int t = 0; t < H; ++t) seq[2 * t + channel] = stored[2 * std::min(std::max(t + (int)shift, 0), H - 1) + channel];
                out.push_back(seq);
            }
        for (float scale : S.w_steer_scales) {
            std::vector<float> seq(stored, stored + H * 2);
            for (int t = 0; t < H; ++t) seq[2 * t] = std::min(std::max(stored[2 * t] * scale, -1.f), 1.f);
            out.push_back(seq);
        }
        return out;
    }

    std::optional<Window> window_job() {
        if (recover_mode_ || car_waypoint_ < 0 || permanently_frozen_) return std::nullopt;
        const int w = mod(car_waypoint_ - S.lag);
        for (int k = 0; k < S.span + 2; ++k)
            if (!waypoint_known(mod(w + k))) return std::nullopt;
        const size_t ha = (size_t)S.H * 2;
        std::vector<float> stored(&wp_actions_[w * ha], &wp_actions_[w * ha] + ha);
        Window job;
        job.w = w;
        job.count = S.w_samples;
        job.samples.resize((size_t)job.count * ha);
        sample_actions(stored, job.count, S.w_noise, S.w_corr, job.samples.data());
        std::copy(stored.begin(), stored.end(), job.samples.begin());
        if (job.count > 1) {
            auto driven = driven_candidate(w, S.H);
            if (driven) std::copy(driven->begin(), driven->end(), job.samples.begin() + ha);
        }
        auto shifted = shifted_sequences(stored.data());
        if (!shifted.empty() && job.count > 2 + (int)shifted.size())
            for (size_t i = 0; i < shifted.size(); ++i) std::copy(shifted[i].begin(), shifted[i].end(), job.samples.begin() + (2 + i) * ha);
        job.start.resize(D);
        waypoint_state(w, job.start.data());
        job.pending.assign(&wp_pending_[(size_t)w * S.delay * 2], &wp_pending_[(size_t)w * S.delay * 2] + S.delay * 2);
        return job;
    }

    bool fragile(const std::vector<int>& idx) const {
        if (last_lap_steps_ < 0) return false;
        for (int i : idx) {
            const long m = wp_fragile_at_[i];
            if (m >= 0 && (float)(total_timesteps_ - m) <= S.fragile_memory * (float)last_lap_steps_) return true;
        }
        return false;
    }

    float window_error(const float* at, int i) const {
        const float* w = row(i);
        const float lateral = std::fabs((at[X] - w[WX]) * ortho_[2 * i] + (at[Y] - w[WY]) * ortho_[2 * i + 1]);
        const float heading = std::fabs(wrap(at[TH] - w[WHEAD]));
        return S.w_error[0] * lateral + S.w_error[1] * heading + S.w_error[2] * std::fabs(at[VX] - w[WSPEED]) +
               S.w_error[3] * std::fabs(at[VY] - w[WDRIFT]) + S.w_error[4] * std::fabs(at[PD] - w[WYAW]);
    }

    void window_update(const Window& win, const float* states, int T) {
        const int span = S.span, down = S.downstream, K = win.count, H = S.H;
        std::vector<int> idx(span + down);
        for (int j = 0; j < span + down; ++j) idx[j] = mod(win.w + 1 + j);
        // The waypoints sample 0 (the stored sequence) crosses in a row.
        int count = 0;
        for (int j = 0; j < span + down; ++j) {
            if (std::isnan(line_crossing(states, T, idx[j], nullptr)) || !waypoint_known(idx[j])) break;
            ++count;
        }
        if (count < span + 1) return;
        idx.resize(count);
        ++window_tried_;
        std::vector<float> times((size_t)K * count), at((size_t)K * count * D);
        std::vector<float> error(K), arrival(K), clearance(K);
        std::vector<uint8_t> valid(K);
        hw::pool().run(K, [&](int k) {
            const float* s = states + (size_t)k * T * D;
            bool crossed = true;
            for (int j = 0; j < count; ++j) {
                times[(size_t)k * count + j] = line_crossing(s, T, idx[j], &at[((size_t)k * count + j) * D]);
                crossed &= !std::isnan(times[(size_t)k * count + j]);
                if (j > 0) crossed &= times[(size_t)k * count + j] > times[(size_t)k * count + j - 1];
            }
            float e = -INF;
            bool any_nan = false;
            for (int j = span; j < count; ++j) {
                const float v = window_error(&at[((size_t)k * count + j) * D], idx[j]);
                any_nan |= std::isnan(v);
                e = std::max(e, v);
            }
            error[k] = any_nan ? NaN : e;
            arrival[k] = times[(size_t)k * count + count - 1];
            const float until = std::ceil(std::isnan(arrival[k]) ? (float)T : arrival[k]) + 1.f;
            float clear = INF, slip = 0.f;
            for (int t = 0; t < T; ++t) {
                if ((float)t > until) break;
                bool crashed;
                clear = std::min(clear, corner_clearance(s + t * D, -INF, &crashed));
                slip = std::max(slip, std::atan2(std::fabs(s[t * D + VY]), std::max(s[t * D + VX], S.creep_speed)));
            }
            clearance[k] = clear;
            valid[k] = crossed && slip <= S.w_max_slip;
        });
        float margin = std::min(S.w_margin, clearance[0]);
        margin = std::max(margin, S.w_min_margin);
        for (int k = 0; k < K; ++k) valid[k] = valid[k] && clearance[k] >= margin;
        const float stored_arrival = valid[0] ? arrival[0] : INF;
        const bool repair = error[0] > S.w_tol;
        const bool is_fragile = !repair && fragile(idx);
        int best = -1;
        float best_score = INF;
        for (int k = 1; k < K; ++k) {
            bool eligible;
            float score;
            if (repair) {
                eligible = valid[k] && error[k] < error[0];
                score = error[k];
            } else if (is_fragile) {
                eligible = valid[k] && error[k] <= S.w_tol && clearance[k] >= clearance[0] + S.fragile_gain;
                score = arrival[k] + S.w_error_cost * error[k];
            } else {
                eligible = valid[k] && error[k] <= S.w_tol && arrival[k] < stored_arrival;
                score = arrival[k] + S.w_error_cost * error[k];
            }
            if (eligible && score < best_score) {
                best_score = score;
                best = k;
            }
        }
        if (best < 0) return;

        const int delay = S.delay;
        std::vector<float> sample_timeline(win.pending);
        sample_timeline.insert(sample_timeline.end(), &win.samples[(size_t)best * H * 2], &win.samples[(size_t)(best + 1) * H * 2]);
        const int last = idx[span];
        const float reached = times[(size_t)best * count + span];
        std::vector<float> stored_timeline(&wp_pending_[(size_t)last * delay * 2], &wp_pending_[(size_t)(last + 1) * delay * 2]);
        stored_timeline.insert(stored_timeline.end(), &wp_actions_[(size_t)last * H * 2], &wp_actions_[(size_t)(last + 1) * H * 2]);
        auto commands = [&](const std::vector<float>& at_times) {
            const int q = (int)at_times.size();
            std::vector<float> own((size_t)q * 2), downstream((size_t)q * 2), out((size_t)q * 2);
            timeline_at(sample_timeline, (float)-delay, at_times.data(), q, own.data());
            timeline_at(stored_timeline, reached - (float)delay, at_times.data(), q, downstream.data());
            for (int i = 0; i < q; ++i)
                for (int c = 0; c < 2; ++c) out[2 * i + c] = at_times[i] < reached ? own[2 * i + c] : downstream[2 * i + c];
            return out;
        };
        std::vector<float> steps(H), queue(delay);
        for (int i = 0; i < H; ++i) steps[i] = (float)i;
        for (int i = 0; i < delay; ++i) queue[i] = (float)(i - delay);
        struct Update {
            int k;
            std::vector<float> seq, queue;
            const float* state;
        };
        std::vector<Update> updates;
        updates.push_back({win.w, commands(steps), {}, nullptr});
        for (int j = 0; j < span; ++j) {
            const float t = times[(size_t)best * count + j];
            std::vector<float> ts(H), tq(delay);
            for (int i = 0; i < H; ++i) ts[i] = t + steps[i];
            for (int i = 0; i < delay; ++i) tq[i] = t + queue[i];
            updates.push_back({idx[j], commands(ts), commands(tq), &at[((size_t)best * count + j) * D]});
        }
        const size_t ha = (size_t)H * 2, pa = (size_t)delay * 2;
        for (auto& u : updates) {
            auto& h = wp_hist_[u.k];
            HistEntry e;
            e.step = total_timesteps_;
            std::copy(row(u.k), row(u.k) + NC, e.row.begin());
            e.actions.assign(&wp_actions_[u.k * ha], &wp_actions_[u.k * ha] + ha);
            e.known = wp_known_[u.k];
            e.pending.assign(&wp_pending_[u.k * pa], &wp_pending_[u.k * pa] + pa);
            h.push_back(std::move(e));
            if ((int)h.size() > S.w_history) h.erase(h.begin(), h.end() - S.w_history);
            std::copy(u.seq.begin(), u.seq.end(), &wp_actions_[u.k * ha]);
            if (u.state) {
                std::copy(u.queue.begin(), u.queue.end(), &wp_pending_[u.k * pa]);
                write_waypoint_state(u.k, u.state);
            }
        }
        ++window_accepted_;
    }

    // ------------------------------------------------------------------ driven commands and filling
    void record_driven(const float* state, const float* action) {
        if (!driven_valid_) {
            driven_valid_ = true;
            driven_step_ = -1;
            driven_first_ = 0;
            driven_actions_.clear();
            driven_previous_.reset();
            driven_cross_.assign(n_, NaN);
        }
        ++driven_step_;
        driven_actions_.push_back({action[0], action[1]});
        if ((int)driven_actions_.size() > S.driven_memory) {
            driven_actions_.pop_front();
            ++driven_first_;
        }
        std::array<float, D> current;
        std::copy(state, state + D, current.begin());
        auto previous = driven_previous_;
        driven_previous_ = current;
        if (!previous || car_waypoint_ < 0 || recover_mode_) return;
        float path[2 * D];
        std::copy(previous->begin(), previous->end(), path);
        std::copy(current.begin(), current.end(), path + D);
        for (int o = -3; o <= 3; ++o) {
            const int i = mod(car_waypoint_ + o);
            const float t = line_crossing(path, 2, i, nullptr);
            if (!std::isnan(t)) driven_cross_[i] = (float)(driven_step_ - 1) + t;
        }
    }

    void fill_passed_waypoints(const float* state, const std::vector<float>& action_seq) {
        std::array<float, D> current;
        std::copy(state, state + D, current.begin());
        auto previous = fill_previous_state_;
        fill_previous_state_ = current;
        if (!previous || car_waypoint_ < 0 || stall_phase_ == 1 || permanently_frozen_ || recover_mode_) return;
        if (current[VX] < S.fill_min_speed) return;
        float path[2 * D];
        std::copy(previous->begin(), previous->end(), path);
        std::copy(current.begin(), current.end(), path + D);
        const auto pending = issued_actions();
        const auto seq = fit_horizon(action_seq, S.H);
        for (int o = -3; o <= 3; ++o) {
            const int k = mod(car_waypoint_ + o);
            if (waypoint_known(k)) continue;
            float at[D];
            if (std::isnan(line_crossing(path, 2, k, at))) continue;
            write_waypoint_state(k, at);
            std::copy(seq.begin(), seq.end(), &wp_actions_[(size_t)k * S.H * 2]);
            std::copy(pending.begin(), pending.end(), &wp_pending_[(size_t)k * S.delay * 2]);
            wp_known_[k] = 1;
        }
    }

    std::vector<float> seed_from_states(const float* path, int horizon) const {
        std::vector<float> seed((size_t)horizon * 2);
        const float gain = S.seed_speed_gain / S.denorm[1];
        const size_t ha = (size_t)S.H * 2;
        for (int k = 0; k < horizon; ++k) {
            const float* s = path + (size_t)k * D;
            int start;
            float alpha;
            segment(s[X], s[Y], nullptr, &start, &alpha);
            const int end = (start + 1) % n_;
            const bool hs = wp_known_[start], he = wp_known_[end];
            if (!(hs || he)) {
                seed[2 * k] = action_buffer_[2 * k];
                seed[2 * k + 1] = action_buffer_[2 * k + 1];
                continue;
            }
            const float* fs = &wp_actions_[start * ha];
            const float* fe = &wp_actions_[end * ha];
            const float vs = row(start)[WSPEED] * S.target_speed_factor, ve = row(end)[WSPEED] * S.target_speed_factor;
            float m0, m1, target;
            if (hs && he) {
                m0 = (1.f - alpha) * fs[0] + alpha * fe[0];
                m1 = (1.f - alpha) * fs[1] + alpha * fe[1];
                target = (1.f - alpha) * vs + alpha * ve;
            } else if (hs) {
                m0 = fs[0];
                m1 = fs[1];
                target = vs;
            } else {
                m0 = fe[0];
                m1 = fe[1];
                target = ve;
            }
            seed[2 * k] = m0;
            seed[2 * k + 1] = m1 + gain * (target - s[VX]);
        }
        return seed;
    }

    // ------------------------------------------------------------------ costs
    int crash_end(int steps) const { return S.crash_horizon < 0 ? steps : std::min(steps, S.crash_horizon + 1); }

    int matched_waypoint_count(int closest, int steps) const {
        const float duration = (float)std::max(steps - 1, 0) * S.dt;
        float total = 0.f;
        int covered = 0;
        for (int k = 0; k + 1 < n_; ++k) {
            const int a = mod(closest + k), b = mod(closest + k + 1);
            const float len = std::hypot(wx(b) - wx(a), wy(b) - wy(a));
            total += len / std::max(row(a)[WSPEED], 0.5f);
            if (total <= S.match_share * duration) ++covered;
        }
        return std::min(std::max(covered + 1, 1), n_);
    }

    void recovery_cost(const float* states, int B, int T, int start, float* out) const {
        const int win = (int)S.rec.window + 3;
        std::vector<int> window(win);
        for (int i = 0; i < win; ++i) window[i] = mod(start - 3 + i);
        hw::pool().run((B + 15) / 16, [&](int b) {
            for (int k = b * 16; k < std::min(B, (b + 1) * 16); ++k) {
                const float* s = states + (size_t)k * T * D;
                double sums[7] = {0, 0, 0, 0, 0, 0, 0};
                float last_dist = 0, last_head = 0;
                for (int t = 0; t < T; ++t) {
                    const float* st = s + t * D;
                    float nd = INF;
                    int near = 0;
                    for (int i = 0; i < win; ++i) {
                        const float d = std::hypot(st[X] - wx(window[i]), st[Y] - wy(window[i]));
                        if (d < nd) {
                            nd = d;
                            near = window[i];
                        }
                    }
                    const float h = row(near)[WHEAD];
                    const float cross = (st[X] - wx(near)) * -std::sin(h) + (st[Y] - wy(near)) * std::cos(h);
                    float wanted = h - std::min(std::max(S.rec.steer_in * cross, -S.rec.steer_in_max), S.rec.steer_in_max);
                    if (S.track_field) {
                        float path, dir[2];
                        int label;
                        track_field_at(st[X], st[Y], &path, &label, dir);
                        const float det = std::min(std::max((detour(st[X], st[Y]) - S.detour_start) / S.detour_span, 0.f), 1.f);
                        const float turn = std::atan2(dir[1], dir[0]) - wanted;
                        wanted += det * wrap(turn);
                        if (det > 0.f) nd = std::max(nd, path);
                    }
                    const float he = std::fabs(wrap(st[TH] - wanted));
                    const float speed = st[VX];
                    const float pointing = he < S.rec.gate ? 1.f : 0.f;
                    sums[0] += nd;
                    sums[1] += he;
                    sums[2] += std::max(S.rec.speed_min - speed, 0.f) * pointing;
                    sums[3] += std::max(speed - S.rec.speed_max, 0.f);
                    sums[4] += std::max(-speed, 0.f);
                    sums[5] += std::max(std::fabs(st[VY]) - 1.f, 0.f);
                    sums[6] += std::max(std::fabs(st[PD]) - 3.f, 0.f);
                    last_dist = nd;
                    last_head = he;
                }
                out[k] = (float)(S.rec.distance * (sums[0] / T + last_dist) + S.rec.heading * (sums[1] / T + last_head) +
                                 S.rec.speed_min_w * sums[2] / T + S.rec.speed_max_w * sums[3] / T +
                                 S.rec.reverse * sums[4] / T + S.rec.slip * sums[5] / T + S.rec.yaw_rate * sums[6] / T);
            }
        });
    }

    void backtrack_cost(const float* states, int B, int T, float* out) const {
        const float tx = wx(backtrack_wp_), ty = wy(backtrack_wp_);
        for (int k = 0; k < B; ++k) {
            const float* s = states + (size_t)k * T * D;
            double dist = 0, slip = 0, yaw = 0, fast = 0;
            float last = 0;
            for (int t = 0; t < T; ++t) {
                const float* st = s + t * D;
                last = std::hypot(st[X] - tx, st[Y] - ty);
                dist += last;
                slip += std::max(std::fabs(st[VY]) - 1.f, 0.f);
                yaw += std::max(std::fabs(st[PD]) - 3.f, 0.f);
                fast += std::max(std::fabs(st[VX]) - S.rec.speed_max, 0.f);
            }
            out[k] = (float)(S.rec.distance * (dist / T + last) + S.rec.slip * slip / T + S.rec.yaw_rate * yaw / T +
                             S.rec.speed_max_w * fast / T);
        }
    }

    // drive reward (B,) and crash part (B,) (MPC_reward_jax).
    void mpc_reward(const float* states, int B, int T, int start, float* reward, float* crash) {
        const int count = matched_waypoint_count(start, T);
        std::vector<float> match_xy((size_t)count * 2), rows((size_t)count * 5);
        for (int j = 0; j < count; ++j) {
            const float* w = row(mod(start + j));
            match_xy[2 * j] = w[WX];
            match_xy[2 * j + 1] = w[WY];
            rows[5 * j] = w[WX];
            rows[5 * j + 1] = w[WY];
            rows[5 * j + 2] = w[WHEAD];
            rows[5 * j + 3] = w[WSPEED];
            rows[5 * j + 4] = w[WDRIFT];
        }
        const float consts[6] = {S.target_speed_factor, S.overspeed_w, S.arrival_w, S.border, S.clearance,
                                 std::max(S.clearance - S.ramp_end, 1e-3f)};
        std::vector<float> drive(B), closeness(B), depth(B);
        std::vector<uint8_t> active(B);
        std::vector<int32_t> late(B);
        hwk::drive_costs(states, B, T, match_xy.data(), rows.data(), count, consts, crash_end(T), drive.data(),
                         active.data(), late.data(), closeness.data(), depth.data());
        if (!off_track_) {
            for (int k = 0; k < B; ++k) drive[k] = active[k] ? drive[k] : 0.f;
        } else {
            recovery_cost(states, B, T, start, drive.data());
        }
        if (stall_phase_ == 1) backtrack_cost(states, B, T, drive.data());
        float scale = 1.f;
        for (int k = 0; k < B; ++k) scale = std::max(scale, std::fabs(drive[k]));
        for (int k = 0; k < B; ++k) {
            const float crash_steps = (float)late[k] + S.depth_w * depth[k];
            crash[k] = -(scale * 10.f * crash_steps);
            reward[k] = -drive[k] - (S.wall_w > 0.f ? S.wall_w * closeness[k] : 0.f);
        }
    }

    // ------------------------------------------------------------------ select_action
    void select_action(const float* state, float* action_out) {
        auto t = Clock::now();
        refresh_recovery(state);
        auto nominal = recover_mode_ ? std::nullopt : interpolated_waypoint_actions(state);
        const int horizon = nominal ? S.H : S.H_free;
        action_buffer_ = fit_horizon(action_buffer_, horizon);
        if (nominal) nominal = fit_horizon(*nominal, horizon);
        if (nominal && (int)mean_state_seed_.size() == (int)nominal->size()) nominal = mean_state_seed_;
        stats_[S_RECOVERY] += since(t);

        t = Clock::now();
        const int K = S.K;
        const std::vector<float>& nom = nominal ? *nominal : action_buffer_;
        actions_.resize((size_t)K * horizon * 2);
        sample_actions(nom, K, 1.f, S.rho, actions_.data());
        if (S.fallback) with_fallback_samples(K, nominal ? *nominal : action_buffer_, state[VX]);
        auto window = window_job();
        stats_[S_SAMPLES] += since(t);

        // Rollouts: the drive samples from the car with its delay queue, the window samples from the waypoint state.
        t = Clock::now();
        const int T = horizon + 1;
        const auto drive_pending = issued_actions();
        const int Kw = window ? window->count : 0;
        const bool together = window && horizon == S.H;
        const int B = K + (together ? Kw : 0);
        s0_.resize((size_t)B * D);
        pending_.resize((size_t)B * S.delay * 2);
        for (int k = 0; k < B; ++k) {
            const bool w = k >= K;
            std::copy(w ? window->start.data() : state, (w ? window->start.data() : state) + D, &s0_[(size_t)k * D]);
            const float* p = w ? window->pending.data() : drive_pending.data();
            std::copy(p, p + S.delay * 2, &pending_[(size_t)k * S.delay * 2]);
        }
        std::vector<float> all_actions;
        const float* act = actions_.data();
        if (together) {
            all_actions.resize((size_t)B * horizon * 2);
            std::copy(actions_.begin(), actions_.end(), all_actions.begin());
            std::copy(window->samples.begin(), window->samples.end(), all_actions.begin() + (size_t)K * horizon * 2);
            act = all_actions.data();
        }
        traj_.resize((size_t)B * T * D);
        hwk::rollout(model_, B, horizon, S.delay, s0_.data(), act, pending_.data(), traj_.data());
        const float* window_states = nullptr;
        int window_T = S.H + 1;
        if (window && together) {
            window_states = traj_.data() + (size_t)K * T * D;
        } else if (window) {
            std::vector<float> ws((size_t)Kw * D), wp((size_t)Kw * S.delay * 2);
            for (int k = 0; k < Kw; ++k) {
                std::copy(window->start.begin(), window->start.end(), &ws[(size_t)k * D]);
                std::copy(window->pending.begin(), window->pending.end(), &wp[(size_t)k * S.delay * 2]);
            }
            win_traj_.resize((size_t)Kw * window_T * D);
            hwk::rollout(model_, Kw, S.H, S.delay, ws.data(), window->samples.data(), wp.data(), win_traj_.data());
            window_states = win_traj_.data();
        }
        hwk::freeze_late(traj_.data(), K, T, crash_end(T), S.border);  // the drive samples only
        stats_[S_ROLLOUT] += since(t);

        t = Clock::now();
        const int nearest = nearest_waypoint_index(state);
        rewards_.resize(K);
        crash_.resize(K);
        mpc_reward(traj_.data(), K, T, nearest, rewards_.data(), crash_.data());
        stats_[S_COSTS] += since(t);

        t = Clock::now();
        total_.resize(K);
        weights_.resize(K);
        float mx = -INF;
        int best = 0;
        for (int k = 0; k < K; ++k) {
            total_[k] = rewards_[k] + crash_[k];
            if (total_[k] > mx) {
                mx = total_[k];
                best = k;
            }
        }
        double z = 0.0;
        for (int k = 0; k < K; ++k) {
            weights_[k] = std::exp((total_[k] - mx) / S.temperature);
            z += weights_[k];
        }
        double reward_mean = 0.0;
        for (int k = 0; k < K; ++k) {
            weights_[k] = (float)(weights_[k] / z);
            reward_mean += (double)rewards_[k] * weights_[k];
        }
        note_reward((float)reward_mean);
        std::vector<float> action_seq((size_t)horizon * 2, 0.f);
        for (int k = 0; k < K; ++k) {
            const float w = weights_[k];
            if (w == 0.f) continue;
            const float* a = &actions_[(size_t)k * horizon * 2];
            for (int i = 0; i < horizon * 2; ++i) action_seq[i] += w * a[i];
        }
        if (S.safe_mean) safe_mean(state, horizon, action_seq, best, drive_pending);
        action_buffer_.assign(action_seq.begin() + 2, action_seq.end());
        action_buffer_.push_back(action_seq[action_seq.size() - 2]);
        action_buffer_.push_back(action_seq[action_seq.size() - 1]);
        if (stall_phase_ != 1 && S.stall_on_blocked && blocked(state[VX], T, best))
            start_backtrack(state[X], state[Y], car_waypoint_, "the best plan does not get going");
        stats_[S_PLAN] += since(t);
        action_out[0] = action_seq[0];
        action_out[1] = action_seq[1];
        if (stall_phase_ == 1) {
            --reverse_left_;
            mean_state_seed_.clear();
            return;
        }
        mark_fragile();
        t = Clock::now();
        if (window) window_update(*window, window_states, window_T);
        stats_[S_WINDOW] += since(t);
        t = Clock::now();
        record_driven(state, action_out);
        fill_passed_waypoints(state, action_seq);
        // Next step's seed from the softmax-weighted mean states.
        std::vector<float> mean((size_t)T * D, 0.f);
        for (int k = 0; k < K; ++k) {
            const float w = weights_[k];
            if (w == 0.f) continue;
            const float* s = &traj_[(size_t)k * T * D];
            for (int i = 0; i < T * D; ++i) mean[i] += w * s[i];
        }
        for (int t2 = 0; t2 < T; ++t2) mean[t2 * D + TH] = std::atan2(mean[t2 * D + SIN], mean[t2 * D + COS]);
        mean_state_seed_ = seed_from_states(mean.data() + D, horizon);
        stats_[S_TABLE] += since(t);
    }

    void note_reward(float reward_mean) {
        const double previous = reward_ema_;
        reward_ema_ = S.ema_gamma * reward_ema_ + (1.0 - S.ema_gamma) * reward_mean;
        if (episode_timesteps_ < 100) return;
        const double drop = previous - reward_mean;
        if (drop > S.reward_drop * std::fabs(previous)) {
            if (!reward_drop_warned_) {
                reward_drop_warned_ = true;
                note_off_track_pass(true);
                log("[HirarchicalPlanner/cpp] Warning: reward %.4f fell under its EMA %.4f.", reward_mean, previous);
            }
            revert_nearby_waypoints();
            update_recover_mode(false);
        }
    }

    void safe_mean(const float* state, int horizon, std::vector<float>& action_seq, int best,
                   const std::vector<float>& drive_pending) {
        bool any_crash = false;
        for (float c : crash_) any_crash |= c < 0.f;
        if (crash_[best] < 0.f || !any_crash) return;
        const int T = horizon + 1;
        std::vector<float> path((size_t)T * D);
        hwk::rollout(model_, 1, horizon, S.delay, state, action_seq.data(), drive_pending.data(), path.data());
        const int end = crash_end(T);
        bool hit = false;
        for (int t = 0; t < end && !hit; ++t) corner_clearance(&path[(size_t)t * D], S.border, &hit);
        if (!hit) return;
        const float* a = &actions_[(size_t)best * horizon * 2];
        action_seq.assign(a, a + horizon * 2);
        std::fill(weights_.begin(), weights_.end(), 0.f);
        weights_[best] = 1.f;
    }

    bool blocked(float speed, int T, int best) const {
        if (speed >= S.creep_speed) return false;
        if (crash_[best] < 0.f) return true;
        float forward = 0.f;
        const float* s = &traj_[(size_t)best * T * D];
        for (int t = 1; t < T; ++t) forward += std::max(s[t * D + VX], 0.f);
        return forward * S.dt < S.blocked_distance;
    }

    void mark_fragile() {
        if (off_track_ || recover_mode_ || car_waypoint_ < 0) return;
        int crashing = 0;
        for (float c : crash_) crashing += c < 0.f;
        if ((float)crashing / (float)crash_.size() < S.fragile_share) return;
        for (int k = 0; k <= S.fragile_ahead; ++k) wp_fragile_at_[mod(car_waypoint_ + k)] = total_timesteps_;
    }

    // ------------------------------------------------------------------ laps
    Snapshot best_lap_snapshot(const Snapshot& start) const { return driven_table_ ? *driven_table_ : start; }

    static int count_changed(const Snapshot& a, const Snapshot& b, int n) {
        int changed = 0;
        const size_t ha = a.actions.size() / n;
        for (int i = 0; i < n; ++i) {
            bool c = !std::equal(&a.waypoints[(size_t)i * NC], &a.waypoints[(size_t)(i + 1) * NC], &b.waypoints[(size_t)i * NC]);
            c |= !std::equal(&a.actions[i * ha], &a.actions[(i + 1) * ha], &b.actions[i * ha]);
            changed += c;
        }
        return changed;
    }

    void update_permanent_freeze(float lap_time) {
        const int patience = S.freeze_laps;
        if (permanently_frozen_ || patience <= 0) return;
        auto start = freeze_lap_start_;
        freeze_lap_start_ = snapshot();
        if (start && (!freeze_best_lap_ || lap_time < *freeze_best_lap_)) {
            freeze_best_table_ = best_lap_snapshot(*start);
            freeze_best_lap_ = lap_time;
            freeze_laps_since_best_ = 0;
            return;
        }
        if (!freeze_best_lap_) return;
        if (++freeze_laps_since_best_ < patience) return;
        restore_snapshot(*freeze_best_table_);
        permanently_frozen_ = true;
        log("[HirarchicalPlanner/cpp] No lap faster than %.3fs for %d laps: waypoints put back to that lap's and frozen permanently",
            *freeze_best_lap_, patience);
    }

    void update_best_lap_table(float lap_time) {
        const int patience = S.best_lap_reset;
        if (patience < 0 || permanently_frozen_) return;
        auto start = lap_start_table_;
        lap_start_table_ = snapshot();
        if (start && (!best_lap_time_ || lap_time < *best_lap_time_)) {
            best_lap_time_ = lap_time;
            best_lap_table_ = best_lap_snapshot(*start);
            laps_since_best_ = 0;
            return;
        }
        if (!best_lap_time_) return;
        if (++laps_since_best_ < patience) return;
        restore_snapshot(*best_lap_table_);
        lap_start_table_ = best_lap_table_;
        laps_since_best_ = 0;
        log("[HirarchicalPlanner/cpp] No lap faster than %.3fs for %d laps: table put back to that lap's", *best_lap_time_, patience);
    }
};

std::string g_error;

template <typename F>
auto guarded(F f, decltype(f()) fallback) -> decltype(f()) {
    try {
        return f();
    } catch (const std::exception& e) {
        g_error = e.what();
        std::fprintf(stderr, "[hw_planner] %s\n", e.what());
        return fallback;
    }
}

}  // namespace

extern "C" {

const char* hwp_error() { return g_error.c_str(); }

// car: 32 vehicle parameters; table (n, 14); anchor/tangent/ortho (n, 2); track field grids (h, w) or null.
void* hwp_create(const char* config, const float* car, int mode, float dt, int substeps, int n, const float* table,
                 const float* anchor, const float* tangent, const float* ortho, const float* track_length,
                 const int32_t* track_label, const float* track_direction, int grid_h, int grid_w) {
    return guarded(
        [&]() -> void* {
            Config c(config);
            CarModel m;
            std::copy(car, car + 32, m.car);
            m.mode = mode;
            m.dt = dt;
            m.substeps = substeps;
            const auto& denorm = c.list("DENORM");
            m.denorm0 = (float)denorm.at(0);
            m.denorm1 = (float)denorm.at(1);
            m.cap = (float)c.num("CAP");
            return new Planner(c, m, n, table, anchor, tangent, ortho, track_length, track_label, track_direction,
                               grid_h, grid_w);
        },
        nullptr);
}

void hwp_destroy(void* p) { delete static_cast<Planner*>(p); }
int hwp_step(void* p, const float* state, float* out) {
    return guarded([&] { static_cast<Planner*>(p)->step(state, out); return 0; }, -1);
}
void hwp_lap(void* p, float lap_time) { static_cast<Planner*>(p)->lap(lap_time); }
void hwp_episode_done(void* p, int crash) { static_cast<Planner*>(p)->episode_done(crash != 0); }
void hwp_reset(void* p) { static_cast<Planner*>(p)->reset(); }
float* hwp_table(void* p) { return static_cast<Planner*>(p)->table(); }
float* hwp_actions(void* p) { return static_cast<Planner*>(p)->actions_table(); }
uint8_t* hwp_known(void* p) { return static_cast<Planner*>(p)->known_table(); }
float* hwp_off_track_counts(void* p) { return static_cast<Planner*>(p)->off_track_counts(); }
int hwp_frozen(void* p) { return static_cast<Planner*>(p)->frozen(); }
int hwp_recover_mode(void* p) { return static_cast<Planner*>(p)->recover_mode(); }
// Mean ms per step of: recovery, samples, rollout, costs, plan, window update, table/seed, total; then the step count.
void hwp_stats(void* p, double* out) { static_cast<Planner*>(p)->stats(out); }
void hwp_clear_stats(void* p) { static_cast<Planner*>(p)->clear_stats(); }

}  // extern "C"
