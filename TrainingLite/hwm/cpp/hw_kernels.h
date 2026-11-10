// Kernels shared by the C API (hw_cpp.HwCpp) and the C++ planner (hw_planner.cpp).
#pragma once

#include <cstdint>

namespace hwk {

constexpr int D = 10;  // state: angular_vel_z, vx, vy, theta, cos, sin, x, y, slip, steering
constexpr int PD = 0, VX = 1, VY = 2, TH = 3, COS = 4, SIN = 5, X = 6, Y = 7, SLIP = 8, STEER = 9;

struct CarModel {
    float car[32];  // VehicleParameters.to_np_array()
    int mode;       // 0 pacejka, 1 ks, 2 ks_pacejka
    float dt;       // control step
    int substeps;
    float denorm0, denorm1, cap;
};

// States (B, H + 1, D) from s0 (B, D), actions (B, H, 2) and per-sample delay queues (B, delay, 2).
void rollout(const CarModel& m, int batch, int horizon, int delay, const float* s0, const float* actions,
             const float* pending, float* out);

// Wall map (process wide). A blank map has no walls.
void set_map(const float* grid, int h, int w, double ox, double oy, double oc, double os, double res,
             const double* cx, const double* cy);
void set_blank_map();
bool map_blank();
// Map cell (row, col) of a world point, clamped, and whether it is on the map (WallGeometry._grid_cells).
void map_cell(double x, double y, int* row, int* col, bool* inside);
int map_width();
// Wall distance at a world point, 0 off the map (WallGeometry.distance_at); infinity on a blank map.
float wall_distance(double x, double y);
// Smallest wall distance of the body corners at state s (0 off the map); crashed: a corner within border or off the map.
float corner_clearance(const float* s, float border, bool* crashed);

void freeze_late(float* traj, int batch, int steps, int end, float border);
void drive_costs(const float* states, int batch, int steps, const float* match_xy, const float* rows, int count,
                 const float* consts, int crash_end, float* drive, uint8_t* active, int32_t* late, float* closeness,
                 float* depth);

}  // namespace hwk
