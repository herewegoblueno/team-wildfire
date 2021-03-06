#ifndef FLUID_H
#define FLUID_H
#include "voxels/voxelgrid.h"
#include "physics.h"

#ifdef CUDA_FLUID
extern "C" void processWindGPU(double* grid_temp, double* grid_q_v, double* grid_q_c, double* grid_q_r,
                               double* grid_h, double* grid_humidity,
                               double* u_xyz, int* id_xyz, int jacobi_iter, double f[3],
                               int resolution, double cell_size, float dt);
#endif

// water particle related equation
double advect(double (*func)(Voxel *), glm::dvec3 vel, double dt, Voxel* v);
dvec3  advect_vel(dvec3 u, double dt, Voxel* v);
double saturate(double pressure, double temperature);
double absolute_temp(double height);
double absolute_pres(double height);
double mole_fraction(double ratio);
double avg_mole_mass(double ratio);
double isentropic_exponent(double ratio);
double heat_capacity(double gamma, double mass);
double get_vorticity_len(Voxel* v);
double calc_density_term(double cell_size, double deltaTime);

// wind related equation
void pressure_projection_Jacobi_cuda(double* diag, double* rhs, int* id_xyz, int N, int Ni, int iter);



#endif // FLUID_H
