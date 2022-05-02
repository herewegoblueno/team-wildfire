#include "simulator.h"
#include <math.h>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <iostream>
//#include <cuda_runtime.h>
extern "C" void jacobiGPU(double* diag, double* rhs, int* id_xyz, int N, int Ni, int iter);


double get_vorticity_len(Voxel* v) {return v->getVorticity().length();}

void Simulator::stepVoxelWind(Voxel* v, double deltaTimeInMs)
{
    dvec3 u = v->getLastFrameState()->u;
    dvec3 u_f = v->getCurrentState()->u;

    u = advect_vel(u_f, deltaTimeInMs, v);

    u = vorticity_confinement(u, v, deltaTimeInMs);

    double T_th = v->getCurrentState()->temperature;
    double T_air = absolute_temp(v->centerInWorldSpace.y);
    double q_v = v->getLastFrameState()->q_v;
    float X_v = mole_fraction(q_v);
    float M_th = avg_mole_mass(X_v);
    dvec3 buoyancy_gravity(0, gravity_acceleration, 0); // upward
    dvec3 buoyancy = -buoyancy_gravity*(28.96*T_th/(M_th*T_air) - 1);
    u = u + buoyancy*(double)deltaTimeInMs; // can't think of externel force

    if(std::isnan(u.x))
    {
        std::cout << "[velocity nan error]";
    }
    v->getCurrentState()->u = u;
}

// vorticity confinement origrinated from Steinhoff and Underhill [1994],
dvec3 Simulator::vorticity_confinement(glm::dvec3 u, Voxel* v, double time)
{
    dvec3 vorticity = v->getVorticity() + 0.000001;
    dvec3 d_vorticity = v->getGradient(get_vorticity_len) + 0.000001;
    d_vorticity = glm::normalize(d_vorticity);
    dvec3 f_omega = 3*vorticity_epsilon*v->grid->cellSideLength()*glm::cross(d_vorticity, vorticity);
    return u + f_omega*time;
}



void Simulator::pressure_projection_Jacobi_cuda(double* diag, double* rhs, int* id_xyz, int N, int Ni, int iter)
{
    jacobiGPU(diag, rhs, id_xyz, N, Ni, iter);
}









