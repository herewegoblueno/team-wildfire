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
    dvec3 d_ux = v->getGradient(get_q_ux);
    dvec3 d_uy = v->getGradient(get_q_uy);
    dvec3 d_uz = v->getGradient(get_q_uz);

    // the original euler step is definitely impossible
//    u.x -= glm::dot(d_ux, u);
//    u.y -= glm::dot(d_uy, u);
//    u.z -= glm::dot(d_uz, u);
    // another advection scheme
    u = advect_vel(u, deltaTimeInMs, v);
    if(glm::length(u)>100)
    {
        std::cout << "[large vel after advect]";
    }

    u.x += 0.1*viscosity*(d_ux.x+d_ux.y+d_ux.z);
    u.y += 0.1*viscosity*(d_uy.x+d_uy.y+d_uy.z);
    u.z += 0.1*viscosity*(d_uz.x+d_uz.y+d_uz.z);
    if(glm::length(u)>100)
    {
        std::cout << "[large vel after diffuse]";
    }
    u = vorticity_confinement(u, v, deltaTimeInMs);
    if(glm::length(u)>100)
    {
        std::cout << "[large vel after vorticity]";
    }
    double T_th = v->getCurrentState()->temperature;
    double T_air = absolute_temp(v->centerInWorldSpace.y);
    double q_v = v->getLastFrameState()->q_v;
    float X_v = mole_fraction(q_v);
    float M_th = avg_mole_mass(X_v);
    dvec3 buoyancy_gravity(0, -gravity_acceleration, 0);
    dvec3 buoyancy = buoyancy_gravity*(28.96*T_th/(M_th*T_air) - 1);
    u = u + (buoyancy + buoyancy_gravity)*(double)deltaTimeInMs; // can't think of externel force

    if(glm::length(u)>100)
    {
        std::cout << "[large vel after pressure]";
    }
    v->getCurrentState()->u = u;
}

// vorticity confinement origrinated from Steinhoff and Underhill [1994],
dvec3 Simulator::vorticity_confinement(glm::dvec3 u, Voxel* v, double time)
{
    dvec3 vorticity = v->getVorticity() + 0.000001;
    dvec3 d_vorticity = v->getGradient(get_vorticity_len) + 0.000001;
    d_vorticity = glm::normalize(d_vorticity);
    dvec3 f_omega = 0.00*vorticity_epsilon*v->grid->cellSideLength()*glm::cross(d_vorticity, vorticity);
    return u + f_omega*time;
}


glm::dvec3 Simulator::advect_vel(glm::dvec3 vel, double dt, Voxel* v)
{
    glm::dvec3 pos = v->centerInWorldSpace - vel*dt;
    VoxelPhysicalData data = v->grid->getStateInterpolatePoint(glm::vec3(pos[0], pos[1], pos[2]));
    return data.u;
}



void Simulator::pressure_projection_Jacobi_cuda(double* diag, double* rhs, int* id_xyz, int N, int Ni, int iter)
{
    jacobiGPU(diag, rhs, id_xyz, N, Ni, iter);
}









