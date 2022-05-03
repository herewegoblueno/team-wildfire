#include "simulator.h"
#include <math.h>
#include <Eigen/Cholesky>
#include <iostream>

#ifdef CUDA_FLUID
extern "C" void jacobiGPU(double* diag, double* rhs, int* id_xyz, int N, int Ni, int iter);
#endif

double get_vorticity_len(Voxel* v) {return v->getVorticity().length();}

void Simulator::stepVoxelWind(Voxel* v, double deltaTimeInMs)
{

    #ifdef CUDA_FLUID
    #else
    return;
    #endif

    dvec3 u = v->getLastFrameState()->u;
    dvec3 d_ux = v->getGradient(get_q_ux);
    dvec3 d_uy = v->getGradient(get_q_uy);
    dvec3 d_uz = v->getGradient(get_q_uz);

    int debug_pos[3] = {v->XIndex, v->YIndex, v->ZIndex};

    // the original euler step is definitely impossible
//    u.x -= glm::dot(d_ux, u)*deltaTimeInMs;
//    u.y -= glm::dot(d_uy, u)*deltaTimeInMs;
//    u.z -= glm::dot(d_uz, u)*deltaTimeInMs;
    // another advection scheme
//    u = advect_vel(u, deltaTimeInMs, v);
    if(glm::length(u)>100)
    {
        std::cout << "[large vel after advect]";
    }

//    u.x += viscosity*(d_ux.x+d_ux.y+d_ux.z)*deltaTimeInMs;
//    u.y += viscosity*(d_uy.x+d_uy.y+d_uy.z)*deltaTimeInMs;
//    u.z += viscosity*(d_uz.x+d_uz.y+d_uz.z)*deltaTimeInMs;
    if(glm::length(u)>100)
    {
//        std::cout << "[large vel after diffuse]";
    }
//    u = vorticity_confinement(u, v, deltaTimeInMs);
    if(glm::length(u)>100)
    {
//        std::cout << "[large vel after vorticity]";
    }
    double T_th = simTempToWorldTemp(v->getCurrentState()->temperature);
    double T_air = absolute_temp(v->centerInWorldSpace.y);
    double q_v = v->getLastFrameState()->q_v;
    float X_v = mole_fraction(q_v);
    float M_th = avg_mole_mass(X_v);

    dvec3 buoyancy_gravity(0, gravity_acceleration, 0);
    dvec3 buoyancy = buoyancy_gravity*(28.96*T_th/(M_th*T_air) - 1);
    u = u + buoyancy*deltaTimeInMs; // can't think of externel force

    if(glm::length(u)>100)
    {
        std::cout << "[large vel after buoyancy]";
    }
    v->getCurrentState()->u = u;
}

glm::dvec3 calc_pressure_effect(int x, int y, int z, int resolution, double* pressure, double time, double cell_size)
{
    glm::dvec3 deltaP(0,0,0);
    int face_num = resolution*resolution;
    int index = x*face_num+y*resolution+z;
    if(x<resolution-1) deltaP.x = pressure[index+face_num]-pressure[index];
    else deltaP.x = 0;
    if(y<resolution-1) deltaP.y = pressure[index+resolution]-pressure[index];
    else deltaP.y = 0;
    if(z<resolution-1) deltaP.z = pressure[index+1] - pressure[index];
    else deltaP.z += 0;

    return deltaP*time/(cell_size*air_density*0.01);
}

double calc_density_term(double cell_size, double time)
{
    return time/(air_density*cell_size*cell_size*0.01);
}


// vorticity confinement origrinated from Steinhoff and Underhill [1994],
dvec3 vorticity_confinement(glm::dvec3 u, Voxel* v, double time)
{
    dvec3 vorticity = v->getVorticity() + 0.000001;
    dvec3 d_vorticity = v->getGradient(get_vorticity_len) + 0.000001;
    d_vorticity = glm::normalize(d_vorticity);
    dvec3 f_omega = 0.1*vorticity_epsilon*v->grid->cellSideLength()*glm::cross(d_vorticity, vorticity);
    return u + f_omega*time;
}

glm::dvec3 advect_vel(glm::dvec3 vel, double dt, Voxel* v)
{
    glm::dvec3 pos = v->centerInWorldSpace - vel*dt;
    VoxelPhysicalData data = v->grid->getStateInterpolatePoint(glm::vec3(pos[0], pos[1], pos[2]));
    return data.u;
}

void fill_jacobi_rhs(Voxel* v, int resolution, int index, double density_term,
                     double* diag_A, double* rhs, int* id_xyz)
{
    double diag = 6;
    glm::dvec3 gradient = v->getVelGradient();
    double this_rhs = -(gradient.x + gradient.y + gradient.z)/density_term;

    if(v->XIndex>resolution-2) diag --;
    if(v->XIndex<1) diag --;
    if(v->YIndex>resolution-2) diag --;
    if(v->YIndex<1) diag --;
    if(v->ZIndex>resolution-2) diag --;
    if(v->ZIndex<1) diag --;

    diag_A[0] = diag;
    id_xyz[0] = v->XIndex;
    id_xyz[1] = v->YIndex;
    id_xyz[2] = v->ZIndex;

    assert(!std::isnan(this_rhs));
    rhs[0] = this_rhs;
}

void pressure_projection_Jacobi_cuda(double* diag, double* rhs, int* id_xyz, int N, int Ni, int iter)
{
    #ifdef CUDA_FLUID
    jacobiGPU(diag, rhs, id_xyz, N, Ni, iter);
    #endif
}






