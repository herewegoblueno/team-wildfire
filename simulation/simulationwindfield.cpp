#include "simulator.h"
#include <math.h>
#include <Eigen/Cholesky>
#include <iostream>


double get_vorticity_len(Voxel* v) {
    return v->getVorticity().length();
}

void Simulator::stepVoxelWind(Voxel* v, double deltaTimeInMs)
{

    #ifndef CUDA_FLUID
    return;
    #endif

    int x=v->XIndex, y=v->YIndex, z=v->ZIndex;
    dvec3 u = v->getLastFrameState()->u;

    if(x==20 && y==20 && z==20) cout << "[" << u.x << "," << u.y << "," << u.z << "]-";
    u = advect_vel(u, deltaTimeInMs, v);
    if(x==20 && y==20 && z==20) cout << "[" << u.x << "," << u.y << "," << u.z << "]\n" << flush;
    dvec3 u_laplace = v->getVelLaplace();
    u += viscosity*u_laplace*deltaTimeInMs;
    u = vorticity_confinement(u, v, deltaTimeInMs);
    double T_th = simTempToWorldTemp(v->getCurrentState()->temperature);
    double T_air = absolute_temp(v->centerInWorldSpace.y);
    double q_v = v->getLastFrameState()->q_v;
    float X_v = mole_fraction(q_v);
    float M_th = avg_mole_mass(X_v);

    dvec3 buoyancy_gravity(0, gravity_acceleration, 0);
    dvec3 buoyancy = buoyancy_gravity*(28.96*T_th/(M_th*T_air) - 1);
    u = u + buoyancy*deltaTimeInMs; // can't think of externel force


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

    return deltaP*time/(cell_size*air_density);
}

double calc_density_term(double cell_size, double time)
{
    return time/(air_density*cell_size*cell_size);
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
// vorticity confinement origrinated from Steinhoff and Underhill [1994],
dvec3 vorticity_confinement(glm::dvec3 u, Voxel* v, double time)
{
    dvec3 vorticity = v->getVorticity() + 0.000001;
    dvec3 d_vorticity = v->getGradient(get_vorticity_len) + 0.000001;
    d_vorticity = glm::normalize(d_vorticity);
    dvec3 f_omega = 0.1*vorticity_epsilon*v->grid->cellSideLength()*glm::cross(d_vorticity, vorticity);
    return u + f_omega*time;
}

glm::dvec3 advect_vel(dvec3 u, double dt, Voxel* v)
{
    VoxelGrid* grid = v->grid;
    int x=v->XIndex, y=v->YIndex, z=v->ZIndex;
    double cell_size = grid->cellSideLength();
    float ua, ub;

    ua = grid->getVel(x-1,y,z,0);
    ub = grid->getVel(x,y,z,0);
    u.x -= (ub - ua)/cell_size*(ub+ua)/2*dt;
    ua = grid->getVel(x,y-1,z,1);
    ub = grid->getVel(x,y,z,1);
    u.y -= (ub - ua)/cell_size*(ub+ua)/2*dt;
    ua = grid->getVel(x,y,z-1,2);
    ub = grid->getVel(x,y,z,2);
    u.z -= (ub - ua)/cell_size*(ub+ua)/2*dt;
    return u;
}



void pressure_projection_Jacobi_cuda(double* diag, double* rhs, int* id_xyz, int N, int Ni, int iter)
{
//    #ifdef CUDA_FLUID
//    jacobiGPU(diag, rhs, id_xyz, N, Ni, iter);
//    #endif
}



//// semi lagrangian advection
//glm::dvec3 advect_vel(dvec3 u, double dt, Voxel* v)
//{
//    VoxelGrid* grid = v->grid;
//    int x=v->XIndex, y=v->YIndex, z=v->ZIndex;
//    double half_cell = grid->cellSideLength()/2;
//    dvec3 out;
//    dvec3 du, pos, pos_v = v->centerInWorldSpace;
//    // for ux(x+0.5, y, z)
//    du.x = v->getLastFrameState()->u.x;
//    du.y = (grid->getVel(x,  y-1,z  , 1) + grid->getVel(x,  y, z, 1) +
//            grid->getVel(x+1,y-1,z  , 1) + grid->getVel(x+1,y, z, 1))/4;
//    du.z = (grid->getVel(x,  y,  z-1, 2) + grid->getVel(x,  y, z, 2) +
//            grid->getVel(x+1,y,  z-1, 2) + grid->getVel(x+1,y, z, 2))/4;
//    pos = pos_v + dvec3(half_cell, 0, 0) - du*dt;
//    out.x = grid->getVelInterpolatePoint(vec3(pos)).x;
//    // for uy(x, y+0.5, z)
//    du.x = (grid->getVel(x-1,y  ,z, 0) + grid->getVel(x,y,  z, 0) +
//            grid->getVel(x-1,y+1,z, 0) + grid->getVel(x,y+1,z, 0))/4;
//    du.y = v->getLastFrameState()->u.x;
//    du.z = (grid->getVel(x,  y,  z-1, 2) + grid->getVel(x,y,  z, 2) +
//            grid->getVel(x,  y+1,z-1, 2) + grid->getVel(x,y+1,z, 2))/4;
//    pos = pos_v + dvec3(0, half_cell, 0) - du*dt;
//    out.y = grid->getVelInterpolatePoint(vec3(pos)).y;
//    // for uz(x, y, z+0.5)
//    du.x = (grid->getVel(x-1,y,z  , 0) + grid->getVel(x,y,z  , 0) +
//            grid->getVel(x-1,y,z+1, 0) + grid->getVel(x,y,z+1, 0))/4;
//    du.y = (grid->getVel(x, y-1,z , 1) + grid->getVel(x,y,z  , 1) +
//            grid->getVel(x,y-1,z+1, 1) + grid->getVel(x,y,z+1, 1))/4;
//    du.z = v->getLastFrameState()->u.z;
//    pos = pos_v + dvec3(0, 0, half_cell) - du*dt;
//    out.z = grid->getVelInterpolatePoint(vec3(pos)).z;
//    return out;
//}

