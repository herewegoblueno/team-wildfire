#include "simulator.h"
#include <math.h>


double get_verticity_len(Voxel* v) {return v->getVerticity().length();}

void Simulator::stepVoxelWind(Voxel* v, int deltaTimeInMs)
{
    dvec3 u = v->getLastFrameState()->u;
    dvec3 u_f = v->getCurrentState()->u;

    u.x = advect(u.x, u_f, v->getGradient(get_q_ux), deltaTimeInMs);
    u.y = advect(u.y, u_f, v->getGradient(get_q_uy), deltaTimeInMs);
    u.z = advect(u.z, u_f, v->getGradient(get_q_uz), deltaTimeInMs);

    u = verticity_confinement(u, v, deltaTimeInMs);

    double T_th = v->getCurrentState()->temperature;
    double T_air = absolute_temp(v->centerInWorldSpace.y);
    double q_v = v->getLastFrameState()->q_v;
    float X_v = mole_fraction(q_v);
    float M_th = avg_mole_mass(X_v);
    dvec3 buoyancy_gravity(0, gravity_acceleration, 0); // upward
    dvec3 buoyancy = buoyancy_gravity*(28.96*T_th/(M_th*T_air) - 1);
    u = u + buoyancy*(double)deltaTimeInMs; // can't think of externel force

    v->getCurrentState()->u = u;
}

// verticity confinement origrinated from Steinhoff and Underhill [1994],
dvec3 Simulator::verticity_confinement(glm::dvec3 u, Voxel* v, double time)
{
    dvec3 verticity = v->getVerticity();
    dvec3 d_verticity = v->getGradient(get_verticity_len);
    d_verticity = glm::normalize(d_verticity);
    dvec3 f_omega = verticity_epsilon*v->grid->cellSideLength()*glm::cross(d_verticity, verticity);
    return u + f_omega*time;
}


// verticity confinement origrinated from Steinhoff and Underhill [1994],
dvec3 Simulator::pressure_projection(glm::dvec3 u, Voxel* v, double time)
{

}
