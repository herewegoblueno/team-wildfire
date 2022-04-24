#include "simulator.h"
#include <math.h>


float get_q_v(Voxel* v) {return v->getLastFrameState()->q_v;}
float get_q_c(Voxel* v) {return v->getLastFrameState()->q_c;}
float get_q_r(Voxel* v) {return v->getLastFrameState()->q_r;}
void Simulator::stepVoxelWater(Voxel* v, int deltaTimeInMs)
{
    float q_v = v->getLastFrameState()->q_v;
    float q_c = v->getLastFrameState()->q_c;
    float q_r = v->getLastFrameState()->q_r;
    glm::vec3 u = v->getCurrentState()->u; // at this point u is already calculated
    float h = v->centerInWorldSpace.y; // need discussion

    q_v = advect(q_v, u, v->getGradient(get_q_v), deltaTimeInMs);
    q_c = advect(q_c, u, v->getGradient(get_q_c), deltaTimeInMs);
    q_r = advect(q_r, u, v->getGradient(get_q_r), deltaTimeInMs);
    float q_vs = saturate(absolute_pres(h), absolute_temp(h));

    float E_r = q_r*evaporation_rate*std::max(q_vs - q_v, 0.f);// evaporation of rain Fire Eq.22
    float A_c = autoconverge_cloud*(q_c - 0.001); // below Stormscape Eq.24
    float K_c = raindrop_accelerate*q_c*q_r;  // below Stormscape Eq.24
    q_v = q_v + std::min(q_vs - q_v, q_c) + E_r;
    q_c = q_c - std::min(q_vs - q_v, q_c) - A_c - K_c;
    q_r = A_c + K_c - E_r;

    float X_v = mole_fraction(q_v);
    float M_th = avg_mole_mass(X_v);
    float Y_v = X_v*18.02/M_th;
    float gamma_th = isentropic_exponent(Y_v);
    float c_th_p = heat_capacity(gamma_th, M_th);
    float evp_temp = 2.5/c_th_p*mole_fraction(-std::min(q_vs-q_v, q_c));

    v->getCurrentState()->q_v = q_v;
    v->getCurrentState()->q_c = q_c;
    v->getCurrentState()->q_r = q_r;
    v->getCurrentState()->temperature += evp_temp;
}



// Advection function based on *Stable Fluids* Jos Stam 1999
float Simulator::advect(float field, glm::vec3 vel, glm::vec3 field_grad, float dt)
{
    float df_dt = -glm::dot(vel, field_grad);
    return field + df_dt*dt;
}

// saturation ratio calculation of Eq.16 Stormscape
float Simulator::saturate(float pressure, float temperature)
{
    return 380.16/pressure*exp(17.67*temperature/(temperature+243.5));
}


// absolute temperature calculation based on Eq.27 Stormscape
float Simulator::absolute_temp(float height)
{
    height = height_scale*height;
    return sealevel_temperature - 0.0065*height;
}

// absolute pressure calculation based on Eq.28 Stormscape
float Simulator::absolute_pres(float height)
{
    height = height_scale*height;
    float tmp = 1 - 0.0065*height/sealevel_temperature;
    return sealevel_pressure*pow(tmp, 5.2561);
}

 // Stormscape Eq.9
float Simulator::mole_fraction(float ratio)
{
    if (ratio < 0) ratio = 0; // fail safe
    return ratio / (1 + ratio);
}

 // Stormscape Eq.7
float Simulator::avg_mole_mass(float ratio)
{
    float real_mass = 18.02*ratio + 28.96*(1-ratio);
//    return real_mass*mass_scale;
    return real_mass;
}

// Stormscape Eq.11
float Simulator::isentropic_exponent(float ratio)
{
    return ratio*1.33 + (1-ratio)*1.4;
}

float Simulator::heat_capacity(float gamma, float mass)
{
    return gamma*8.3/(mass*(gamma-1));
}
