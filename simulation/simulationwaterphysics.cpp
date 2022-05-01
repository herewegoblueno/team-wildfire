#include "simulator.h"
#include <math.h>
#include <iostream>

//extern "C" double* addVectorsGPU(double* a, double* b, int n);
//extern "C" double* getDisplacement(double* a, double* b, double t, int n);


void Simulator::stepVoxelWater(Voxel* v, double deltaTimeInMs)
{
    double q_v = v->getLastFrameState()->q_v;
    double q_c = v->getLastFrameState()->q_c;
    double q_r = v->getLastFrameState()->q_r;
    glm::dvec3 u = v->getCurrentState()->u; // at this point u is already calculated
    double h = v->centerInWorldSpace.y; // need discussion

    q_v = advect(get_q_v, u, deltaTimeInMs, v);
    q_c = advect(get_q_c, u, deltaTimeInMs, v);
    q_r = advect(get_q_r, u, deltaTimeInMs, v);

    float q_vs = saturate(absolute_pres(h), ambientTemperatureFunc(v->centerInWorldSpace));
    float E_r = q_r*evaporation_rate*std::max(q_vs - q_v, 0.);// evaporation of rain Fire Eq.22
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
double Simulator::advect(double (*func)(Voxel *), glm::dvec3 vel, double dt, Voxel* v)
{
    glm::dvec3 pos = v->centerInWorldSpace - vel*dt;
    Voxel* v_trace = v->grid->getVoxelClosestToPoint(glm::vec3(pos[0], pos[1], pos[2]));
    return func(v_trace);
}

glm::dvec3 Simulator::advect_vel(glm::dvec3 vel, double dt, Voxel* v)
{
    glm::dvec3 pos = v->centerInWorldSpace - vel*dt;
    Voxel* v_trace = v->grid->getVoxelClosestToPoint(glm::vec3(pos[0], pos[1], pos[2]));
    if(v_trace==nullptr) return glm::dvec3(0,0,0);
    return v_trace->getLastFrameState()->u;
}

// saturation ratio calculation of Eq.16 Stormscape
double Simulator::saturate(double pressure, double temperature)
{
    return 380.16/pressure*exp(17.67*temperature/(temperature+243.5));
}


// absolute temperature calculation based on Eq.27 Stormscape
double Simulator::absolute_temp(double height)
{
    height = height_scale*height;
    return sealevel_temperature - 0.0065*height - 273.15;
}

// absolute pressure calculation based on Eq.28 Stormscape
double Simulator::absolute_pres(double height)
{
    height = height_scale*height;
    float tmp = 1 - 0.0065*height/sealevel_temperature;
    return sealevel_pressure*std::pow(tmp, 5.2561);
}

 // Stormscape Eq.9
double Simulator::mole_fraction(double ratio)
{
    if (ratio < 0) ratio = 0; // fail safe
    return ratio / (1 + ratio);
}

 // Stormscape Eq.7
double Simulator::avg_mole_mass(double ratio)
{
    float real_mass = 18.02*ratio + 28.96*(1-ratio);
//    return real_mass*mass_scale;
    return real_mass;
}

// Stormscape Eq.11
double Simulator::isentropic_exponent(double ratio)
{
    return ratio*1.33 + (1-ratio)*1.4;
}

double Simulator::heat_capacity(double gamma, double mass)
{
    return gamma*8.3/(mass*(gamma-1));
}
