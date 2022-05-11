#include "simulator.h"
#include <math.h>
#include <iostream>
using namespace std;

void Simulator::stepVoxelWater(Voxel* v, double deltaTimeInMs)
{
    double q_v = v->getLastFrameState()->q_v;
    double q_c = v->getLastFrameState()->q_c;
    double q_r = v->getLastFrameState()->q_r;
    glm::dvec3 u = v->getCurrentState()->u; // at this point u is already calculated
    double h = v->centerInWorldSpace.y; // need discussion

    double cell_size = v->grid->cellSideLengthForGradients();
    int resolution = v->grid->getResolution();

    dvec3 grad_v = v->getGradient(get_q_v);
    dvec3 grad_c = v->getGradient(get_q_c);
    dvec3 grad_r = v->getGradient(get_q_r);
    dvec3 u_center = (v->getNegfaceVel(true)+u)/2.;

    // grad correction (boundary condition)
    if(v->XIndex==0 || v->XIndex==resolution-1) {
        grad_v.x = 0; grad_c.x = 0; grad_r.x = 0;
    }
    if(v->ZIndex==0 || v->ZIndex==resolution-1) {
        grad_v.z = 0; grad_c.z = 0; grad_r.z = 0;
    }
    if(v->YIndex==1 || v->YIndex==resolution-1) {
        grad_v.y = 0; grad_c.y = 0; grad_r.y = 0;
    }
    if(v->YIndex==1) // random evaporation from land
    {
        grad_v.y = -(3*std::sin(v->XIndex*3) + 1*std::sin(v->ZIndex*5) + 0.5*std::sin(v->ZIndex*7)+4.5)/cell_size;
//        v->getCurrentState()->temperature += (3*std::sin(v->XIndex*3) + 1*std::sin(v->ZIndex*5)
//                                              + 0.5*std::sin(v->ZIndex*7)+4.5)*0.01;
    }


    q_v -= glm::dot(grad_v, u_center)*deltaTimeInMs*500;
    q_c -= glm::dot(grad_c, u_center)*deltaTimeInMs*500;
    q_r -= glm::dot(grad_r, u_center)*deltaTimeInMs*500;

    float X_v = mole_fraction(q_v);
    float M_th = avg_mole_mass(X_v);
    float Y_v = X_v*18.02/M_th;
    float gamma_th = isentropic_exponent(Y_v);
    float c_th_p = heat_capacity(gamma_th, M_th);

//    double ambient_temperature = absolute_temp(v->centerInWorldSpace.y);
    double temperature = simTempToWorldTemp(v->getCurrentState()->temperature);
    double abs_pres = absolute_pres(h);
    double abs_temp = (temperature+273.15)*std::pow(abs_pres/100000, 1/0.287)-273.15;
    float q_vs = saturate(absolute_pres(h), abs_temp);
    float E_r = q_r*evaporation_rate*std::min(std::max(q_vs - q_v, 0.), 10.);// evaporation of rain Fire Eq.22
    float A_c = autoconverge_cloud*(q_c - 0.001); // below Stormscape Eq.24
    float K_c = raindrop_accelerate*q_c*q_r;  // below Stormscape Eq.24
    q_v = q_v + std::min(q_vs - q_v, q_c) + E_r;
    q_c = q_c - std::min(q_vs - q_v, q_c) - A_c - K_c;
    q_r = A_c + K_c - E_r;

    float evp_temp = 2.5/c_th_p/0.287*mole_fraction(-std::min(q_vs-q_v, q_c));

//    if(v->XIndex==8 && v->YIndex==2 && v->ZIndex==15)
//    {
//        cout << "q_v change:" << glm::dot(grad_v, u_center)*deltaTimeInMs*100 << " u_y:" << u_center.y ;
//        cout << " evp temp:" << evp_temp << endl << flush;
//    }

    v->getCurrentState()->q_v = q_v;
    v->getCurrentState()->q_c = q_c;
    v->getCurrentState()->q_r = q_r;
    v->getCurrentState()->temperature += evp_temp;
}



// Advection function based on *Stable Fluids* Jos Stam 1999
double advect(double (*func)(Voxel *), glm::dvec3 vel, double dt, Voxel* v)
{
    glm::dvec3 pos = v->centerInWorldSpace - vel*dt;
    Voxel* v_trace = v->grid->getVoxelClosestToPoint(glm::vec3(pos[0], pos[1], pos[2]));
    return func(v_trace);
}


// saturation ratio calculation of Eq.16 Stormscape
double saturate(double pressure, double temperature)
{
    return 380.16/pressure*exp(17.67*temperature/(temperature+243.5));
}


// absolute temperature calculation based on Eq.27 Stormscape
double absolute_temp(double height)
{
//    height = height_scale*height;
    height = (height + 20)*100;
    return sealevel_temperature - 0.0065*height;
}

// absolute pressure calculation based on Eq.28 Stormscape
double absolute_pres(double height)
{
    float tmp = 1 - 0.65*height/(sealevel_temperature+273.15);
    return sealevel_pressure*std::pow(tmp, 5.2561);
}

 // Stormscape Eq.9
double mole_fraction(double ratio)
{
    if (ratio < 0) ratio = 0; // fail safe
    return ratio / (1 + ratio);
}

 // Stormscape Eq.7
double avg_mole_mass(double ratio)
{
    float real_mass = 18.02*ratio + 28.96*(1-ratio);
//    return real_mass*mass_scale;
    return real_mass;
}

// Stormscape Eq.11
double isentropic_exponent(double ratio)
{
    return ratio*1.33 + (1-ratio)*1.4;
}

double heat_capacity(double gamma, double mass)
{
    return gamma*8.3/(mass*(gamma-1));
}
