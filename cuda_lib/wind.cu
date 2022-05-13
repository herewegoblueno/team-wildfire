#include<stdio.h>
#include<stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <memory>

#define prob_x 8
#define prob_y 6
#define prob_z 15

std::chrono::time_point<std::chrono::high_resolution_clock> now() {
    return std::chrono::high_resolution_clock::now();
}

template <typename T>
double milliseconds(T t) {
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
}

__device__ double getVel(int x, int y, int z, double* u, int resolution, int dim);
__device__ double safe_get(int x, int y, int z, double* u, int resolution);
__device__ double safe_get_zero(int x, int y, int z, double* u, int resolution);
__global__ void jacobi(double* x_next, double* A, double* x_now, double* b, int* xyz, int Ni, int Res, int segment);


__global__
void buoyancyKernel(double* grid_temp, double* grid_q_v, double* grid_h, double* su_xyz,
                    double* f, int resolution, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < resolution*resolution*resolution && idx>-1)
    {
        double* src_u = su_xyz + idx*3;
        double height = grid_h[idx];
        double X_v = grid_q_v[idx]/(1+grid_q_v[idx])*10;
        double M_th = 18.02*X_v + 28.96*(1-X_v);
        double Y_v = X_v*18.02/M_th;
        double y_th = Y_v*1.33 + (1-Y_v)*1.4;
        double T_air = 301;
        if(height < 9) T_air = 293.15 - 0.65*height;
        double p_z_r = pow(1 - 0.65*height/293.15, 5.2561);
        double T_th = 287.3*pow(p_z_r, 1 - 1/y_th);
        double buoyancy =  0.05*(28.96*T_th/(M_th*T_air)-1);

        if (buoyancy<0) buoyancy=0;
        if (buoyancy>0.1) buoyancy=0.1;
        if(src_u[1]<0.2) src_u[1] += buoyancy*dt;

        if(height<6)
        {
            if(fabs(src_u[0]) < fabs(f[0])) src_u[0] += f[0]*0.02*dt;
            if(fabs(src_u[1]) < fabs(f[1])) src_u[1] += f[1]*0.02*dt;
            if(fabs(src_u[2]) < fabs(f[2])) src_u[2] += f[2]*0.02*dt;
        }
    }
}

__global__
void advectKernel(double* su_xyz, int* id_xyz, double* tu_xyz,
                  int resolution, double cell_size, double dt)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < resolution*resolution*resolution && idx>-1)
    {
        int* this_xyz = id_xyz + idx*3;
        double* src_u = su_xyz + idx*3;
        double* dst_u = tu_xyz + idx*3;
        int x=this_xyz[0], y=this_xyz[1], z=this_xyz[2];
        double ua, ub;

        ua = getVel(x, y-1, z, su_xyz, resolution, 1);
        ub = getVel(x, y+1, z, su_xyz, resolution, 1);
        dst_u[1] = src_u[1] - 0.95*(ub*ub - ua*ua)/2/cell_size*dt;

        if(z==0) dt==0;
        ua = getVel(x-1, y, z, su_xyz, resolution, 0);
        ub = getVel(x+1, y, z, su_xyz, resolution, 0);
        dst_u[0] = src_u[0] - 0.95*(ub*ub - ua*ua)/2/cell_size*dt;
        ua = getVel(x, y, z-1, su_xyz, resolution, 2);
        ub = getVel(x, y, z+1, su_xyz, resolution, 2);
        dst_u[2] = src_u[2] - 0.95*(ub*ub - ua*ua)/2/cell_size*dt;
    }
}

__global__
void viscosityKernel(double* su_xyz, int* id_xyz, double* tu_xyz, double viscosity,
                     int resolution, double cell_size, double dt)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < resolution*resolution*resolution && idx>-1)
    {
        int* this_xyz = id_xyz + idx*3;
        double* src_u = su_xyz + idx*3;
        double* dst_u = tu_xyz + idx*3;
        int x=this_xyz[0], y=this_xyz[1], z=this_xyz[2];

        if(x<resolution-1 && y<resolution-1 && z<resolution-1)
        {
            double factor = viscosity*dt/cell_size/cell_size;
            double laplace = -src_u[0]*6;
            laplace += getVel(x+1, y, z, su_xyz, resolution, 0);
            laplace += getVel(x-1, y, z, su_xyz, resolution, 0);
            laplace += getVel(x, y+1, z, su_xyz, resolution, 0);
            laplace += getVel(x, y-1, z, su_xyz, resolution, 0);
            laplace += getVel(x, y, z+1, su_xyz, resolution, 0);
            laplace += getVel(x, y, z-1, su_xyz, resolution, 0);
            dst_u[0] = src_u[0] + laplace*factor;
            laplace = -src_u[1]*6;
            laplace += getVel(x+1, y, z, su_xyz, resolution, 1);
            laplace += getVel(x-1, y, z, su_xyz, resolution, 1);
            laplace += getVel(x, y+1, z, su_xyz, resolution, 1);
            laplace += getVel(x, y-1, z, su_xyz, resolution, 1);
            laplace += getVel(x, y, z+1, su_xyz, resolution, 1);
            laplace += getVel(x, y, z-1, su_xyz, resolution, 1);
            dst_u[1] = src_u[1] + laplace*factor;
            laplace = -src_u[2]*6;
            laplace += getVel(x+1, y, z, su_xyz, resolution, 2);
            laplace += getVel(x-1, y, z, su_xyz, resolution, 2);
            laplace += getVel(x, y+1, z, su_xyz, resolution, 2);
            laplace += getVel(x, y-1, z, su_xyz, resolution, 2);
            laplace += getVel(x, y, z+1, su_xyz, resolution, 2);
            laplace += getVel(x, y, z-1, su_xyz, resolution, 2);
            dst_u[2] = src_u[2] + laplace*factor;
        }
        else
        {
            dst_u[0] = 0; dst_u[1] = 0; dst_u[2] = 0;
        }
    }
}

__global__
void pre_vorticityKernel(double* su_xyz, int* id_xyz, double* vorticity, double* vorticity_len,
                         int resolution, double cell_size, double dt)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < resolution*resolution*resolution && idx>-1)
    {
        int* this_xyz = id_xyz + idx*3;
        double* dst_v = vorticity + idx*3;
        double* dst_vL = vorticity_len + idx;

        int x=this_xyz[0], y=this_xyz[1], z=this_xyz[2];

        double tmp, len = 0;
        double grad0, grad1;
        grad0 = (getVel(x, y, z+1, su_xyz, resolution, 1) + getVel(x, y-1, z+1, su_xyz, resolution, 1) -
                 getVel(x, y, z-1, su_xyz, resolution, 1) - getVel(x, y-1, z-1, su_xyz, resolution, 1)); // grad_uy.z
        grad1 = (getVel(x, y+1, z, su_xyz, resolution, 2) + getVel(x, y+1, z-1, su_xyz, resolution, 2) -
                 getVel(x, y-1, z, su_xyz, resolution, 2) - getVel(x, y-1, z-1, su_xyz, resolution, 2)); // grad_uz.y
        tmp = (grad1 - grad0)/4/cell_size;
        dst_v[0] = tmp;
        len += tmp*tmp;

        grad0 = (getVel(x+1, y, z, su_xyz, resolution, 2) + getVel(x+1, y, z-1, su_xyz, resolution, 2) -
                 getVel(x-1, y, z, su_xyz, resolution, 2) - getVel(x-1, y, z-1, su_xyz, resolution, 2)); // grad_uz.x
        grad1 = (getVel(x, y, z+1, su_xyz, resolution, 0) + getVel(x-1, y, z+1, su_xyz, resolution, 0) -
                 getVel(x, y, z-1, su_xyz, resolution, 0) - getVel(x-1, y, z-1, su_xyz, resolution, 0)); // grad_ux.z
        tmp = (grad1 - grad0)/4/cell_size;
        dst_v[1] = tmp;
        len += tmp*tmp;

        grad0 = (getVel(x, y+1, z, su_xyz, resolution, 0) + getVel(x-1, y+1, z, su_xyz, resolution, 0) -
                 getVel(x, y-1, z, su_xyz, resolution, 0) - getVel(x-1, y-1, z, su_xyz, resolution, 0)); // grad_ux.y
        grad1 = (getVel(x+1, y, z, su_xyz, resolution, 1) + getVel(x+1, y-1, z, su_xyz, resolution, 1) -
                 getVel(x-1, y, z, su_xyz, resolution, 1) - getVel(x-1, y-1, z, su_xyz, resolution, 1)); // grad_uy.x
        tmp = (grad1 - grad0)/4/cell_size;
        dst_v[2] = tmp;
        len += tmp*tmp;

        dst_vL[0] = sqrt(len);
    }
}

__global__
void vorticityKernel(double* su_xyz, int* id_xyz, double* tu_xyz, double* vorticity, double* vorticity_len,
                     int resolution, double cell_size, double dt)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < resolution*resolution*resolution && idx>-1)
    {
        int* this_xyz = id_xyz + idx*3;
        double* src_u = su_xyz + idx*3;
        double* dst_u = tu_xyz + idx*3;
        int x=this_xyz[0], y=this_xyz[1], z=this_xyz[2];

        double vor_x = *(vorticity + idx*3);
        double vor_y = *(vorticity + idx*3 + 1);
        double vor_z = *(vorticity + idx*3 + 2);
        double dvor_x = (safe_get(x+1, y, z, vorticity_len, resolution) -
                         safe_get(x-1, y, z, vorticity_len, resolution))/2/cell_size;
        double dvor_y = (safe_get(x, y+1, z, vorticity_len, resolution) -
                         safe_get(x, y-1, z, vorticity_len, resolution))/2/cell_size;
        double dvor_z = (safe_get(x, y, z+1, vorticity_len, resolution) -
                         safe_get(x, y, z-1, vorticity_len, resolution))/2/cell_size;
        double len = sqrt(dvor_x*dvor_x+dvor_y*dvor_y+dvor_z*dvor_z);
        if(len>0)
        {
            dvor_x /= len; dvor_y /= len; dvor_z /= len;

            dst_u[0] = src_u[0] + (dvor_y*vor_z - dvor_z*vor_y)*cell_size*dt*0.001;
            dst_u[1] = src_u[1] + (dvor_z*vor_x - dvor_x*vor_z)*cell_size*dt*0.001;
            dst_u[2] = src_u[2] + (dvor_x*vor_y - dvor_y*vor_x)*cell_size*dt*0.001;
        }
        else
        {
            dst_u[0] = src_u[0];
            dst_u[1] = src_u[1];
            dst_u[2] = src_u[2];
        }
    }
}

__global__
void pre_JacobiKernel(double* su_xyz, int* id_xyz, double density_term, int resolution, double cell_size,
                     double* diag_A, double* rhs)
{

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < resolution*resolution*resolution && idx>-1)
    {
        int* this_xyz = id_xyz + idx*3;
        int x=this_xyz[0], y=this_xyz[1], z=this_xyz[2];
        double ua, ub;
        double this_rhs = 0;

        ua = getVel(x-1, y, z, su_xyz, resolution, 0);
        ub = getVel(x,   y, z, su_xyz, resolution, 0);
        this_rhs -= (ub - ua)/cell_size;
        ua = getVel(x, y-1, z, su_xyz, resolution, 1);
        ub = getVel(x, y,   z, su_xyz, resolution, 1);
        this_rhs -= (ub - ua)/cell_size;
        ua = getVel(x, y, z-1, su_xyz, resolution, 2);
        ub = getVel(x, y, z,   su_xyz, resolution, 2);
        this_rhs -= (ub - ua)/cell_size;
        rhs[idx] = this_rhs/density_term;

        double diag = 6;
        if(x>resolution-2) diag --;
        if(x<1) diag --;
        if(y>resolution-2) diag --;
        if(y<1) diag --;
        if(z>resolution-2) diag --;
        if(z<1) diag --;

        diag_A[idx] = diag;
    }
}

// Optimized device version of the Jacobi method
__global__ void jacobi(double* x_next, double* A, double* x_now, double* b, int* xyz, int Ni, int Res, int segment)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < Ni && idx>-1)
    {
        double sigma = 0.0;
        int id_x = xyz[idx*3];
        int id_y = xyz[idx*3+1];
        int id_z = xyz[idx*3+2];

        if(id_x > 0)     sigma += x_now[idx - Res*Res];
        if(id_x < Res-1) sigma += x_now[idx + Res*Res];
        if(id_y > 0)     sigma += x_now[idx - Res];
        if(id_y < Res-1) sigma += x_now[idx + Res];
        if(id_z > 0)     sigma += x_now[idx - 1];
        if(id_z < Res-1) sigma += x_now[idx + 1];

        x_next[idx] = (b[idx] + sigma) / A[idx];
    }
}


__global__
void pressureKernel(double* su_xyz, int* id_xyz, double* tu_xyz, double* pressure, double density,
                    int resolution, double cell_size, double dt)
{

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < resolution*resolution*resolution && idx>-1)
    {
        int* this_xyz = id_xyz + idx*3;
        double* src_u = su_xyz + idx*3;
        double* dst_u = tu_xyz + idx*3;
        int x=this_xyz[0], y=this_xyz[1], z=this_xyz[2];
        double dP;

        if(x<resolution-1) dP = safe_get(x+1,y,z,pressure,resolution) - safe_get(x,y,z,pressure,resolution);
        else dP = 0;
        dst_u[0] = src_u[0] - dP*dt/(cell_size*density);
        if(y<resolution-1) dP = safe_get(x,y+1,z,pressure,resolution) - safe_get(x,y,z,pressure,resolution);
        else dP = 0;
        dst_u[1] = src_u[1] - dP*dt/(cell_size*density);
        if(z<resolution-1) dP = safe_get(x,y,z+1,pressure,resolution) - safe_get(x,y,z,pressure,resolution);
        else dP = 0;
        dst_u[2] = src_u[2] - dP*dt/(cell_size*density);

//        if(fabs(dst_u[0])>0.3) dst_u[0] = dst_u[0]/fabs(dst_u[0])*0.3;
//        if(fabs(dst_u[1])>0.3) dst_u[1] = dst_u[1]/fabs(dst_u[1])*0.3;
//        if(fabs(dst_u[2])>0.3) dst_u[2] = dst_u[2]/fabs(dst_u[2])*0.3;

        if(x==resolution-1 || x==0) dst_u[0] = 0;
        if(y==resolution-1) dst_u[1] = 0;
        if(z==resolution-1 || z==0) dst_u[2] = 0;
        if(y==0)
        {
            dst_u[1]=0;
            dst_u[0]=0;
            dst_u[2]=0;
        }

    }
}

__device__
float getGrad(int x, int y, int z, int dim, int resolution, double cell_size, double* vals)
{
    if(dim==0)
        return (safe_get_zero(x+1,y,z,vals,resolution) - safe_get_zero(x-1,y,z,vals,resolution))/2/cell_size;
    if(dim==1)
        return (safe_get_zero(x,y+1,z,vals,resolution) - safe_get_zero(x,y-1,z,vals,resolution))/2/cell_size;
    if(dim==2)
        return (safe_get_zero(x,y,z+1,vals,resolution) - safe_get_zero(x,y,z-1,vals,resolution))/2/cell_size;
}

__global__
void waterKernel(double* u_xyz, int* id_xyz,  double* d_h, double* d_temp, double* d_humi,
                 double* d_q_v, double*d_q_c, double*d_q_r,
                 double* d_q_v2, double*d_q_c2, double*d_q_r2,
                       int resolution, double cell_size, double dt)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < resolution*resolution*resolution && idx>-1)
    {
        int* this_xyz = id_xyz + idx*3;
        int x=this_xyz[0], y=this_xyz[1], z=this_xyz[2];
        d_q_v2[idx] = d_q_v[idx];
        d_q_c2[idx] = d_q_c[idx];
        d_q_r2[idx] = d_q_r[idx];

        float uc_x = (getVel(x,y,z,u_xyz,resolution,0)+getVel(x-1,y,z,u_xyz,resolution,0))*0.5*700;
        float uc_y = (getVel(x,y,z,u_xyz,resolution,1)+getVel(x,y-1,z,u_xyz,resolution,1))*0.5*700;
        float uc_z = (getVel(x,y,z,u_xyz,resolution,2)+getVel(x,y,z-1,u_xyz,resolution,2))*0.5*700;

        d_q_v2[idx] -= uc_x*getGrad(x,y,z,0,resolution,cell_size,d_q_v)*dt;
        d_q_v2[idx] -= uc_y*getGrad(x,y,z,1,resolution,cell_size,d_q_v)*dt;
        d_q_v2[idx] -= uc_z*getGrad(x,y,z,2,resolution,cell_size,d_q_v)*dt;

        d_q_c2[idx] -= uc_x*getGrad(x,y,z,0,resolution,cell_size,d_q_c)*dt;
        d_q_c2[idx] -= uc_y*getGrad(x,y,z,1,resolution,cell_size,d_q_c)*dt;
        d_q_c2[idx] -= uc_z*getGrad(x,y,z,2,resolution,cell_size,d_q_c)*dt;

        d_q_r2[idx] -= uc_x*getGrad(x,y,z,0,resolution,cell_size,d_q_r)*dt;
        d_q_r2[idx] -= uc_y*getGrad(x,y,z,1,resolution,cell_size,d_q_r)*dt;
        d_q_r2[idx] -= uc_z*getGrad(x,y,z,2,resolution,cell_size,d_q_r)*dt;

        d_q_v2[idx] = max(0., min(d_q_v2[idx], 1.));
        d_q_c2[idx] = max(0., min(d_q_c2[idx], 1.));
        d_q_r2[idx] = max(0., min(d_q_r2[idx], 1.));

        double X_v = d_q_v2[idx]/(1+d_q_v2[idx]);
        double M_th = 18.02*X_v + 28.96*(1-X_v);
        double Y_v = X_v*18.02/M_th;
        double gamma_th = Y_v*1.33 + (1-Y_v)*1.4;
        double c_th_p = gamma_th*8.3/(M_th*(gamma_th-1));

        double ambient_temperature = 20 - 0.65*d_h[idx];
        double temperature = 20 + 30*d_temp[idx];
        double abs_pres = 100000*pow(1 - 0.65*d_h[idx]/293.75, 5.2561);
        double q_vs = 380.16/abs_pres*exp(17.67*temperature/(temperature+243.5));
        double E_r = d_q_r2[idx]*0.0001*min(max(q_vs - d_q_v2[idx], 0.), 10.);// evaporation of rain Fire Eq.22
        double A_c = 0.01*max(d_q_c2[idx] - 0.01, 0.); // below Stormscape Eq.24
        double K_c = 0.01*d_q_c2[idx]*d_q_r2[idx];  // below Stormscape Eq.24
        double saturate_cmp = min(q_vs - d_q_v2[idx], d_q_c2[idx]);
        d_q_v2[idx] = d_q_v2[idx] + saturate_cmp + E_r;
        d_q_c2[idx] = d_q_c2[idx] - saturate_cmp - A_c - K_c;
        d_q_r2[idx] = A_c + K_c - E_r;

//        double q_vs_amb = 380.16/abs_pres*exp(17.67*ambient_temperature/(ambient_temperature+243.5));
//        d_humi[idx] = min(d_q_v2[idx],0.093)/10/q_vs_amb;
        d_humi[idx] = d_q_v2[idx]/q_vs;

        if(saturate_cmp<0)
        {
            float evp_temp = 2.5/c_th_p/0.287*(-saturate_cmp)/(1-saturate_cmp);
            d_temp[idx] += evp_temp;
        }
    }
}


extern "C"
void processWindGPU(double* grid_temp, double* grid_q_v, double* grid_q_c, double* grid_q_r,
                    double* grid_h, double* grid_humidity,
                    double* u_xyz, int* id_xyz, int jacobi_iter, double f[3],
                    int resolution, double cell_size, float dt)
{
    double air_density = 1.225;
    double viscosity = 0.1;
    dt = dt/2;
    cudaError err;


    auto t1 = now();
    int cell_num = resolution*resolution*resolution;
    double *d_temp, *d_q_v, *d_q_c, *d_q_r, *d_h, *d_humi, *d_u, *d_u2, *d_f;
    double *d_q_v2, *d_q_c2, *d_q_r2;
    int *d_id;
    cudaMalloc(&d_temp, cell_num * sizeof(double)); // temperature
    cudaMalloc(&d_q_v,  cell_num * sizeof(double)); // q_v
    cudaMalloc(&d_q_c,  cell_num * sizeof(double)); // q_c
    cudaMalloc(&d_q_r,  cell_num * sizeof(double)); // q_r
    cudaMalloc(&d_h,    cell_num * sizeof(double)); // height
    cudaMalloc(&d_humi,    cell_num * sizeof(double)); // humidity
    cudaMalloc(&d_u,    cell_num * sizeof(double) * 3); // vel 1
    cudaMalloc(&d_u2,   cell_num * sizeof(double) * 3); // vel 2(for switching values)
    cudaMalloc(&d_id,   cell_num * sizeof(int) * 3); // temperature
    cudaMalloc(&d_f,    sizeof(double) * 3); // wind field

    cudaMemcpy(d_temp,  grid_temp, cell_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_v,   grid_q_v,  cell_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_c,   grid_q_c,  cell_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_r,   grid_q_r,  cell_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h,     grid_h,    cell_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u,     u_xyz,     cell_num * sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2,    u_xyz,     cell_num * sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_id,    id_xyz,    cell_num * sizeof(int) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f,     f,         sizeof(double) * 3, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (cell_num - 1) / blockSize + 1;
    // advection
    auto t2 = now();
//    advectKernel <<<numBlocks, blockSize>>> (d_u, d_id, d_u2, resolution, cell_size, dt);
//    cudaDeviceSynchronize();
//     diffusion
    auto t3 = now();
//    viscosityKernel <<<numBlocks, blockSize>>> (d_u2, d_id, d_u, viscosity, resolution, cell_size, dt);
//    cudaDeviceSynchronize();
    // vorticity confinement
    auto t4 = now();
    cudaMemcpy(d_u,    d_u2,     cell_num * sizeof(double) * 3, cudaMemcpyDeviceToDevice);
//    double *vorticity, *vorticity_len;
//    cudaMalloc(&vorticity,      cell_num * sizeof(double) * 3);
//    cudaMalloc(&vorticity_len,  cell_num * sizeof(double));
//    pre_vorticityKernel <<<numBlocks, blockSize>>> (d_u, d_id, vorticity, vorticity_len, resolution, cell_size, dt);
//    cudaDeviceSynchronize();
//    vorticityKernel <<<numBlocks, blockSize>>> (d_u, d_id, d_u2, vorticity, vorticity_len, resolution, cell_size, dt);
//    cudaDeviceSynchronize();
//    cudaFree(vorticity);  cudaFree(vorticity_len);
    // buoyancy
    auto t5 = now();
    buoyancyKernel <<<numBlocks, blockSize>>> (d_temp, d_q_v, d_h, d_u2, d_f, resolution, dt);
    cudaDeviceSynchronize();

    // pressure projection
    auto t6 = now();
    double *d_diag, *d_rhs = d_u, *x_now_d, *x_next_d;
    cudaMalloc(&d_diag,   cell_num * sizeof(double));
    cudaMalloc(&x_now_d,  cell_num * sizeof(double));
    cudaMalloc(&x_next_d, cell_num * sizeof(double));
    cudaMemset(x_now_d,  0, cell_num * sizeof(double));
    cudaMemset(x_next_d, 0, cell_num * sizeof(double));
    //// jacoby iteration
    double density_term = dt/(air_density*cell_size*cell_size);
    pre_JacobiKernel<<<numBlocks, blockSize>>>(d_u2, d_id, density_term, resolution, cell_size, d_diag, d_rhs);
    cudaDeviceSynchronize();

    for (int k=0; k<jacobi_iter; k++)
    {
        if (k%2)
            jacobi <<<numBlocks, blockSize>>> (x_now_d, d_diag, x_next_d,
                                                 d_rhs, d_id, cell_num, resolution, 0);
        else
            jacobi <<<numBlocks, blockSize>>> (x_next_d, d_diag, x_now_d,
                                                 d_rhs, d_id, cell_num, resolution, 0);
        cudaDeviceSynchronize();
    }
    double* pressure = x_next_d;
    //// apply pressure
    pressureKernel<<<numBlocks, blockSize>>>(d_u2, d_id, d_u, pressure, air_density,
                                             resolution, cell_size, dt);
    cudaDeviceSynchronize();
    auto t7 = now();
    err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString( err) );
    }
    cudaFree(d_diag);
    cudaFree(x_now_d);
    cudaFree(x_next_d);

    cudaMalloc(&d_q_v2,  cell_num * sizeof(double)); // q_v
    cudaMalloc(&d_q_c2,  cell_num * sizeof(double)); // q_c
    cudaMalloc(&d_q_r2,  cell_num * sizeof(double)); // q_r

    waterKernel<<<numBlocks, blockSize>>>(d_u, d_id, d_h, d_temp, d_humi,
                                          d_q_v, d_q_c, d_q_r, d_q_v2, d_q_c2, d_q_r2,
                                          resolution, cell_size, dt);
    err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString( err) );
    }
    auto t8 = now();
    cudaMemcpy(u_xyz, d_u, cell_num * sizeof(double) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_q_v, d_q_v2, cell_num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_q_c, d_q_c2, cell_num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_q_r, d_q_r2, cell_num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_humidity, d_humi, cell_num * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_temp, d_temp, cell_num * sizeof(double), cudaMemcpyDeviceToHost);
    auto t9 = now();
    cudaFree(d_temp);
    cudaFree(d_q_v); cudaFree(d_q_c); cudaFree(d_q_r);
    cudaFree(d_q_v2); cudaFree(d_q_c2); cudaFree(d_q_r2);
    cudaFree(d_h);
    cudaFree(d_humi);
    cudaFree(d_u); cudaFree(d_u2);
    cudaFree(d_id);
    cudaFree(d_f);
//    printf("[param]-[density:%f]-[viscosity:%f]-[cell:%f]-", air_density, viscosity, cell_size);
    std::cout << "[Wind-Water Update Ellapse Summary]\n";
    std::cout << "[load- " << milliseconds(t2 - t1) << "]-";
    std::cout << "[Wind- " << milliseconds(t7 - t2) << "]";
    std::cout << "[Water- " << milliseconds(t8 - t7) << "]";
    std::cout << "[back- " << milliseconds(t9 - t7) << "]";
//    std::cout << "[advect- " << milliseconds(t3 - t2) << "]-";
//    std::cout << "[diffuse- " << milliseconds(t4 - t3) << "]-";
//    std::cout << "[vorticity- " << milliseconds(t5 - t4) << "]-";
//    std::cout << "[buoyancy- " << milliseconds(t6 - t5) << "]-";
//    std::cout << "[pressure- " << milliseconds(t7 - t6) << "]";
    std::cout << "\n";
    std::cout << std::flush;
}



__device__ double getVel(int x, int y, int z, double* u, int resolution, int dim)
{
    if(x<0 || y<0 || z<0 || x>resolution-1 || y>resolution-1 || z>resolution-1)
        return 0;
    if(dim==0 && x==resolution-1) return 0;
    if(dim==1 && y==resolution-1) return 0;
    if(dim==2 && z==resolution-1) return 0;
    int index = x*resolution*resolution + y*resolution + z;
    return u[index*3 + dim];
}

__device__ double safe_get(int x, int y, int z, double* u, int resolution)
{
    if(x<0) x=0;
    if(y<0) y=0;
    if(z<0) z=0;
    if(x>resolution-1) x=resolution-1;
    if(y>resolution-1) y=resolution-1;
    if(z>resolution-1) z=resolution-1;
    int index = x*resolution*resolution + y*resolution + z;
    return u[index];
}

__device__ double safe_get_zero(int x, int y, int z, double* u, int resolution)
{
    if(x<0 || y<0 || z<0 || x>resolution-1 || y>resolution-1 || z>resolution-1)
        return 0;
    int index = x*resolution*resolution + y*resolution + z;
    return u[index];
}
