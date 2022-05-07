#include<stdio.h>
#include<stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <memory>
//#include "info.cuh"



std::chrono::time_point<std::chrono::high_resolution_clock> now() {
    return std::chrono::high_resolution_clock::now();
}

template <typename T>
double milliseconds(T t) {
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
}

__device__ double getVel(int x, int y, int z, double* u, int resolution, int dim);
__device__ double safe_get(int x, int y, int z, double* u, int resolution);
__global__ void jacobi(double* x_next, double* A, double* x_now, double* b, int* xyz, int Ni, int Res, int segment);


__global__
void buoyancyKernel(double* grid_temp, double* grid_q_v, double* grid_h, double* su_xyz,
                    double* f, int resolution, double dt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < resolution*resolution*resolution && idx>-1)
    {
        double* src_u = su_xyz + idx*3;

        double T_th = 10*grid_temp[idx] + 273.15;
        double X_v = grid_q_v[idx]/(1+grid_q_v[idx]);
        double M_th = 18.02*X_v + 28.96*(1-X_v);
        double T_air = 35-grid_h[idx]*3 + 273.15;
        double buoyancy =   1*(28.96*T_th/(M_th*T_air) - 1);

        src_u[1] += buoyancy*dt;
        src_u[0] += f[0]*0.1*dt;
        src_u[1] += f[1]*0.1*dt;
        src_u[2] += f[2]*0.1*dt;
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

        ua = getVel(x-1, y, z, su_xyz, resolution, 0);
        ub = getVel(x+1, y, z, su_xyz, resolution, 0);
        dst_u[0] = src_u[0] - 0.95*(ub - ua)/2/cell_size*src_u[0]*dt;
        ua = getVel(x, y-1, z, su_xyz, resolution, 1);
        ub = getVel(x, y+1, z, su_xyz, resolution, 1);
        dst_u[1] = src_u[1] - 0.95*(ub - ua)/2/cell_size*src_u[1]*dt;
        ua = getVel(x, y, z-1, su_xyz, resolution, 2);
        ub = getVel(x, y, z+1, su_xyz, resolution, 2);
        dst_u[2] = src_u[2] - 0.95*(ub - ua)/2/cell_size*src_u[2]*dt;
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
            double u0, u1;
            u1 = getVel(x+1, y, z, su_xyz, resolution, 0);
            u0 = getVel(x-1, y, z, su_xyz, resolution, 0);
            dst_u[0] = src_u[0] + ((u1 - src_u[0]) - (src_u[0] - u0))*factor;

            u1 = getVel(x, y+1, z, su_xyz, resolution, 1);
            u0 = getVel(x, y-1, z, su_xyz, resolution, 1);
            dst_u[1] = src_u[1] + ((u1 - src_u[1]) - (src_u[1] - u0))*factor;

            u1 = getVel(x, y, z+1, su_xyz, resolution, 2);
            u0 = getVel(x, y, z-1, su_xyz, resolution, 2);
            dst_u[2] = src_u[2] + ((u1 - src_u[2]) - (src_u[2] - u0))*factor;
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
//            printf("(%f,%f,%f)-[%f]",dvor_x,dvor_y,dvor_z,len);
            dst_u[0] = src_u[0] + (dvor_y*vor_z - dvor_z*vor_y)*cell_size*dt*0.1;
            dst_u[1] = src_u[1] + (dvor_z*vor_x - dvor_x*vor_z)*cell_size*dt*0.1;
            dst_u[2] = src_u[2] + (dvor_x*vor_y - dvor_y*vor_x)*cell_size*dt*0.1;
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
        if(x==5 && y==5 && z==5)
        {
             printf("(%f, %f, %f)=>(%f, %f, %f)\n", src_u[0], src_u[1], src_u[2], dst_u[0], dst_u[1], dst_u[2]);
        }

        if(x==0 || y==0 || z==0 || x==resolution-1 || y==resolution-1 || y==resolution-1)
        {
            dst_u[0] = 0; dst_u[1] = 0; dst_u[2] = 0;
        }
    }
}


extern "C"
void processWindGPU(double* grid_temp, double* grid_q_v, double* grid_h,
                    double* u_xyz, int* id_xyz, int jacobi_iter, double f[3],
                    int resolution, double cell_size, float dt)
{
    double air_density = 1.225;
    double viscosity = 0.2;
    dt = dt/20;
    cudaError err;


    auto t1 = now();
    int cell_num = resolution*resolution*resolution;
    double *d_temp, *d_q_v, *d_h, *d_u, *d_u2, *d_f;
    int *d_id;
    cudaMalloc(&d_temp, cell_num * sizeof(double)); // temperature
    cudaMalloc(&d_q_v,  cell_num * sizeof(double)); // q_v
    cudaMalloc(&d_h,    cell_num * sizeof(double)); // height
    cudaMalloc(&d_u,    cell_num * sizeof(double) * 3); // vel 1
    cudaMalloc(&d_u2,   cell_num * sizeof(double) * 3); // vel 2(for switching values)
    cudaMalloc(&d_id,   cell_num * sizeof(int) * 3); // temperature
    cudaMalloc(&d_f,    sizeof(double) * 3); // wind field

    cudaMemcpy(d_temp,  grid_temp, cell_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q_v,   grid_q_v,  cell_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h,     grid_h,    cell_num * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u,     u_xyz,     cell_num * sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u2,    u_xyz,     cell_num * sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_id,    id_xyz,    cell_num * sizeof(int) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f,     f,         sizeof(double) * 3, cudaMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (cell_num - 1) / blockSize + 1;
    // advection
    auto t2 = now();
    advectKernel <<<numBlocks, blockSize>>> (d_u, d_id, d_u2, resolution, cell_size, dt);
    cudaDeviceSynchronize();
//     diffusion
    auto t3 = now();
    viscosityKernel <<<numBlocks, blockSize>>> (d_u2, d_id, d_u, viscosity, resolution, cell_size, dt);
    cudaDeviceSynchronize();
    // vorticity confinement
    auto t4 = now();
//    cudaMemcpy(d_u2,    d_u,     cell_num * sizeof(double) * 3, cudaMemcpyDeviceToDevice);
    double *vorticity, *vorticity_len;
    cudaMalloc(&vorticity,      cell_num * sizeof(double) * 3);
    cudaMalloc(&vorticity_len,  cell_num * sizeof(double));
    pre_vorticityKernel <<<numBlocks, blockSize>>> (d_u, d_id, vorticity, vorticity_len, resolution, cell_size, dt);
    cudaDeviceSynchronize();
    vorticityKernel <<<numBlocks, blockSize>>> (d_u, d_id, d_u2, vorticity, vorticity_len, resolution, cell_size, dt);
    cudaDeviceSynchronize();
    cudaFree(vorticity);  cudaFree(vorticity_len);
    // buoyancy
    auto t5 = now();
    buoyancyKernel <<<numBlocks, blockSize>>> (d_temp, d_q_v, d_h, d_u2, d_f, resolution, dt);
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString( err) );
    }
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

    err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString( err) );
    }
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

    cudaFree(d_diag);
    cudaFree(x_now_d);
    cudaFree(x_next_d);
    cudaMemcpy(u_xyz, d_u, cell_num * sizeof(double) * 3, cudaMemcpyDeviceToHost);

    cudaFree(d_temp);
    cudaFree(d_q_v);
    cudaFree(d_h);
    cudaFree(d_u);
    cudaFree(d_u2);
    cudaFree(d_id);
    cudaFree(d_f);

//    cudaError err;
    err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString( err) );
    }

    printf("[param]-[density:%f]-[viscosity:%f]-[cell:%f]-", air_density, viscosity, cell_size);
    std::cout << "[Wind Update Ellapse Summary]";
    std::cout << "-[Total- " << milliseconds(t7 - t1) << "]\n";
    std::cout << "[load- " << milliseconds(t2 - t1) << "]-";
    std::cout << "[advect- " << milliseconds(t3 - t2) << "]-";
    std::cout << "[diffuse- " << milliseconds(t4 - t3) << "]-";
    std::cout << "[vorticity- " << milliseconds(t5 - t4) << "]-";
    std::cout << "[buoyancy- " << milliseconds(t6 - t5) << "]-";
    std::cout << "[pressure- " << milliseconds(t7 - t6) << "]\n";
    std::cout << std::flush;
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
//        printf(" (%d: %d, %d, %d) ", idx, id_x, id_y, id_z);
    }
}


__device__ double getVel(int x, int y, int z, double* u, int resolution, int dim)
{
    if(x<0 || y<0 || z<0 || x>resolution-2 || y>resolution-2 || z>resolution-2)
        return 0;
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
