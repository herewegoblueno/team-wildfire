#include<stdio.h>
#include<stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <iostream>
#include <chrono>
#include <memory>

std::chrono::time_point<std::chrono::high_resolution_clock> now() {
    return std::chrono::high_resolution_clock::now();
}

template <typename T>
double milliseconds(T t) {
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
}


// Optimized device version of the Jacobi method
__global__ void jacobi(double* x_next, double* A, double* x_now, double* b, int* xyz, int Ni, int Res, int segment)
{
    // Optimization step 1: tiling
    int idx = blockIdx.x*blockDim.x + threadIdx.x + segment;

    if (idx < Ni && idx>segment-1)
    {
        assert(idx>0);
        assert(idx<Ni);
        double sigma = 0.0;
        // Optimization step 2: store index in register
        // Multiplication is not executed in every iteration.
        int id_x = xyz[idx*3];
        int id_y = xyz[idx*3+1];
        int id_z = xyz[idx*3+2];

        // for water boundary assume boundary pressure is always the same
        if(id_x > 0)   x_next[idx] += x_now[idx - Res*Res];
        else x_next[idx] += x_now[idx];
        if(id_x < Res-1) x_next[idx] += x_now[idx + Res*Res];
        else x_next[idx] += x_now[idx];
        if(id_z > 0)   x_next[idx] += x_now[idx - 1];
        else x_next[idx] += x_now[idx];
        if(id_z < Res-1) x_next[idx] += x_now[idx + 1];
        else x_next[idx] += x_now[idx];

        // bottom wall, top air
        if(id_y > 0)   x_next[idx] += x_now[idx - Res];
        if(id_y < Res-1) x_next[idx] += x_now[idx + Res];

        x_next[idx] = (b[idx] - sigma) / (A[idx]+0.000001);
//        printf(" (%d: %d, %d, %d) ", idx, id_x, id_y, id_z);
    }
}

extern "C" void jacobiGPU(double* diag, double* rhs, int* id_xyz, int Res, int Ni, int iter)
{

    auto t1 = now();

    double* test_space;

    double *x_next_d, *diag_d, *x_now_d, *rhs_d;
    int* xyz_d;
    // Allocate memory on the device
    /*  Allocate Data  */

    cudaMalloc((void **) &x_next_d, Ni*sizeof(double));
    cudaMalloc((void **) &diag_d, Ni*sizeof(double))  ;
    cudaMalloc((void **) &x_now_d, Ni*sizeof(double)) ;
    cudaMalloc((void **) &rhs_d, Ni*sizeof(double))   ;
    cudaMalloc((void **) &xyz_d, Ni*sizeof(int))      ;


    // Copy data -> device
    cudaMemcpy(diag_d, diag, sizeof(double)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_d, rhs, sizeof(double)*Ni, cudaMemcpyHostToDevice);
    cudaMemset(x_next_d, 0, sizeof(double)*Ni);
    cudaMemset(x_now_d, 0, sizeof(double)*Ni);
    cudaMemcpy(xyz_d, id_xyz, sizeof(int)*Ni, cudaMemcpyHostToDevice);


    auto t2 = now();


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = 512;
    int numBlocks = 100;
    int totalThreads = blockSize*numBlocks;
    int segments = (Ni - 1)/totalThreads + 1;

    std::cout << "Threads per block:" << blockSize <<"\n Number of block:"  << numBlocks << "\n";
    for (int k=0; k<iter; k++)
    {
        for (int s=0;s<segments;s++)
        {
            int curent_max = (s+1)*totalThreads;
            if(curent_max>Ni) curent_max = Ni;

            int array_offset = s*totalThreads;
            if(array_offset>0)
            if (k%2)
                jacobi <<< numBlocks, blockSize >>> (x_now_d, diag_d, x_next_d,
                                                     rhs_d, xyz_d, curent_max, Res, s*totalThreads);
            else
                jacobi <<< numBlocks, blockSize >>> (x_next_d, diag_d, x_now_d,
                                                     rhs_d, xyz_d, curent_max, Res, s*totalThreads);
            cudaDeviceSynchronize();
        }

    }


    cudaError err;
    err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf( stderr, "Cuda error in file '%s' in line %i : %s.\n",
                 __FILE__, __LINE__, cudaGetErrorString( err) );
    }


    cudaMemcpy(rhs, x_next_d, sizeof(double)*Ni, cudaMemcpyDeviceToHost);
    cudaFree(x_next_d); cudaFree(diag_d); cudaFree(x_now_d); cudaFree(rhs_d); cudaFree(xyz_d);

    auto t3 = now();

    std::cout << "[Jacobi Iteration]\n";
    std::cout << "Loading cost " << milliseconds(t2 - t1) << " milliseconds\n";
    std::cout << iter << " jacobi iterations cost " << milliseconds(t3 - t2) << " milliseconds\n";
    std::cout << std::flush;
}

////// for debug cuda mem check

//int main()
//{
//    int Res = 80;
//    int Ni = Res*Res*Res;
//    double* diag = (double*) malloc(sizeof(double)*Ni);
//    double* rhs = (double*) malloc(sizeof(double)*Ni);
//    int* id_xyz = (int*) malloc(3*sizeof(int)*Ni);

//    jacobiGPU(diag, rhs, id_xyz, Res, Ni, 20);
//    return 0;
//}
