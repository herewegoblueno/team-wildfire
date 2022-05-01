#include<stdio.h>
#include<stdlib.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <iostream>
#include <chrono>

std::chrono::time_point<std::chrono::high_resolution_clock> now() {
    return std::chrono::high_resolution_clock::now();
}

template <typename T>
double milliseconds(T t) {
    return (double) std::chrono::duration_cast<std::chrono::nanoseconds>(t).count() / 1000000;
}

// Device version of the Jacobi method
__global__ void jacobi(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj)
{
    float sigma = 0.0;
    int idx = threadIdx.x;
    for (int j=0; j<Nj; j++)
    {
        if (idx != j)
            sigma += A[idx*Nj + j] * x_now[j];
    }
    x_next[idx] = (b[idx] - sigma) / A[idx*Nj + idx];
}


// Optimized device version of the Jacobi method
__global__ void jacobiOptimized(double* x_next, double* A, double* x_now, double* b, int Ni, int Nj)
{
    // Optimization step 1: tiling
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx < Ni)
    {
        float sigma = 0.0;

        // Optimization step 2: store index in register
        // Multiplication is not executed in every iteration.
        int idx_Ai = idx*Nj;

        for (int j=0; j<Nj; j++)
            if (idx != j)
                sigma += A[idx_Ai + j] * x_now[j];

        x_next[idx] = (b[idx] - sigma) / A[idx_Ai + idx];
    }
}

extern "C" void jacobiGPU(double* A_mat, double* dvg, int N, int Ni, int iter)
{

    auto t1 = now();

    double *x_next_d, *A_d, *x_now_d, *d_d;
    // Allocate memory on the device
    assert(cudaSuccess == cudaMalloc((void **) &x_next_d, Ni*sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &A_d, N*sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &x_now_d, Ni*sizeof(double)));
    assert(cudaSuccess == cudaMalloc((void **) &d_d, Ni*sizeof(double)));

    double* zero = (double*)malloc(Ni*sizeof(double));
    memset(zero, 0, Ni*sizeof(double));
    // Copy data -> device
    cudaMemcpy(A_d, A_mat, sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, dvg, sizeof(double)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(x_next_d, zero, sizeof(double)*Ni, cudaMemcpyHostToDevice);
    cudaMemcpy(x_now_d, zero, sizeof(double)*Ni, cudaMemcpyHostToDevice);

    // Compute grid and block size.
    int kernel=2, tileSize=4;
    int nTiles = Ni/tileSize + (Ni%tileSize == 0?0:1);
    int gridHeight = Ni/tileSize + (Ni%tileSize == 0?0:1);
    int gridWidth = Ni/tileSize + (Ni%tileSize == 0?0:1);
    printf("w=%d, h=%d\n",gridWidth,gridHeight);
    dim3 dGrid(gridHeight, gridWidth),
        dBlock(tileSize, tileSize);

    for (int k=0; k<iter; k++)
    {
        if (k%2)
            jacobiOptimized <<< nTiles, tileSize >>> (x_now_d, A_d, x_next_d, d_d, Ni, Ni);
        else
            jacobiOptimized <<< nTiles, tileSize >>> (x_next_d, A_d, x_now_d, d_d, Ni, Ni);
        //cudaMemcpy(x_now_d, x_next_d, sizeof(float)*Ni, cudaMemcpyDeviceToDevice);
    }
    cudaFree(x_next_d); cudaFree(A_d); cudaFree(x_now_d); cudaFree(d_d);

    cudaMemcpy(dvg, x_next_d, sizeof(float)*Ni, cudaMemcpyDeviceToHost);

    auto t2 = now();

    std::cout << "[Jacobi Iteration]\n";
    std::cout << "Including loading, total of " << milliseconds(t2 - t1) << " milliseconds ";
    std::cout << "on " << iter << " iterations\n";
    std::cout << std::flush;
}
