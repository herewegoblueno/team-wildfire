
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
__global__ void jacobiOptimized(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj)
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
