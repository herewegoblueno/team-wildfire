

__global__
void windKernel(double* grid_temp, double* grid_q_v, double* grid_h) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    grid_q_v[index] = grid_q_v[index]/(1+grid_q_v[index]);
    grid_q_v[index] = 18.02*grid_q_v[index] + 28.96*(1-grid_q_v[index]);
    grid_h[index] = 7-0.0065*grid_h[index];
    grid_temp[index] = -0.1*(28.96*grid_temp[index]/grid_q_v[index]/grid_h[index] - 1);
}


extern "C"
void processWindGPU(double* grid_temp, double* grid_q_v, double* grid_h, int resolution) {

    double* cuda_temp;
    double* cuda_q_V;
    double* cuda_h;
    cudaMalloc(&cuda_temp, resolution * sizeof(double));
    cudaMalloc(&cuda_q_V, resolution * sizeof(double));
    cudaMalloc(&cuda_h, resolution * sizeof(double));
    cudaMemcpy(cuda_temp, grid_temp, resolution * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_q_V, grid_q_v, resolution * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_h, grid_h, resolution * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int blockSize = deviceProp.maxThreadsPerBlock;
    int numBlocks = (resolution - 1) / blockSize + 1;

    windKernel<<<numBlocks, blockSize>>>(cuda_temp, cuda_q_V, cuda_h);
    cudaDeviceSynchronize();


    cudaMemcpy(grid_temp, cuda_temp, resolution * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(cuda_temp);
    cudaFree(cuda_q_V);


}
