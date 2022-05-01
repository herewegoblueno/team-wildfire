#include "simulator.h"
#include <math.h>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <iostream>
//#include <cuda_runtime.h>

extern "C" void jacobiOptimized(float* x_next, float* A, float* x_now, float* b, int Ni, int Nj);

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

void Simulator::pressure_projection_PCG(VoxelGrid *grid, double time)
{
//    int cell_num = A.
}

void Simulator::pressure_projection_Jacobi_cuda(VoxelGrid *grid, double time)
{
//    int resolution = grid->getResolution();
//    int cell_num = resolution*resolution*resolution;
//    int face_num = resolution*resolution;
//    double cell_size = grid->cellSideLength();
//    double density_term = time/1/cell_size/cell_size;

//    double *x_now, *x_next, *A, *d, *x_h, *x_d;
//    double *x_now_d, *x_next_d, *A_d, *d_d;

//    int N = cell_num*cell_num, Ni = cell_num, Nj = cell_num, iter=20, tileSize = 32;
//    int i,j,k;

//    x_next = (double *) malloc(Ni*sizeof(double));
//    A = (double *) malloc(N*sizeof(double));
//    x_now = (double *) malloc(Ni*sizeof(double));
//    d = (double *) malloc(Ni*sizeof(double));
//    x_h = (double *) malloc(Ni*sizeof(double));
//    x_d = (double *) malloc(Ni*sizeof(double));

//    memset(x_now, 0, Ni*sizeof(double));
//    memset(x_next, 0, N*sizeof(double));
//    memset(A, 0, Ni*sizeof(double));

//    for(i=0; i<resolution;i++)
//        for(j=0; j<resolution;j++)
//            for(k=0; k<resolution;k++)
//            {
//                int index = i*face_num+j*resolution+k;
//                double diag = 6;

//                if(i<resolution-1) A[index*cell_num + index+face_num] = -1;
//                else diag --;
//                if(i>0) A[index*cell_num +  index-face_num] = -1;
//                else diag --;
//                if(j<resolution-1) A[index*cell_num +  index+resolution] = -1;
//                else diag --;
//                if(j>0) A[index*cell_num + index-resolution] = -1;
//                else diag --;
//                if(k<resolution-1) A[index*cell_num + index+1] = -1;
//                else diag --;
//                if(k>0) A[index*cell_num + index-1] = -1;
//                else diag --;

//                A[index*cell_num + index] = diag;

//                glm::dvec3 gradientX = grid->getVoxel(i,j,k)->getGradient(get_ux);
//                glm::dvec3 gradientY = grid->getVoxel(i,j,k)->getGradient(get_uy);
//                glm::dvec3 gradientZ = grid->getVoxel(i,j,k)->getGradient(get_uz);
//                d[index] = (gradientX.x + gradientY.y + gradientZ.z)/density_term;
//            }


//    // Allocate memory on the device
//    assert(cudaSuccess == cudaMalloc((void **) &x_next_d, Ni*sizeof(double)));
//    assert(cudaSuccess == cudaMalloc((void **) &A_d, N*sizeof(double)));
//    assert(cudaSuccess == cudaMalloc((void **) &x_now_d, Ni*sizeof(double)));
//    assert(cudaSuccess == cudaMalloc((void **) &d_d, Ni*sizeof(double)));

//    // Copy data -> device
//    cudaMemcpy(x_next_d, x_next, sizeof(double)*Ni, cudaMemcpyHostToDevice);
//    cudaMemcpy(A_d, A, sizeof(double)*N, cudaMemcpyHostToDevice);
//    cudaMemcpy(x_now_d, x_now, sizeof(double)*Ni, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_d, d, sizeof(double)*Ni, cudaMemcpyHostToDevice);

//    // Compute grid and block size.
//    int nTiles = Ni/tileSize + (Ni%tileSize == 0?0:1);
//    int gridHeight = Nj/tileSize + (Nj%tileSize == 0?0:1);
//    int gridWidth = Ni/tileSize + (Ni%tileSize == 0?0:1);
//    printf("w=%d, h=%d\n",gridWidth,gridHeight);
//    dim3 dGrid(gridHeight, gridWidth),
//        dBlock(tileSize, tileSize);


//    // Run "iter" iterations of the Jacobi method on DEVICE

//    for (k=0; k<iter; k++)
//    {
//        if (k%2)
//            jacobiOptimized <<< nTiles, tileSize >>> (x_now_d, A_d, x_next_d, d_d, Ni, Nj);
//        else
//            jacobiOptimized <<< nTiles, tileSize >>> (x_next_d, A_d, x_now_d, d_d, Ni, Nj);
//        //cudaMemcpy(x_now_d, x_next_d, sizeof(float)*Ni, cudaMemcpyDeviceToDevice);
//    }


//    cudaMemcpy(x_d, x_next_d, sizeof(float)*Ni, cudaMemcpyDeviceToHost);

//    // Free memory
//    free(x_next); free(A); free(x_now); free(d);
//    cudaFree(x_next_d); cudaFree(A_d); cudaFree(x_now_d); cudaFree(d_d);

//    for(i=0; i<resolution;i++)
//        for(j=0; j<resolution;j++)
//            for(k=0; k<resolution;k++)
//            {
//                glm::dvec3 deltaP(0,0,0);
//                int index = i*face_num+j*resolution+k;
//                if(i<resolution-1) deltaP.x += x_d[index+face_num];
//                else deltaP.x += x_d[index];
//                if(i>0) deltaP.x -= x_d[index-face_num];
//                else deltaP.x -= x_d[index];
//                if(j<resolution-1) deltaP.y += x_d[index+resolution];
//                else deltaP.y += x_d[index];
//                if(j>0) deltaP.y -= x_d[index-resolution];
//                else deltaP.y -= x_d[index];
//                if(k<resolution-1) deltaP.z += x_d[index+1];
//                else deltaP.z += x_d[index];
//                if(k>0) deltaP.z -= x_d[index-1];
//                else deltaP.z -= x_d[index];
//                grid->getVoxel(i,j,k)->getCurrentState()->u -= deltaP*time;
//            }

}

void Simulator::pressure_projection_Jacobi(VoxelGrid *grid, double time)
{
    int resolution = grid->getResolution();
    int cell_num = resolution*resolution*resolution;
    int face_num = resolution*resolution;
    double cell_size = grid->cellSideLength();
    double density_term = time/1/cell_size/cell_size;

    Eigen::SparseMatrix<double> A(cell_num, cell_num);
    Eigen::SparseMatrix<double> LU(cell_num, cell_num);
    Eigen::VectorXd d = Eigen::VectorXd(cell_num,1);
    for(int i=0; i<resolution;i++)
    {
        for(int j=0; j<resolution;j++)
        {
            for(int k=0; k<resolution;k++)
            {
                int index = i*face_num+j*resolution+k;
                A.insert(index, index) = 6;

                if(i<resolution-1) LU.insert(index, index+face_num) = -1;
                else A.coeffRef(index, index) --;
                if(i>0) LU.insert(index, index-face_num) = -1;
                else A.coeffRef(index, index) --;
                if(j<resolution-1) LU.insert(index, index+resolution) = -1;
                else A.coeffRef(index, index) --;
                if(j>0) LU.insert(index, index-resolution) = -1;
                else A.coeffRef(index, index) --;
                if(k<resolution-1) LU.insert(index, index+1) = -1;
                else A.coeffRef(index, index) --;
                if(k>0) LU.insert(index, index-1) = -1;
                else A.coeffRef(index, index) --;

                A.coeffRef(index, index) = 1./A.coeffRef(index, index); // already inverse

                glm::dvec3 gradientX = grid->getVoxel(i,j,k)->getGradient(get_ux);
                glm::dvec3 gradientY = grid->getVoxel(i,j,k)->getGradient(get_uy);
                glm::dvec3 gradientZ = grid->getVoxel(i,j,k)->getGradient(get_uz);
                d[index] = (gradientX.x + gradientY.y + gradientZ.z)/density_term;
            }
        }
    }

    Eigen::VectorXd p = Eigen::VectorXd(cell_num,1);
    p.setZero();

    for(int iter=0;iter<3;iter++)
        p = A*(LU*p + d);

    for(int i=0; i<resolution;i++)
    {
        for(int j=0; j<resolution;j++)
        {
            for(int k=0; k<resolution;k++)
            {
                glm::dvec3 deltaP(0,0,0);
                int index = i*face_num+j*resolution+k;
                if(i<resolution-1) deltaP.x += p[index+face_num];
                else deltaP.x += p[index];
                if(i>0) deltaP.x -= p[index-face_num];
                else deltaP.x -= p[index];
                if(j<resolution-1) deltaP.y += p[index+resolution];
                else deltaP.y += p[index];
                if(j>0) deltaP.y -= p[index-resolution];
                else deltaP.y -= p[index];
                if(k<resolution-1) deltaP.z += p[index+1];
                else deltaP.z += p[index];
                if(k>0) deltaP.z -= p[index-1];
                else deltaP.z -= p[index];
                grid->getVoxel(i,j,k)->getCurrentState()->u -= deltaP*time;
            }
        }
    }
}


// pressure projection based on Robert Bridson [2007]
void Simulator::pressure_projection_LLT(VoxelGrid *grid, double time)
{
    int resolution = grid->getResolution();
    int cell_num = resolution*resolution*resolution;
    int face_num = resolution*resolution;
    double cell_size = grid->cellSideLength();
    double density_term = time/1/cell_size/cell_size;

    Eigen::SparseMatrix<double> A(cell_num, cell_num);
    Eigen::VectorXd d = Eigen::VectorXd(cell_num,1);
    for(int i=0; i<resolution;i++)
    {
        for(int j=0; j<resolution;j++)
        {
            for(int k=0; k<resolution;k++)
            {
                int index = i*face_num+j*resolution+k;
                A.insert(index, index) = 6;

                if(i<resolution-1) A.insert(index, index+face_num) = -1;
                else A.coeffRef(index, index) --;
                if(i>0) A.insert(index, index-face_num) = -1;
                else A.coeffRef(index, index) --;
                if(j<resolution-1)A.insert(index, index+resolution) = -1;
                else A.coeffRef(index, index) --;
                if(j>0)A.insert(index, index-resolution) = -1;
                else A.coeffRef(index, index) --;
                if(k<resolution-1)A.insert(index, index+1) = -1;
                else A.coeffRef(index, index) --;
                if(k>0)A.insert(index, index-1) = -1;
                else A.coeffRef(index, index) --;

                glm::dvec3 gradientX = grid->getVoxel(i,j,k)->getGradient(get_ux);
                glm::dvec3 gradientY = grid->getVoxel(i,j,k)->getGradient(get_uy);
                glm::dvec3 gradientZ = grid->getVoxel(i,j,k)->getGradient(get_uz);
                d[index] = (gradientX.x + gradientY.y + gradientZ.z)/density_term;
            }
        }
    }

    Eigen::SimplicialLLT <Eigen::SparseMatrix<double>> solver(A);
    solver.compute(A);

    Eigen::VectorXd p = Eigen::VectorXd(cell_num,1);
    p = solver.solve(d);

    for(int i=0; i<resolution;i++)
    {
        for(int j=0; j<resolution;j++)
        {
            for(int k=0; k<resolution;k++)
            {
                glm::dvec3 deltaP(0,0,0);
                int index = i*face_num+j*resolution+k;
                if(i<resolution-1) deltaP.x += p[index+face_num];
                else deltaP.x += p[index];
                if(i>0) deltaP.x -= p[index-face_num];
                else deltaP.x -= p[index];
                if(j<resolution-1) deltaP.y += p[index+resolution];
                else deltaP.y += p[index];
                if(j>0) deltaP.y -= p[index-resolution];
                else deltaP.y -= p[index];
                if(k<resolution-1) deltaP.z += p[index+1];
                else deltaP.z += p[index];
                if(k>0) deltaP.z -= p[index-1];
                else deltaP.z -= p[index];
                grid->getVoxel(i,j,k)->getCurrentState()->u -= deltaP*time;
            }
        }
    }
}












