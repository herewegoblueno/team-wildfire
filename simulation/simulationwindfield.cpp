#include "simulator.h"
#include <math.h>
#include <Eigen/Cholesky>
#include <iostream>


double get_verticity_len(Voxel* v) {return v->getVerticity().length();}

void Simulator::stepVoxelWind(Voxel* v, double deltaTimeInMs)
{
    dvec3 u = v->getLastFrameState()->u;
    dvec3 u_f = v->getCurrentState()->u;

    u.x = advect(get_q_ux, u_f, deltaTimeInMs, v);
    u.y = advect(get_q_uy, u_f, deltaTimeInMs, v);
    u.z = advect(get_q_uz, u_f, deltaTimeInMs, v);
    u = verticity_confinement(u, v, deltaTimeInMs);

    double T_th = v->getCurrentState()->temperature;
    double T_air = absolute_temp(v->centerInWorldSpace.y);
    double q_v = v->getLastFrameState()->q_v;
    float X_v = mole_fraction(q_v);
    float M_th = avg_mole_mass(X_v);
    dvec3 buoyancy_gravity(0, gravity_acceleration, 0); // upward
    dvec3 buoyancy = -buoyancy_gravity*(28.96*T_th/(M_th*T_air) - 1);
    u = u + buoyancy*(double)deltaTimeInMs; // can't think of externel force

    v->getCurrentState()->u = u;
}

// verticity confinement origrinated from Steinhoff and Underhill [1994],
dvec3 Simulator::verticity_confinement(glm::dvec3 u, Voxel* v, double time)
{
    dvec3 verticity = v->getVerticity() + 0.0001;
    dvec3 d_verticity = v->getGradient(get_verticity_len) + 0.0001;
    d_verticity = glm::normalize(d_verticity);
    dvec3 f_omega = verticity_epsilon*v->grid->cellSideLength()*glm::cross(d_verticity, verticity);
    return u + f_omega*time;
}

void Simulator::pressure_projection_PCG(VoxelGrid *grid, double time)
{
//    int cell_num = A.
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

    Eigen::VectorXd p = Eigen::VectorXd(cell_num,1);

    Eigen::SimplicialLLT <Eigen::SparseMatrix<double>> solver(A);
    solver.compute(A);
    p = solver.solve(d);

    for(int i=0; i<resolution;i++)
        for(int j=0; j<resolution;j++)
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












