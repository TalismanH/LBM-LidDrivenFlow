
#ifndef _BASICVAR_CUH
#define _BASICVAR_CUH

/*cuda runtime includes*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int NX = 128;
const int NY = 128;

const double rho0 = 1.0;
const double U = 0.1;

const double niu = 0.01;
const double tau_f = 3. * niu + 0.5;

#define NUM_THREADS_LBM 64

//#define NUM_THREADS_COL 32 // 碰撞步骤中 共享存储器 使用的数据量

// 函数原型

__device__ __host__ double feq(int k, double rho, double ux, double uy);

__global__ void LBColl(int nx, int ny, int* flag, double* S, double* rho, double* ux, double* uy, double* euler_xforce, double* euler_yforce,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8);

__global__ void LBProp(int nx, int ny, int* flag,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8);

__global__ void LBBC(int nx, int ny, int* Flag, double* rho, double* ux, double* uy,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8);

__global__ void LBUpgrade(int nx, int ny, double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8, int* flag);

__global__ void LBmacro(int nx, int ny, int* flag, double* rho, double* ux, double* uy,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* err_ux, double* err_uy);

#endif