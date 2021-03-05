
#ifndef _BASICVAR_CUH
#define _BASICVAR_CUH

/*cuda runtime includes*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const int NX = 64;
const int NY = 64;
const int NZ = 64;

const double rho0 = 1.0;
const double U = 0.1;

const double niu = 0.01;
const double tau_f = 3. * niu + 0.5;

#define NUM_THREADS_LBM 64 // 必须等于 NX，因为线程的组织方式

#define NUM_THREADS_COL 32 // 碰撞步骤中 共享存储器 使用的数据量

// 函数原型

__device__ __host__ double feq(int k, double rho, double ux, double uy, double uz);

__global__ void LBColl(int nx, int ny, int nz, int* flag, double* S, double* rho, double* ux, double* uy, double* uz, double* euler_xforce, double* euler_yforce, double* euler_zforce,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* f9, double* f10, double* f11, double* f12, double* f13, double* f14, double* f15, double* f16, double* f17, double* f18);

__global__ void LBProp(int nx, int ny, int nz, int* flag,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* f9, double* f10, double* f11, double* f12, double* f13, double* f14, double* f15, double* f16, double* f17, double* f18);

__global__ void LBBC(int nx, int ny, int nz, int* flag, double* rho, double* ux, double* uy, double* uz,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* f9, double* f10, double* f11, double* f12, double* f13, double* f14, double* f15, double* f16, double* f17, double* f18);

__global__ void LBmacro(int nx, int ny, int nz, int* flag, double* rho, double* ux, double* uy, double* uz, //double * euler_xforce, double * euler_yforce, double * euler_zforce,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* f9, double* f10, double* f11, double* f12, double* f13, double* f14, double* f15, double* f16, double* f17, double* f18,
	double* err_ux, double* err_uy, double* err_uz);

#endif