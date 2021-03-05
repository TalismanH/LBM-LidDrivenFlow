//文件开头，防止重复编译头文件
#ifndef BASICVAR_CUH_ 
#define BASICVAR_CUH_ 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std; //命名空间

/********************常量声明*************************/

//const int e[9][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} }; //离散速度张量
//const double w[9] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //权系数
const int NX = 256; //x方向
const int NY = 256; //y方向
const double U = 0.1; //顶盖速度
const double rho0 = 1.0;
const int Re = 1000;
const double niu = U * NX / Re;
const double tau_f = 3.0 * niu + 0.5;

/***********************函数原型**********************/

void initializer(); //初始化函数
//double feq(const int k, const double rho, const double ux, const double uy); //平衡态分布函数
//void evolution(double* rho, double* ux, double* uy,
//	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8);
//演化函数
void output(int m); //输出函数
double Error(const double* ux, const double* uy, const double* u0x, const double* u0y); //误差函数

void memoryInitiate(); //分配内存
void memoryfinalize(); //清除内存

inline cudaError_t checkCuda(cudaError_t result);

//1. 计算平衡态分布函数
__device__ double feq(const int k, const double rho, const double ux, const double uy);

__global__ void LBInitializer(const int NX, const int NY, const double rho0, const double U, double* ux, double* uy, double* u0x, double* u0y, double* rho, int* Flag,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8);

//2. 碰撞
__global__ void LBColl(double* rho, double* ux, double* uy, double* f0, double* f1, double* f2, double* f3,
	double* f4, double* f5, double* f6, double* f7, double* f8, int* Flag, const double tau_f);

__global__ void LBBoundary(const int NX, const int NY, double* rho, double* ux, double* uy, double* f0, double* f1, double* f2, double* f3,
	double* f4, double* f5, double* f6, double* f7, double* f8, int* Flag);

__global__ void LBProp(const int NX, const int NY, double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8, int* Flag);

__global__ void LBUpgrade(double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8, int* Flag);

__global__ void LBmacro_kernel(double* rho, double* ux, double* uy, double* u0x, double* u0y,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	int* Flag);


#endif