#include "basicVar.cuh"

__constant__ int e[9][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} }; //离散速度张量
__constant__ double w[9] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //权系数

__device__ double feq(const int k, const double rho, const double ux, const double uy) { //计算平衡态分布函数

	//const int e[9][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} }; //离散速度张量
	//const double w[9] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //权系数

	double eu, uv;
	eu = e[k][0] * ux + e[k][1] * uy; //e_alpha*u
	uv = ux * ux + uy * uy; //u^2
	return w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
}

//碰撞
__global__ void LBColl(double* rho, double* ux, double* uy, double* f0, double* f1, double* f2, double* f3,
	double* f4, double* f5, double* f6, double* f7, double* f8, int* Flag, const double tau_f) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = idx + idy * blockDim.x * gridDim.x;

	//const double tau_f = 0.5384;

	if (Flag[threadId] == 'F') {
		f0[threadId] = f0[threadId] + (feq(0, rho[threadId], ux[threadId], uy[threadId]) - f0[threadId]) / tau_f;
		f1[threadId] = f1[threadId] + (feq(1, rho[threadId], ux[threadId], uy[threadId]) - f1[threadId]) / tau_f;
		f2[threadId] = f2[threadId] + (feq(2, rho[threadId], ux[threadId], uy[threadId]) - f2[threadId]) / tau_f;
		f3[threadId] = f3[threadId] + (feq(3, rho[threadId], ux[threadId], uy[threadId]) - f3[threadId]) / tau_f;
		f4[threadId] = f4[threadId] + (feq(4, rho[threadId], ux[threadId], uy[threadId]) - f4[threadId]) / tau_f;
		f5[threadId] = f5[threadId] + (feq(5, rho[threadId], ux[threadId], uy[threadId]) - f5[threadId]) / tau_f;
		f6[threadId] = f6[threadId] + (feq(6, rho[threadId], ux[threadId], uy[threadId]) - f6[threadId]) / tau_f;
		f7[threadId] = f7[threadId] + (feq(7, rho[threadId], ux[threadId], uy[threadId]) - f7[threadId]) / tau_f;
		f8[threadId] = f8[threadId] + (feq(8, rho[threadId], ux[threadId], uy[threadId]) - f8[threadId]) / tau_f;
	}
}
//迁移
__global__ void LBProp(const int NX, const int NY, double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8, int* Flag) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = idx + idy * blockDim.x * gridDim.x;
	// idy * NX + idx 可以作为索引
	if (Flag[threadId] == 'F') {
		F0[threadId] = f0[threadId];
		F1[threadId] = f1[threadId - 1];
		F2[threadId] = f2[threadId - NX];
		F3[threadId] = f3[threadId + 1];
		F4[threadId] = f4[threadId + NX];
		F5[threadId] = f5[threadId - NX - 1];
		F6[threadId] = f6[threadId - NX + 1];
		F7[threadId] = f7[threadId + NX + 1];
		F8[threadId] = f8[threadId + NX - 1];
	}
	
}

__global__ void LBUpgrade(double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* F0, double* F1, double* F2, double* F3, double* F4, double* F5, double* F6, double* F7, double* F8, int* Flag) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = idx + idy * blockDim.x * gridDim.x;

	if (Flag[threadId] == 'F') {

		f0[threadId] = F0[threadId];
		f1[threadId] = F1[threadId];
		f2[threadId] = F2[threadId];
		f3[threadId] = F3[threadId];
		f4[threadId] = F4[threadId];
		f5[threadId] = F5[threadId];
		f6[threadId] = F6[threadId];
		f7[threadId] = F7[threadId];
		f8[threadId] = F8[threadId];
	}
}