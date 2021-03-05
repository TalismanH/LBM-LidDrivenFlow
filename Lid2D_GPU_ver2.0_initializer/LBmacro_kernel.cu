//宏观量计算并行

#include "basicVar.cuh"

__global__ void LBmacro_kernel(double* rho, double* ux, double* uy, double* u0x, double* u0y,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	int* Flag) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = idx + idy * blockDim.x * gridDim.x;

	if (Flag[threadId] == 'F') {

		u0x[threadId] = ux[threadId];
		u0y[threadId] = uy[threadId];

		rho[threadId] = f0[threadId] + f1[threadId] + f2[threadId] + f3[threadId] + f4[threadId] + f5[threadId] + f6[threadId] + f7[threadId] + f8[threadId];
		ux[threadId] = (f1[threadId] - f3[threadId] + f5[threadId] - f6[threadId] - f7[threadId] + f8[threadId]) / rho[threadId];
		uy[threadId] = (f2[threadId] - f4[threadId] + f5[threadId] + f6[threadId] - f7[threadId] - f8[threadId]) / rho[threadId];

	}

}




