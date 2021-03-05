
#include "basicVar.cuh"

__global__ void LBInitializer(const int NX, const int NY, const double rho0, const double U,
							double* ux, double* uy, double* u0x, double* u0y, double* rho, int* Flag,
							double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = idx + idy * blockDim.x * gridDim.x;

	ux[threadId] = 0;
	uy[threadId] = 0;
	u0x[threadId] = 0;
	u0y[threadId] = 0;

	rho[threadId] = rho0;

	ux[(NY - 1) * NX + idx] = U;

	f0[threadId] = feq(0, rho[threadId], ux[threadId], uy[threadId]);
	f1[threadId] = feq(1, rho[threadId], ux[threadId], uy[threadId]);
	f2[threadId] = feq(2, rho[threadId], ux[threadId], uy[threadId]);
	f3[threadId] = feq(3, rho[threadId], ux[threadId], uy[threadId]);
	f4[threadId] = feq(4, rho[threadId], ux[threadId], uy[threadId]);
	f5[threadId] = feq(5, rho[threadId], ux[threadId], uy[threadId]);
	f6[threadId] = feq(6, rho[threadId], ux[threadId], uy[threadId]);
	f7[threadId] = feq(7, rho[threadId], ux[threadId], uy[threadId]);
	f8[threadId] = feq(8, rho[threadId], ux[threadId], uy[threadId]);

	if (idx > 0 && idy > 0 && idx < NX - 1 && idy < NY - 1) Flag[idy * NX + idx] = 'F';
	if (idx > 0 && idx < NX - 1) {
		Flag[idx] = 'B';
		Flag[(NY - 1) * NX + idx] = 'T';
	}
	if (idy > 0 && idy < NY - 1) {
		Flag[idy * NX] = 'L';
		Flag[idy * NX + NX - 1] = 'R';
	}
	Flag[0] = 'M'; Flag[NX - 1] = 'N'; Flag[(NY - 1) * NX] = 'Q'; Flag[NY * NX - 1] = 'P';
}
