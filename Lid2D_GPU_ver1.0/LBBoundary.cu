
#include "basicVar.cuh"

__global__ void LBBoundary(const int NX, const int NY, double* rho, double* ux, double* uy, double* f0, double* f1, double* f2, double* f3,
	double* f4, double* f5, double* f6, double* f7, double* f8, int* Flag) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int threadId = idx + idy * blockDim.x * gridDim.x;
	//int threadId = idx + idy * NX;

	double _rho;
	double _ux;
	double _uy;

	//左边界
	if (Flag[threadId] == 'L') {

		_rho = rho[threadId + 1];
		_ux = ux[threadId + 1];
		_uy = uy[threadId + 1];

		rho[threadId] = _rho;
		f1[threadId] = feq(1, _rho, 0, 0) + f1[threadId + 1] - feq(1, _rho, _ux, _uy);
		f5[threadId] = feq(5, _rho, 0, 0) + f5[threadId + 1] - feq(5, _rho, _ux, _uy);
		f8[threadId] = feq(8, _rho, 0, 0) + f8[threadId + 1] - feq(8, _rho, _ux, _uy);

		//rho[threadId] = rho[threadId + 1];
		//f1[threadId] = feq(1, rho[threadId], 0, 0) + f1[threadId + 1] - feq(1, rho[threadId + 1], ux[threadId + 1], uy[threadId + 1]);
		//f5[threadId] = feq(5, rho[threadId], 0, 0) + f5[threadId + 1] - feq(5, rho[threadId + 1], ux[threadId + 1], uy[threadId + 1]);
		//f8[threadId] = feq(8, rho[threadId], 0, 0) + f8[threadId + 1] - feq(8, rho[threadId + 1], ux[threadId + 1], uy[threadId + 1]);
	}
	//右边界
	if (Flag[threadId] == 'R') {

		_rho = rho[threadId - 1];
		_ux = ux[threadId - 1];
		_uy = uy[threadId - 1];

		rho[threadId] = _rho;
		f3[threadId] = feq(3, _rho, 0, 0) + f3[threadId - 1] - feq(3, _rho, _ux, _uy);
		f6[threadId] = feq(6, _rho, 0, 0) + f6[threadId - 1] - feq(6, _rho, _ux, _uy);
		f7[threadId] = feq(7, _rho, 0, 0) + f7[threadId - 1] - feq(7, _rho, _ux, _uy);

		//rho[threadId] = rho[threadId - 1];
		//f3[threadId] = feq(3, rho[threadId], 0, 0) + f3[threadId - 1] - feq(3, rho[threadId - 1], ux[threadId - 1], uy[threadId - 1]);
		//f6[threadId] = feq(6, rho[threadId], 0, 0) + f6[threadId - 1] - feq(6, rho[threadId - 1], ux[threadId - 1], uy[threadId - 1]);
		//f7[threadId] = feq(7, rho[threadId], 0, 0) + f7[threadId - 1] - feq(7, rho[threadId - 1], ux[threadId - 1], uy[threadId - 1]);
	}
	//上边界
	if (Flag[threadId] == 'T') {

		_rho = rho[threadId - NX];
		_ux = ux[threadId - NX];
		_uy = uy[threadId - NX];

		rho[threadId] = _rho;
		ux[threadId] = 0.1;
		f4[threadId] = feq(4, _rho, 0.1, 0) + f4[threadId - NX] - feq(4, _rho, _ux, _uy);
		f7[threadId] = feq(7, _rho, 0.1, 0) + f7[threadId - NX] - feq(7, _rho, _ux, _uy);
		f8[threadId] = feq(8, _rho, 0.1, 0) + f8[threadId - NX] - feq(8, _rho, _ux, _uy);
	}
	//下边界
	if (Flag[threadId] == 'B') {

		_rho = rho[threadId + NX];
		_ux = ux[threadId + NX];
		_uy = uy[threadId + NX];

		rho[threadId] = _rho;
		f2[threadId] = feq(2, _rho, 0, 0) + f2[threadId + NX] - feq(2, _rho, _ux, _uy);
		f5[threadId] = feq(5, _rho, 0, 0) + f5[threadId + NX] - feq(5, _rho, _ux, _uy);
		f6[threadId] = feq(6, _rho, 0, 0) + f6[threadId + NX] - feq(6, _rho, _ux, _uy);
	}

	if (Flag[threadId] == 'M') {
		_rho = rho[threadId + NX + 1];
		_ux = ux[threadId + NX + 1];
		_uy = uy[threadId + NX + 1];
		rho[threadId] = _rho;
		f5[threadId] = feq(5, _rho, 0, 0) + f5[threadId + NX + 1] - feq(5, _rho, _ux, _uy);
	}
	if (Flag[threadId] == 'N') {
		_rho = rho[threadId + NX - 1];
		_ux = ux[threadId + NX - 1];
		_uy = uy[threadId + NX - 1];
		rho[threadId] = _rho;
		f6[threadId] = feq(6, _rho, 0, 0) + f6[threadId + NX - 1] - feq(6, _rho, _ux, _uy);
	}
	if (Flag[threadId] == 'Q') {
		_rho = rho[threadId - NX + 1];
		_ux = ux[threadId - NX + 1];
		_uy = uy[threadId - NX + 1];
		rho[threadId] = _rho;
		ux[threadId] = 0.1;
		f8[threadId] = feq(8, _rho, 0.1, 0) + f8[threadId - NX + 1] - feq(8, _rho, _ux, _uy);
	}
	if (Flag[threadId] == 'P') {
		_rho = rho[threadId - NX - 1];
		_ux = ux[threadId - NX - 1];
		_uy = uy[threadId - NX - 1];
		rho[threadId] = _rho;
		ux[threadId] = 0.1;
		f7[threadId] = feq(7, _rho, 0.1, 0) + f7[threadId - NX - 1] - feq(7, _rho, _ux, _uy);
	}
}