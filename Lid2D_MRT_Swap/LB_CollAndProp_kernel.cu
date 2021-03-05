
#include "basicVar.cuh"

__device__ double meq(int k, double rho, double ux, double uy)
{
	switch (k) {

		case(0): return rho;
		case(1): return -2. * rho + 3. * (ux * ux + uy * uy) * rho;
		case(2): return rho - 3. * (ux * ux + uy * uy) * rho;
		case(3): return ux * rho;
		case(4): return -ux * rho;
		case(5): return uy * rho;
		case(6): return -uy * rho;
		case(7): return (ux * ux - uy * uy) * rho;
		case(8): return ux * uy * rho;
	}
}

__device__ double mForceGenerator(int k, double ux, double uy, double euler_xforce, double euler_yforce)
{
	switch (k) {

		case(0): return 0;
		case(1): return 6. * (ux * euler_xforce + uy * euler_yforce);
		case(2): return -6. * (ux * euler_xforce + uy * euler_yforce);
		case(3): return euler_xforce;
		case(4): return -euler_xforce;
		case(5): return euler_yforce;
		case(6): return -euler_yforce;
		case(7): return  2. * (ux * euler_xforce - uy * euler_yforce);
		case(8): return (ux * euler_yforce + uy * euler_xforce);
	}

}

__device__ __host__ double feq(int k, double rho, double ux, double uy)
{
	int e[9][2] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 1 }, { -1, 1 }, { -1, -1 }, { 1, -1 } };
	double w[9] = { 4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36. };

	double eu, uv;
	eu = (e[k][0] * ux + e[k][1] * uy);
	uv = ux * ux + uy * uy;
	return w[k] * (rho + 3. * eu + 4.5 * eu * eu - 1.5 * uv);
}

__device__ inline void Swap( double & f1, double & f2 )
{
	double temp = f1;
	f1 = f2;
	f2 = temp;
}

__global__ void LBColl(int nx, int ny, int* flag, double* S, double* rho, double* ux, double* uy, double* euler_xforce, double* euler_yforce,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8)
{
	int tx = threadIdx.x;
	int k = blockIdx.y * nx + blockIdx.x * NUM_THREADS_LBM + tx;

	__shared__ double F0 [ NUM_THREADS_LBM ];		__shared__ double mf0 [ NUM_THREADS_LBM ];
	__shared__ double F1 [ NUM_THREADS_LBM ];		__shared__ double mf1 [ NUM_THREADS_LBM ];
	__shared__ double F2 [ NUM_THREADS_LBM ];		__shared__ double mf2 [ NUM_THREADS_LBM ];
	__shared__ double F3 [ NUM_THREADS_LBM ];		__shared__ double mf3 [ NUM_THREADS_LBM ];
	__shared__ double F4 [ NUM_THREADS_LBM ];		__shared__ double mf4 [ NUM_THREADS_LBM ];
	__shared__ double F5 [ NUM_THREADS_LBM ];		__shared__ double mf5 [ NUM_THREADS_LBM ];
	__shared__ double F6 [ NUM_THREADS_LBM ];		__shared__ double mf6 [ NUM_THREADS_LBM ];
	__shared__ double F7 [ NUM_THREADS_LBM ];		__shared__ double mf7 [ NUM_THREADS_LBM ];
	__shared__ double F8 [ NUM_THREADS_LBM ];		__shared__ double mf8 [ NUM_THREADS_LBM ];

	double _rho;	_rho = rho[ k ];
	double _ux;		_ux = ux[ k ];
	double _uy;		_uy = uy[ k ];
	
	double EXF;		EXF = euler_xforce[ k ];
	double EYF;		EYF = euler_yforce[ k ];

	F0 [ tx ] = f0 [ k ];
	F1 [ tx ] = f1 [ k ];
	F2 [ tx ] = f2 [ k ];
	F3 [ tx ] = f3 [ k ];
	F4 [ tx ] = f4 [ k ];
	F5 [ tx ] = f5 [ k ];
	F6 [ tx ] = f6 [ k ];
	F7 [ tx ] = f7 [ k ];
	F8 [ tx ] = f8 [ k ];

	__syncthreads();

	if( flag[ k ] == 'F' ) {

		mf0[tx] = F0[tx] + F1[tx] + F2[tx] + F3[tx] + F4[tx] + F5[tx] + F6[tx] + F7[tx] + F8[tx];
		mf1[tx] = -4. * F0[tx] - F1[tx] - F2[tx] - F3[tx] - F4[tx] + 2. * F5[tx] + 2. * F6[tx] + 2. * F7[tx] + 2. * F8[tx];
		mf2[tx] = 4. * F0[tx] - 2. * F1[tx] - 2. * F2[tx] - 2. * F3[tx] - 2. * F4[tx] + F5[tx] + F6[tx] + F7[tx] + F8[tx];
		mf3[tx] = F1[tx] - F3[tx] + F5[tx] - F6[tx] - F7[tx] + F8[tx];
		mf4[tx] = -2. * F1[tx] + 2. * F3[tx] + F5[tx] - F6[tx] - F7[tx] + F8[tx];
		mf5[tx] = F2[tx] - F4[tx] + F5[tx] + F6[tx] - F7[tx] - F8[tx];
		mf6[tx] = -2. * F2[tx] + 2. * F4[tx] + F5[tx] + F6[tx] - F7[tx] - F8[tx];
		mf7[tx] = F1[tx] - F2[tx] + F3[tx] - F4[tx];
		mf8[tx] = F5[tx] - F6[tx] + F7[tx] - F8[tx];

		mf0[tx] = mf0[tx] + S[0] * (meq(0, _rho, _ux, _uy) - mf0[tx]) + (1. - 0.5 * S[0]) * mForceGenerator(0, _ux, _uy, EXF, EYF);
		mf1[tx] = mf1[tx] + S[1] * (meq(1, _rho, _ux, _uy) - mf1[tx]) + (1. - 0.5 * S[1]) * mForceGenerator(1, _ux, _uy, EXF, EYF);
		mf2[tx] = mf2[tx] + S[2] * (meq(2, _rho, _ux, _uy) - mf2[tx]) + (1. - 0.5 * S[2]) * mForceGenerator(2, _ux, _uy, EXF, EYF);
		mf3[tx] = mf3[tx] + S[3] * (meq(3, _rho, _ux, _uy) - mf3[tx]) + (1. - 0.5 * S[3]) * mForceGenerator(3, _ux, _uy, EXF, EYF);
		mf4[tx] = mf4[tx] + S[4] * (meq(4, _rho, _ux, _uy) - mf4[tx]) + (1. - 0.5 * S[4]) * mForceGenerator(4, _ux, _uy, EXF, EYF);
		mf5[tx] = mf5[tx] + S[5] * (meq(5, _rho, _ux, _uy) - mf5[tx]) + (1. - 0.5 * S[5]) * mForceGenerator(5, _ux, _uy, EXF, EYF);
		mf6[tx] = mf6[tx] + S[6] * (meq(6, _rho, _ux, _uy) - mf6[tx]) + (1. - 0.5 * S[6]) * mForceGenerator(6, _ux, _uy, EXF, EYF);
		mf7[tx] = mf7[tx] + S[7] * (meq(7, _rho, _ux, _uy) - mf7[tx]) + (1. - 0.5 * S[7]) * mForceGenerator(7, _ux, _uy, EXF, EYF);
		mf8[tx] = mf8[tx] + S[8] * (meq(8, _rho, _ux, _uy) - mf8[tx]) + (1. - 0.5 * S[8]) * mForceGenerator(8, _ux, _uy, EXF, EYF);

		__syncthreads();

		f0[k] = mf0[tx] / 9. - mf1[tx] / 9. + mf2[tx] / 9.;
		f3[k] = mf0[tx] / 9. - mf1[tx] / 36. - mf2[tx] / 18. + mf3[tx] / 6. - mf4[tx] / 6. + mf7[tx] / 4.;
		f4[k] = mf0[tx] / 9. - mf1[tx] / 36. - mf2[tx] / 18. + mf5[tx] / 6. - mf6[tx] / 6. - mf7[tx] / 4.;
		f1[k] = mf0[tx] / 9. - mf1[tx] / 36. - mf2[tx] / 18. - mf3[tx] / 6. + mf4[tx] / 6. + mf7[tx] / 4.;
		f2[k] = mf0[tx] / 9. - mf1[tx] / 36. - mf2[tx] / 18. - mf5[tx] / 6. + mf6[tx] / 6. - mf7[tx] / 4.;
		f7[k] = mf0[tx] / 9. + mf1[tx] / 18. + mf2[tx] / 36. + mf3[tx] / 6. + mf4[tx] / 12. + mf5[tx] / 6. + mf6[tx] / 12. + mf8[tx] / 4.;
		f8[k] = mf0[tx] / 9. + mf1[tx] / 18. + mf2[tx] / 36. - mf3[tx] / 6. - mf4[tx] / 12. + mf5[tx] / 6. + mf6[tx] / 12. - mf8[tx] / 4.;
		f5[k] = mf0[tx] / 9. + mf1[tx] / 18. + mf2[tx] / 36. - mf3[tx] / 6. - mf4[tx] / 12. - mf5[tx] / 6. - mf6[tx] / 12. + mf8[tx] / 4.;
		f6[k] = mf0[tx] / 9. + mf1[tx] / 18. + mf2[tx] / 36. + mf3[tx] / 6. + mf4[tx] / 12. - mf5[tx] / 6. - mf6[tx] / 12. - mf8[tx] / 4.;

	}
}

__global__ void LBProp(int nx, int ny, int* flag,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8)
{
	int tx = threadIdx.x;
	int k = blockIdx.y * nx + blockIdx.x * NUM_THREADS_LBM + tx;

	if( flag[ k ] == 'F' ) {

		Swap( f1 [ k + 1 ],				f3 [ k ] );
		Swap( f2 [ k + nx ],			f4 [ k ] );
		Swap( f5 [ k + nx + 1 ],		f7 [ k ] );
		Swap( f6 [ k + nx - 1 ],		f8 [ k ] );
	}

	if( flag[ k ] == 'L' ) {

		Swap(f1[k + 1], f3[k]);
		Swap(f5[k + nx + 1], f7[k]);
	}

	if( flag[ k ] == 'B' ) {

		Swap(f2[k + nx], f4[k]);
		Swap(f5[k + nx + 1], f7[k]);
		Swap(f6[k + nx - 1], f8[k]);
	}

	if( flag[ k ] == 'R' ) {

		Swap(f6[k + nx - 1], f8[k]);
	}

	if( flag[ k ] == 'M' ) {

		Swap(f5[k + nx + 1], f7[k]);
	}

	if( flag[ k ] == 'N' ) {

		Swap(f6[k + nx - 1], f8[k]);
	}
}



__global__ void LBBC(int nx, int ny, int* Flag, double* rho, double* ux, double* uy,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8)
{
	int tx = threadIdx.x;
	int k = blockIdx.y * nx + blockIdx.x * NUM_THREADS_LBM + tx;

	double _rho;
	double _ux;
	double _uy;

	//左边界
	if (Flag[k] == 'L') {

		_rho = rho[k + 1];
		_ux = ux[k + 1];
		_uy = uy[k + 1];

		rho[k] = _rho;
		f3[k] = feq(1, _rho, 0, 0) + f3[k + 1] - feq(1, _rho, _ux, _uy);
		f7[k] = feq(5, _rho, 0, 0) + f7[k + 1] - feq(5, _rho, _ux, _uy);
		f6[k] = feq(8, _rho, 0, 0) + f6[k + 1] - feq(8, _rho, _ux, _uy);

	}
	//右边界
	if (Flag[k] == 'R') {

		_rho = rho[k - 1];
		_ux = ux[k - 1];
		_uy = uy[k - 1];

		rho[k] = _rho;
		f1[k] = feq(3, _rho, 0, 0) + f1[k - 1] - feq(3, _rho, _ux, _uy);
		f8[k] = feq(6, _rho, 0, 0) + f8[k - 1] - feq(6, _rho, _ux, _uy);
		f5[k] = feq(7, _rho, 0, 0) + f5[k - 1] - feq(7, _rho, _ux, _uy);

	}
	//上边界
	if (Flag[k] == 'T') {

		_rho = rho[k - nx];
		_ux = ux[k - nx];
		_uy = uy[k - nx];

		rho[k] = _rho;
		ux[k] = 0.1;
		f2[k] = feq(4, _rho, 0.1, 0) + f2[k - nx] - feq(4, _rho, _ux, _uy);
		f5[k] = feq(7, _rho, 0.1, 0) + f5[k - nx] - feq(7, _rho, _ux, _uy);
		f6[k] = feq(8, _rho, 0.1, 0) + f6[k - nx] - feq(8, _rho, _ux, _uy);
	}
	//下边界
	if (Flag[k] == 'B') {

		_rho = rho[k + nx];
		_ux = ux[k + nx];
		_uy = uy[k + nx];

		rho[k] = _rho;
		f4[k] = feq(2, _rho, 0, 0) + f4[k + nx] - feq(2, _rho, _ux, _uy);
		f7[k] = feq(5, _rho, 0, 0) + f7[k + nx] - feq(5, _rho, _ux, _uy);
		f8[k] = feq(6, _rho, 0, 0) + f8[k + nx] - feq(6, _rho, _ux, _uy);
	}

	if (Flag[k] == 'M') {
		_rho = rho[k + nx + 1];
		_ux = ux[k + nx + 1];
		_uy = uy[k + nx + 1];
		rho[k] = _rho;
		f7[k] = feq(5, _rho, 0, 0) + f7[k + nx + 1] - feq(5, _rho, _ux, _uy);
	}
	if (Flag[k] == 'N') {
		_rho = rho[k + nx - 1];
		_ux = ux[k + nx - 1];
		_uy = uy[k + nx - 1];
		rho[k] = _rho;
		f8[k] = feq(6, _rho, 0, 0) + f8[k + nx - 1] - feq(6, _rho, _ux, _uy);
	}
	if (Flag[k] == 'Q') {
		_rho = rho[k - nx + 1];
		_ux = ux[k - nx + 1];
		_uy = uy[k - nx + 1];
		rho[k] = _rho;
		ux[k] = 0.1;
		f6[k] = feq(8, _rho, 0.1, 0) + f6[k - nx + 1] - feq(8, _rho, _ux, _uy);
	}
	if (Flag[k] == 'P') {
		_rho = rho[k - nx - 1];
		_ux = ux[k - nx - 1];
		_uy = uy[k - nx - 1];
		rho[k] = _rho;
		ux[k] = 0.1;
		f5[k] = feq(7, _rho, 0.1, 0) + f5[k - nx - 1] - feq(7, _rho, _ux, _uy);
	}
}