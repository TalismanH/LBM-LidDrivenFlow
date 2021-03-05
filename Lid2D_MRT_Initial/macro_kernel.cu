
#include "basicVar.cuh"

__global__ void LBmacro( int nx, int ny, int * flag, double * rho, double * ux, double * uy,
						 double * f0, double * f1, double * f2, double * f3, double * f4, double * f5, double * f6, double * f7, double * f8,
						 double * err_ux, double * err_uy )
{
	int tx = threadIdx.x;
	int k = blockIdx.y * nx + blockIdx.x * NUM_THREADS_LBM + tx;

	if( flag[ k ] == 'F' ) {

		err_ux[ k ] = ux[ k ];
		err_uy[ k ] = uy[ k ];

		rho[k] = f0[k] + f1[k] + f2[k] + f3[k] + f4[k] + f5[k] + f6[k] + f7[k] + f8[k];
		ux[k] = (f1[k] - f3[k] + f5[k] - f6[k] - f7[k] + f8[k]) / rho[k];
		uy[k] = (f2[k] - f4[k] + f5[k] + f6[k] - f7[k] - f8[k]) / rho[k];
	}
}