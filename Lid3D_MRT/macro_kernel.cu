
#include "basicVar.cuh"

__global__ void LBmacro( int nx, int ny, int nz, int * flag, double * rho, double * ux, double * uy, double * uz, //double * euler_xforce, double * euler_yforce, double * euler_zforce,
						 double * f0, double * f1, double * f2, double * f3, double * f4, double * f5, double * f6, double * f7, double * f8,
						 double * f9, double * f10, double * f11, double * f12, double * f13, double * f14, double * f15, double * f16, double * f17, double * f18,
						 double * err_ux, double * err_uy, double * err_uz )
{
	int tx = threadIdx.x;
	int k = blockIdx.y * nx * ny + blockIdx.x * nx + tx;

	if( flag[ k ] == 'F' ) {

		err_ux[ k ] = ux[ k ];
		err_uy[ k ] = uy[ k ];
		err_uz[ k ] = uz[ k ];

		rho[ k ] = f0[ k ] + f1[ k ] + f2[ k ] + f3[ k ] + f4[ k ] + f5[ k ] + f6[ k ] + f7[ k ] + f8[ k ] + f9[ k ] + f10[ k ] + f11[ k ] + f12[ k ] + f13[ k ] + f14[ k ] + f15[ k ] + f16[ k ] + f17[ k ] + f18[ k ];
		ux[k] = (f1[k] - f2[k] + f7[k] - f8[k] + f9[k] - f10[k] + f11[k] - f12[k] + f13[k] - f14[k]) / rho[k];		 //+ 0.5 * euler_xforce[ k ];
		uy[k] = (f3[k] - f4[k] + f7[k] - f8[k] - f9[k] + f10[k] + f15[k] - f16[k] + f17[k] - f18[k]) / rho[k];		 //+ 0.5 * euler_yforce[ k ];
		uz[k] = (f5[k] - f6[k] + f11[k] - f12[k] - f13[k] + f14[k] + f15[k] - f16[k] - f17[k] + f18[k]) / rho[k];    //+ 0.5 * euler_zforce[ k ];
	}
}