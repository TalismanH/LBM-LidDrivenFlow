
/*System includes*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iomanip>
#include <math.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <omp.h>

#include "basicVar.cuh"

using namespace std;

//LBM MACROSCOPIC VARIABLES
double* ux, * uy, * ux0, * uy0, * rho, * euler_xforce, * euler_yforce;
//LBM DISTRIBUTION FUNCTIONS
double* f0, * f1, * f2, * f3, * f4, * f5, * f6, * f7, * f8;
double* F0, * F1, * F2, * F3, * F4, * F5, * F6, * F7, * F8;
//TOPOLOGICAL IDENTIFIER
int* flag;
//MRT Operator
double* S;
//ERROR
double error = 0;

void initializer()
{
/***********************************************************************/
/*****************Unified Memory Allocation for BasicVar****************/
/***********************************************************************/
	size_t size_int_euler = sizeof( int ) * NX * NY;
	size_t size_double_euler = sizeof( double ) * NX * NY;
	
	cudaMallocManaged( ( void** ) & f0 , size_double_euler );	cudaMallocManaged( ( void** ) & ux,				size_double_euler );		cudaMallocManaged( ( void** ) & ux0,size_double_euler );
	cudaMallocManaged( ( void** ) & f1 , size_double_euler );	cudaMallocManaged( ( void** ) & uy,				size_double_euler );		cudaMallocManaged( ( void** ) & uy0,size_double_euler );
	cudaMallocManaged( ( void** ) & f2 , size_double_euler );	
	cudaMallocManaged( ( void** ) & f3 , size_double_euler );	
	cudaMallocManaged( ( void** ) & f4 , size_double_euler );	cudaMallocManaged( ( void** ) & rho,			size_double_euler );
	cudaMallocManaged( ( void** ) & f5 , size_double_euler );	cudaMallocManaged( ( void** ) & euler_xforce,	size_double_euler );
	cudaMallocManaged( ( void** ) & f6 , size_double_euler );	cudaMallocManaged( ( void** ) & euler_yforce,	size_double_euler );
	cudaMallocManaged( ( void** ) & f7 , size_double_euler );	
	cudaMallocManaged( ( void** ) & f8 , size_double_euler );

	cudaMallocManaged((void**)&F0, size_double_euler);
	cudaMallocManaged((void**)&F1, size_double_euler);
	cudaMallocManaged((void**)&F2, size_double_euler);
	cudaMallocManaged((void**)&F3, size_double_euler);
	cudaMallocManaged((void**)&F4, size_double_euler);
	cudaMallocManaged((void**)&F5, size_double_euler);
	cudaMallocManaged((void**)&F6, size_double_euler);
	cudaMallocManaged((void**)&F7, size_double_euler);
	cudaMallocManaged((void**)&F8, size_double_euler);

	cudaMallocManaged( ( void** ) & flag,			size_int_euler );

	cudaMallocManaged( ( void * * ) & S, 9 * sizeof( double ) );

	S[0] = 0; S[1] = 1.4; S[2] = 1.4; S[3] = 1.0; S[4] = 1.2; S[5] = 1.0; S[6] = 1.2; S[7] = 1.0 / tau_f; S[8] = 1.0 / tau_f;

	for( int i = 0; i < NX; ++i )
		for( int j = 0; j < NY; ++j ) {

				rho[ j * NX + i ] = rho0;

				ux[ j * NX + i ] = 0;		ux0[ j * NX + i ] = 0;
				uy[ j * NX + i ] = 0;		uy0[ j * NX + i ] = 0;

				ux[(NY - 1) * NX + i] = U;

				euler_xforce[ j * NX + i ] = 0;
				euler_yforce[ j * NX + i ] = 0;

				f0[ j * NX + i ] = feq( 0, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );
				f1[ j * NX + i ] = feq( 1, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );
				f2[ j * NX + i ] = feq( 2, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );
				f3[ j * NX + i ] = feq( 3, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );
				f4[ j * NX + i ] = feq( 4, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );
				f5[ j * NX + i ] = feq( 5, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );
				f6[ j * NX + i ] = feq( 6, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );
				f7[ j * NX + i ] = feq( 7, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );
				f8[ j * NX + i ] = feq( 8, rho[ j * NX + i ], ux[ j * NX + i ], uy[ j * NX + i ] );

				flag[ j * NX + i ] = 'F';
			}
//------------------------------------------

	for (int i = 1; i < NX - 1; i++) {
		flag[i] = 'B';
		flag[(NY - 1) * NX + i] = 'T';
	}
	for (int j = 1; j < NY - 1; j++) {
		flag[j * NX] = 'L';
		flag[j * NX + NX - 1] = 'R';
	}

	flag[0] = 'M'; 
	flag[NX - 1] = 'N'; 
	flag[(NY - 1) * NX] = 'Q'; 
	flag[NY * NX - 1] = 'P';
}

void output( const int m )
{
	ostringstream name;
	name<<"C:\\Users\\Admin\\Desktop\\MRT_test\\flow field_"<< m<<".dat";
	ofstream out(name.str().c_str());

	out<<"title=\"cavity flow\"\n"<<"VARIABLES=\"X\",\"Y\",\"U\",\"V\",\"Pressure\"\n"<<"ZONE T= \"BOX\",I="<<NX<<",J="<<NY<<",F=POINT"<<std::endl;

	for (int j = 0; j < NY; j++)
		for (int i = 0; i < NX; i++)
		{
			out << i << " " << j << " " << ux[j * NX + i] << " " << uy[j * NX + i] << " " << rho[j * NX + i] / 3. << endl;
		}
}

double Error(const double* ux, const double* uy, const double* u0x, const double* u0y)
{
	double temp1, temp2;
	temp1 = 0;
	temp2 = 0;
	for (int i = 1; i < NX - 1; i++)
		for (int j = 1; j < NY - 1; j++)
		{
			temp1 += ((ux[j * NX + i] - u0x[j * NX + i]) * (ux[j * NX + i] - u0x[j * NX + i]) + (uy[j * NX + i] - u0y[j * NX + i]) * (uy[j * NX + i] - u0y[j * NX + i]));
			temp2 += (ux[j * NX + i] * ux[j * NX + i] + uy[j * NX + i] * uy[j * NX + i]);
		}
	temp1 = sqrt(temp1);
	temp2 = sqrt(temp2);
	return temp1 / (temp2 + 1e-30);
}

inline void Swap(double& f1, double& f2)
{
	double temp = f1;
	f1 = f2;
	f2 = temp;
}

cudaError_t RunLBM()
{
	cudaError_t cudaStatus = cudaSetDevice( 0 );
	if( cudaStatus != cudaSuccess ) {				
		printf( "cudaSetDevice failed !\n" );
		goto Error;
	}

	dim3 grid_lbm(NX / NUM_THREADS_LBM, NY);
	dim3 threads_lbm(NUM_THREADS_LBM, 1, 1);

	//dim3 threads_col(NUM_THREADS_COL, 1, 1);
	//dim3 grid_col(NX / NUM_THREADS_COL, NY, NZ);

	double st = omp_get_wtime();

	for( int n = 0; ; ++n ) {

		LBColl<<< grid_lbm, threads_lbm >>>( NX, NY, flag, S, rho, ux, uy, euler_xforce, euler_yforce,
											 f0, f1, f2, f3, f4, f5, f6, f7, f8 );
		LBBC<<< grid_lbm, threads_lbm >>>( NX, NY, flag, rho, ux, uy, f0, f1, f2, f3, f4, f5, f6, f7, f8 );
		LBProp<<< grid_lbm, threads_lbm >>>( NX, NY, flag, f0, f1, f2, f3, f4, f5, f6, f7, f8, F0, F1, F2, F3, F4, F5, F6, F7, F8);
		//cudaDeviceSynchronize();
		
		//for (int i = 0; i < NX; i++)
		//	for (int j = 0; j < NY; j++)
		//	{
		//		if (flag[j * NX + i] == 'F') {
		//			Swap(f3[j * NX + i], f1[j * NX + i + 1]);
		//			Swap(f4[j * NX + i], f2[(j + 1) * NX + i]);
		//			Swap(f7[j * NX + i], f5[(j + 1) * NX + i + 1]);
		//			Swap(f8[j * NX + i], f6[(j + 1) * NX + i - 1]);
		//		}
		//		if (flag[j * NX + i] == 'L') {
		//			Swap(f3[j * NX + i], f1[j * NX + i + 1]);
		//			Swap(f7[j * NX + i], f5[(j + 1) * NX + i + 1]);
		//		}
		//		if (flag[j * NX + i] == 'B') {
		//			Swap(f4[j * NX + i], f2[(j + 1) * NX + i]);
		//			Swap(f7[j * NX + i], f5[(j + 1) * NX + i + 1]);
		//			Swap(f8[j * NX + i], f6[(j + 1) * NX + i - 1]);
		//		}
		//		if (flag[j * NX + i] == 'R') {
		//			Swap(f8[j * NX + i], f6[(j + 1) * NX + i - 1]);
		//		}
		//		if (flag[j * NX + i] == 'M') Swap(f7[j * NX + i], f5[(j + 1) * NX + i + 1]);
		//		if (flag[j * NX + i] == 'N') Swap(f8[j * NX + i], f6[(j + 1) * NX + i - 1]);
		//	}
		LBUpgrade<<< grid_lbm, threads_lbm >>> (NX, NY, f0, f1, f2, f3, f4, f5, f6, f7, f8,
			F0, F1, F2, F3, F4, F5, F6, F7, F8, flag);

		LBmacro<<< grid_lbm, threads_lbm >>>( NX, NY, flag, rho, ux, uy, f0, f1, f2, f3, f4, f5, f6, f7, f8, ux0, uy0 );

		cudaDeviceSynchronize();

		if( n % 100 == 0 ) {

			error = Error( ux, uy, ux0, uy0 );
			cout<<"The"<<n<<"th computation result:"<<endl<<"The u,v of point (NX/2,NY/2)is:"<<setprecision(6)<<ux[ NX / 2 + NY * NX / 2 ]<<","<<uy[ NX / 2 + NY * NX / 2 ]<<endl;
			cout<<"The max relative error of uv is :"<<setiosflags(ios::scientific)<<error<<endl;
			cout<<"-----------------------------------------------------\n"<<endl;
			if (n >= 1000)
			{
				if (n % 1000 == 0) output(n);
				if (error < 1.0e-6) break;
			}
		}
	}

	double runtime = omp_get_wtime() - st;

	cout << "Total time is " << runtime << endl;

Error:

/***********************************************************************/
/****************Unified Memory Deallocation for BasicVar***************/
/***********************************************************************/
	cudaFree( f0 );		cudaFree( ux );		cudaFree( ux0 );
	cudaFree( f1 );		cudaFree( uy );		cudaFree( uy0 );
	cudaFree( f2 );		
	cudaFree( f3 );
	cudaFree( f4 );		cudaFree( rho );	
	cudaFree( f5 );		cudaFree( euler_xforce );
	cudaFree( f6 );		cudaFree( euler_yforce );
	cudaFree( f7 );		
	cudaFree( f8 );

	cudaFree(F0);
	cudaFree(F1);
	cudaFree(F2);
	cudaFree(F3);
	cudaFree(F4);
	cudaFree(F5);
	cudaFree(F6);
	cudaFree(F7);
	cudaFree(F8);

	cudaFree( S );		cudaFree( flag );
	
	return cudaStatus;
}

int main( int argc, char *argv[] )
{
	initializer();
	printf( "Initialization Successed!\n" );

	cudaError_t cudaStatus = RunLBM();
	if( cudaStatus != cudaSuccess ) {
		printf( "RunLBM failed !\n" );
		return 1;
	}

	cudaStatus = cudaDeviceReset();
	if( cudaStatus != cudaSuccess ) {
		printf( "cudaDeviceReset failed !\n" );
		return 1;
	}

	return 0;
}