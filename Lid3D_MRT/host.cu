
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
double* ux, * uy, * uz, * ux0, * uy0, * uz0, * rho, * euler_xforce, * euler_yforce, * euler_zforce;
//LBM DISTRIBUTION FUNCTIONS
double* f0, * f1, * f2, * f3, * f4, * f5, * f6, * f7, * f8, * f9, * f10, * f11, * f12, * f13, * f14, * f15, * f16, * f17, * f18;
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
	size_t size_int_euler = sizeof( int ) * NX * NY * NZ;
	size_t size_double_euler = sizeof( double ) * NX * NY * NZ;
	
	cudaMallocManaged( ( void** ) & f0 , size_double_euler );	cudaMallocManaged( ( void** ) & ux,				size_double_euler );		cudaMallocManaged( ( void** ) & ux0,size_double_euler );
	cudaMallocManaged( ( void** ) & f1 , size_double_euler );	cudaMallocManaged( ( void** ) & uy,				size_double_euler );		cudaMallocManaged( ( void** ) & uy0,size_double_euler );
	cudaMallocManaged( ( void** ) & f2 , size_double_euler );	cudaMallocManaged( ( void** ) & uz,				size_double_euler );		cudaMallocManaged( ( void** ) & uz0,size_double_euler );
	cudaMallocManaged( ( void** ) & f3 , size_double_euler );	
	cudaMallocManaged( ( void** ) & f4 , size_double_euler );	cudaMallocManaged( ( void** ) & rho,			size_double_euler );
	cudaMallocManaged( ( void** ) & f5 , size_double_euler );	cudaMallocManaged( ( void** ) & euler_xforce,	size_double_euler );
	cudaMallocManaged( ( void** ) & f6 , size_double_euler );	cudaMallocManaged( ( void** ) & euler_yforce,	size_double_euler );
	cudaMallocManaged( ( void** ) & f7 , size_double_euler );	cudaMallocManaged( ( void** ) & euler_zforce,	size_double_euler );
	cudaMallocManaged( ( void** ) & f8 , size_double_euler );	
	cudaMallocManaged( ( void** ) & f9 , size_double_euler );	cudaMallocManaged( ( void** ) & flag,			size_int_euler );
	cudaMallocManaged( ( void** ) & f10 , size_double_euler );
	cudaMallocManaged( ( void** ) & f11 , size_double_euler );
	cudaMallocManaged( ( void** ) & f12 , size_double_euler );
	cudaMallocManaged( ( void** ) & f13 , size_double_euler );
	cudaMallocManaged( ( void** ) & f14 , size_double_euler );
	cudaMallocManaged( ( void** ) & f15 , size_double_euler );
	cudaMallocManaged( ( void** ) & f16 , size_double_euler );
	cudaMallocManaged( ( void** ) & f17 , size_double_euler );
	cudaMallocManaged( ( void** ) & f18 , size_double_euler );

	cudaMallocManaged( ( void * * ) & S, 19 * sizeof( double ) );

	S[ 0 ] = 0;S[ 1 ] = 1.19;S[ 2 ] = 1.4;S[ 3 ] = 0;S[ 4 ] = 1.2;S[ 5 ] = 0;S[ 6 ] = 1.2;S[ 7 ] = 0;S[ 8 ] = 1.2;S[ 9 ] = 1. / tau_f;S[ 10 ] = 1.4;S[ 11 ] = 1. / tau_f;S[ 12 ] = 1.4;S[ 13 ] = 1. / tau_f;S[ 14 ] = 1. / tau_f;S[ 15 ] = 1. / tau_f;S[ 16 ] = 1.98;S[ 17 ] = 1.98;S[ 18 ] = 1.98;

	for( int i = 0; i < NX; ++i )
		for( int j = 0; j < NY; ++j )
			for( int k = 0; k < NZ; ++k ) {

				rho[ k * NX * NY + j * NX + i ] = rho0;

				ux[ k * NX * NY + j * NX + i ] = 0;		ux0[ k * NX * NY + j * NX + i ] = 0;
				uy[ k * NX * NY + j * NX + i ] = 0;		uy0[ k * NX * NY + j * NX + i ] = 0;
				uz[ k * NX * NY + j * NX + i ] = 0;		uz0[ k * NX * NY + j * NX + i ] = 0;

				ux[(NZ - 1) * NX * NY + j * NX + i] = U;

				euler_xforce[ k * NX * NY + j * NX + i ] = 0;
				euler_yforce[ k * NX * NY + j * NX + i ] = 0;
				euler_zforce[ k * NX * NY + j * NX + i ] = 0;

				f0[ k * NX * NY + j * NX + i ] = feq( 0, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f1[ k * NX * NY + j * NX + i ] = feq( 1, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f2[ k * NX * NY + j * NX + i ] = feq( 2, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f3[ k * NX * NY + j * NX + i ] = feq( 3, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f4[ k * NX * NY + j * NX + i ] = feq( 4, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f5[ k * NX * NY + j * NX + i ] = feq( 5, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f6[ k * NX * NY + j * NX + i ] = feq( 6, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f7[ k * NX * NY + j * NX + i ] = feq( 7, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f8[ k * NX * NY + j * NX + i ] = feq( 8, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f9[ k * NX * NY + j * NX + i ] = feq( 9, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f10[ k * NX * NY + j * NX + i ] = feq( 10, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f11[ k * NX * NY + j * NX + i ] = feq( 11, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f12[ k * NX * NY + j * NX + i ] = feq( 12, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f13[ k * NX * NY + j * NX + i ] = feq( 13, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f14[ k * NX * NY + j * NX + i ] = feq( 14, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f15[ k * NX * NY + j * NX + i ] = feq( 15, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f16[ k * NX * NY + j * NX + i ] = feq( 16, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f17[ k * NX * NY + j * NX + i ] = feq( 17, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );
				f18[ k * NX * NY + j * NX + i ] = feq( 18, rho[ k * NX * NY + j * NX + i ], ux[ k * NX * NY + j * NX + i ], uy[ k * NX * NY + j * NX + i ], uz[ k * NX * NY + j * NX + i ] );

				flag[ k * NX * NY + j * NX + i ] = 'F';
			}
//------------------------------------------

	for( int i = 1; i < NX - 1; ++i )
		for( int j = 1; j < NY - 1; ++j ) {

			flag[ j * NX + i ] = 'B';
			flag[ ( NZ - 1 ) * NX * NY + j * NX + i ] = 'T';
		}

	for( int i = 1; i < NX - 1; ++i )
		for( int k = 1; k < NZ - 1; ++k ) {

			flag[ k * NX * NY + i ] = 'J';
			flag[ k * NX * NY + ( NY - 1 ) * NX + i ] = 'I';
		}

	for( int j = 1; j < NY - 1; ++j )
		for( int k = 1; k < NZ - 1; ++k ) {

			flag[ k * NX * NY + j * NX ] = 'L';
			flag[ k * NX * NY + j * NX + NX - 1 ] = 'R';
		}

	for( int i = 1; i < NX - 1; ++i ) {

		flag[ i ] = 'Q';
		flag[ ( NY - 1 ) * NX + i ] = 'W';
		flag[ ( NZ - 1 ) * NX * NY + ( NY - 1 ) * NX + i ] = 'E';
		flag[ ( NZ - 1 ) * NX * NY + i ] = 'Y';
	}

	for( int j = 1; j < NY - 1; ++j ) {

		flag[ NX * j ] = 'A';
		flag[ NX * j + NX - 1 ] = 'S';
		flag[ ( NZ - 1 ) * NX * NY + NX * j ] = 'D';
		flag[ ( NZ - 1 ) * NX * NY + NX * j + NX - 1 ] = 'G';
	}

	for( int k = 1; k < NZ - 1; ++k ) {

		flag[ k * NX * NY ] = 'Z';
		flag[ k * NX * NY + NX - 1 ] = 'X';
		flag[ k * NX * NY + ( NY - 1 ) * NX ] = 'C';
		flag[ k * NX * NY + ( NY - 1 ) * NX + NX - 1 ] = 'V';
	}

	flag[ 0 ] = 'K';flag[ NX - 1 ] = 'K';flag[ ( NZ - 1 ) * NX * NY + NX - 1 ] = 'K';flag[ ( NZ - 1 ) * NX * NY ] = 'K';
	flag[ ( NY - 1 ) * NX ] = 'K';flag[ ( NY - 1 ) * NX + NX - 1 ] = 'K';flag[ ( NY - 1 ) * NX + ( NZ - 1 ) * NX * NY + NX - 1 ] = 'K';flag[ ( NY - 1 ) * NX + ( NZ - 1 ) * NX * NY ] = 'K';
}

void output( const int m )
{
	ostringstream name;
	name<<"C:\\Users\\Shepherd.DESKTOP-5PQJIS4\\Desktop\\LidDrivenFLow\\Lid3D_MRT\\output\\flow field_"<< m<<".dat";
	ofstream out(name.str().c_str());

	out<<"title=\"cavity flow\"\n"<<"VARIABLES=\"X\",\"Y\",\"Z\",\"U\",\"V\",\"W\",\"Pressure\"\n"<<"ZONE T= \"BOX\",I="<<NX<<",J="<<NY<<",K="<<NZ<<",F=POINT"<<std::endl;

	for( int k = 0; k < NZ; ++k )
		for( int j = 0; j < NY;++j )
			for( int i = 0; i < NX; ++i )
				out<<i<<" "<<j<<" "<<k<<" "<<ux[ k * NX * NY + j * NX + i ]<<" "<<uy[ k * NX * NY + j * NX + i ]<<" "<<uz[ k * NX * NY + j * NX + i ]<<" "<<rho[ k * NX * NY + j * NX + i ] / 3.<<std::endl;
}

double Error( const double * ux, const double * uy, const double * uz, const double * ux0, const double * uy0, const double * uz0 )
{
	double temp1 = 0, temp2 = 0;

	for( int i = 1; i < NX - 1; ++i )
		for( int j = 1; j < NY - 1; ++j )
			for( int k = 1; k < NZ - 1; ++k ) {

				temp1 += ( ( ux[ k * NX * NY + j * NX + i ] - ux0[ k * NX * NY + j * NX + i ] ) * ( ux[ k * NX * NY + j * NX + i ] - ux0[ k * NX * NY + j * NX + i ] )
						 + ( uy[ k * NX * NY + j * NX + i ] - uy0[ k * NX * NY + j * NX + i ] ) * ( uy[ k * NX * NY + j * NX + i ] - uy0[ k * NX * NY + j * NX + i ] )
						 + ( uz[ k * NX * NY + j * NX + i ] - uz0[ k * NX * NY + j * NX + i ] ) * ( uz[ k * NX * NY + j * NX + i ] - uz0[ k * NX * NY + j * NX + i ] ) );
				
				temp2 += ( ux[ k * NX * NY + j * NX + i ] * ux[ k * NX * NY + j * NX + i ]
						 + uy[ k * NX * NY + j * NX + i ] * uy[ k * NX * NY + j * NX + i ]
						 + uz[ k * NX * NY + j * NX + i ] * uz[ k * NX * NY + j * NX + i ] );
			}

	temp1 = sqrt( temp1 );
	temp2 = sqrt( temp2 );

	return temp1 / ( temp2 + 1e-30 );
}

cudaError_t RunLBM()
{
	cudaError_t cudaStatus = cudaSetDevice( 0 );
	if( cudaStatus != cudaSuccess ) {				
		printf( "cudaSetDevice failed !\n" );
		goto Error;
	}

	dim3 threads_lbm( NUM_THREADS_LBM, 1, 1 );
	dim3 grid_lbm(NY, NZ);
	//dim3 grid_lbm( NX / threads_lbm.x, NY / threads_lbm.y, NZ /  threads_lbm.z );

	dim3 threads_col(NUM_THREADS_COL, 1, 1);
	dim3 grid_col(NX / NUM_THREADS_COL, NY, NZ);

	double st = omp_get_wtime();

	for( int n = 0; ; ++n ) {

		LBColl<<< grid_col, threads_col >>>( NX, NY, NZ, flag, S, rho, ux, uy, uz, euler_xforce, euler_yforce, euler_zforce,
											 f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18 );
		LBBC<<< grid_lbm, threads_lbm >>>( NX, NY, NZ, flag, rho, ux, uy, uz, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18 );
		LBProp<<< grid_lbm, threads_lbm >>>( NX, NY, NZ, flag, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18 );
		LBmacro<<< grid_lbm, threads_lbm >>>( NX, NY, NZ, flag, rho, ux, uy, uz, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, ux0, uy0, uz0 );

		cudaDeviceSynchronize();

		if( n % 100 == 0 ) {

			error = Error( ux, uy, uz, ux0, uy0, uz0 );
			cout<<"The"<<n<<"th computation result:"<<endl<<"The u,v,w of point (NX/2,NY/2,NZ/2)is:"<<setprecision(6)<<ux[ NZ / 2 * NX * NY + NX / 2 + NY * NX / 2 ]<<","<<uy[ NZ / 2 * NX * NY + NX / 2 + NY * NX / 2 ]<<","<<uz[ NZ / 2 * NX * NY + NX / 2 + NY * NX / 2 ]<<endl;
			cout<<"The max relative error of uv is :"<<setiosflags(ios::scientific)<<error<<endl;
			cout<<"-----------------------------------------------------\n"<<endl;
			if (n >= 1000)
			{
				if (n % 1000 == 0) output(n);
				if (error < 1.0e-6) break;
			}
		}

		//if( n % 100 == 0 && n != 0 ) output( n );
	}

	double runtime = omp_get_wtime() - st;

	cout << "Total time is " << runtime << endl;

Error:

/***********************************************************************/
/****************Unified Memory Deallocation for BasicVar***************/
/***********************************************************************/
	cudaFree( f0 );		cudaFree( ux );		cudaFree( ux0 );
	cudaFree( f1 );		cudaFree( uy );		cudaFree( uy0 );
	cudaFree( f2 );		cudaFree( uz );		cudaFree( uz0 );
	cudaFree( f3 );
	cudaFree( f4 );		cudaFree( rho );	
	cudaFree( f5 );		cudaFree( euler_xforce );
	cudaFree( f6 );		cudaFree( euler_yforce );
	cudaFree( f7 );		cudaFree( euler_zforce );
	cudaFree( f8 );
	cudaFree( f9 );		cudaFree( S );		cudaFree( flag );
	cudaFree( f10 );	
	cudaFree( f11 );
	cudaFree( f12 );
	cudaFree( f13 );
	cudaFree( f14 );
	cudaFree( f15 );
	cudaFree( f16 );
	cudaFree( f17 );
	cudaFree( f18 );

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