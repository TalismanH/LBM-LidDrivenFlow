
#include "basicVar.cuh"

__device__ double meq( int k, double rho, double ux, double uy, double uz )
{
	double we = 3.0, wej = -11. / 2., wxx = -0.5;
	//double we = 0, wej = -475. / 63., wxx = 0;

	switch( k ) {

		case( 0 ) : return 0; break;
		case( 1 ) : return -11. * rho + 19. * ( ux * ux + uy * uy + uz * uz ); break;
		case( 2 ) : return we * rho + wej * ( ux * ux + uy * uy + uz * uz ); break;
		case( 3 ) : return 0; break;
		case( 4 ) : return -2. * ux / 3.; break;
		case( 5 ) : return 0; break;
		case( 6 ) : return -2. * uy / 3.; break;
		case( 7 ) : return 0; break;
		case( 8 ) : return -2. * uz / 3.; break;
		case( 9 ) : return 2. * ux * ux - uy * uy - uz * uz; break;
		case( 10 ) : return wxx * ( 2. * ux * ux - uy * uy - uz * uz ); break;
		case( 11 ) : return uy * uy - uz * uz; break;
		case( 12 ) : return wxx * ( uy * uy - uz * uz ); break;
		case( 13 ) : return ux * uy; break;
		case( 14 ) : return uy * uz; break;
		case( 15 ) : return ux * uz; break;
		case( 16 ) : return 0; break;
		case( 17 ) : return 0; break;
		case( 18 ) : return 0; break;
	}
}

__device__ double mForceGenerator( int k, double ux, double uy, double uz, double fx, double fy, double fz )
{
	switch( k ) {

		case( 0 ) : return 0; break;
		case( 1 ) : return 38.0 * ( ux * fx + uy * fy + uz * fz ); break;
		case( 2 ) : return -11.0 *( ux * fx + uy * fy + uz * fz ); break;
		case( 3 ) : return fx; break;
		case( 4 ) : return -2. / 3. * fx; break;
		case( 5 ) : return fy; break;
		case( 6 ) : return -2. / 3. * fy; break;
		case( 7 ) : return fz; break;
		case( 8 ) : return -2. / 3. * fz; break;
		case( 9 ) : return 4. * ux * fx - 2. * uy * fy - 2. * uz * fz; break;
		case( 10 ) : return -2 * ux * fx + uy * fy + uz * fz; break;
		case( 11 ) : return 2. * uy * fy - 2. * uz * fz; break;
		case( 12 ) : return -uy * fy + uz * fz; break;
		case( 13 ) : return ux * fy + uy * fx; break;
		case( 14 ) : return uz * fy + uy * fz; break;
		case( 15 ) : return ux * fz + uz * fx; break;
		case( 16 ) : return 0; break;
		case( 17 ) : return 0; break;
		case( 18 ) : return 0; break;
	}
}

__device__ __host__ double feq( int k, double rho, double ux, double uy, double uz )
{
	int e[ 19 ][ 3 ] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 }, { 1, 1, 0 }, { -1, -1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { 1, 0, 1 }, { -1, 0, -1 }, { 1, 0, -1 }, { -1, 0, 1 }, { 0, 1, 1 }, { 0, -1, -1 }, { 0, 1, -1 }, { 0, -1, 1 } };
	double w[ 19 ] = { 1. / 3, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 18, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36, 1. / 36 };

	double eu = e[ k ][ 0 ] * ux + e[ k ][ 1 ] * uy + e[ k ][ 2 ] * uz;
	double uvw = ux * ux + uy * uy + uz * uz;
	
	return w[ k ] * ( rho + 3. * eu + 4.5 * eu * eu - 1.5 * uvw );
}

__device__ inline void Swap( double & f1, double & f2 )
{
	double temp = f1;
	f1 = f2;
	f2 = temp;
}

__global__ void LBColl( int nx, int ny, int nz, int * flag, double * S, double * rho, double * ux, double * uy, double * uz, double * euler_xforce, double * euler_yforce, double * euler_zforce,
					    double* f0,double* f1,double* f2,double* f3,double* f4,double* f5,double* f6,double* f7,double* f8,
					    double* f9,double* f10,double* f11,double* f12,double* f13,double* f14,double* f15,double* f16,double* f17,double* f18 )
{
	// 将线程网格组织成二维的形式，而不是三维形式
	// 计算效率会比三维形式更高？

	//int tx = threadIdx.x;
	//int k = blockIdx.y * nx * ny + blockIdx.x * nx + tx;

	int tx = threadIdx.x;
	int k = blockIdx.z * nx * ny + blockIdx.y * nx + tx + blockIdx.x * NUM_THREADS_COL;

	__shared__ double F0 [ NUM_THREADS_COL ];		__shared__ double MF0 [ NUM_THREADS_COL ];
	__shared__ double F1 [ NUM_THREADS_COL ];		__shared__ double MF1 [ NUM_THREADS_COL ];
	__shared__ double F2 [ NUM_THREADS_COL ];		__shared__ double MF2 [ NUM_THREADS_COL ];
	__shared__ double F3 [ NUM_THREADS_COL ];		__shared__ double MF3 [ NUM_THREADS_COL ];
	__shared__ double F4 [ NUM_THREADS_COL ];		__shared__ double MF4 [ NUM_THREADS_COL ];
	__shared__ double F5 [ NUM_THREADS_COL ];		__shared__ double MF5 [ NUM_THREADS_COL ];
	__shared__ double F6 [ NUM_THREADS_COL ];		__shared__ double MF6 [ NUM_THREADS_COL ];
	__shared__ double F7 [ NUM_THREADS_COL ];		__shared__ double MF7 [ NUM_THREADS_COL ];
	__shared__ double F8 [ NUM_THREADS_COL ];		__shared__ double MF8 [ NUM_THREADS_COL ];
	__shared__ double F9 [ NUM_THREADS_COL ];		__shared__ double MF9 [ NUM_THREADS_COL ];
	__shared__ double F10[ NUM_THREADS_COL ];		__shared__ double MF10[ NUM_THREADS_COL ];
	__shared__ double F11[ NUM_THREADS_COL ];		__shared__ double MF11[ NUM_THREADS_COL ];
	__shared__ double F12[ NUM_THREADS_COL ];		__shared__ double MF12[ NUM_THREADS_COL ];
	__shared__ double F13[ NUM_THREADS_COL ];		__shared__ double MF13[ NUM_THREADS_COL ];
	__shared__ double F14[ NUM_THREADS_COL ];		__shared__ double MF14[ NUM_THREADS_COL ];
	__shared__ double F15[ NUM_THREADS_COL ];		__shared__ double MF15[ NUM_THREADS_COL ];
	__shared__ double F16[ NUM_THREADS_COL ];		__shared__ double MF16[ NUM_THREADS_COL ];
	__shared__ double F17[ NUM_THREADS_COL ];		__shared__ double MF17[ NUM_THREADS_COL ];
	__shared__ double F18[ NUM_THREADS_COL ];		__shared__ double MF18[ NUM_THREADS_COL ];

	double _rho;	_rho = rho[ k ];
	double _ux;		_ux = ux[ k ];
	double _uy;		_uy = uy[ k ];
	double _uz;		_uz = uz[ k ];
	double EXF;		EXF = euler_xforce[ k ];
	double EYF;		EYF = euler_yforce[ k ];
	double EZF;		EZF = euler_zforce[ k ];

	F0 [ tx ] = f0 [ k ];
	F1 [ tx ] = f1 [ k ];
	F2 [ tx ] = f2 [ k ];
	F3 [ tx ] = f3 [ k ];
	F4 [ tx ] = f4 [ k ];
	F5 [ tx ] = f5 [ k ];
	F6 [ tx ] = f6 [ k ];
	F7 [ tx ] = f7 [ k ];
	F8 [ tx ] = f8 [ k ];
	F9 [ tx ] = f9 [ k ];
	F10[ tx ] = f10[ k ];
	F11[ tx ] = f11[ k ];
	F12[ tx ] = f12[ k ];
	F13[ tx ] = f13[ k ];
	F14[ tx ] = f14[ k ];
	F15[ tx ] = f15[ k ];
	F16[ tx ] = f16[ k ];
	F17[ tx ] = f17[ k ];
	F18[ tx ] = f18[ k ];

	__syncthreads();

	if( flag[ k ] == 'F' ) {

		MF0 [ tx ] = F0[ tx ] + F1[ tx ] + F2[ tx ] + F3[ tx ] + F4[ tx ] + F5[ tx ] + F6[ tx ] + F7[ tx ] + F8[ tx ] + F9[ tx ] + F10[ tx ] + F11[ tx ] + F12[ tx ] + F13[ tx ] + F14[ tx ] + F15[ tx ] + F16[ tx ] + F17[ tx ] + F18[ tx ];
		MF1 [ tx ] =  - 30. * F0[ tx ] - 11. * F1[ tx ] - 11. * F2[ tx ] - 11. * F3[ tx ] - 11. * F4[ tx ] - 11. * F5[ tx ] - 11. * F6[ tx ] + 8. * F7[ tx ] + 8. * F8[ tx ] + 8. * F9[ tx ] + 8. * F10[ tx ] + 8. * F11[ tx ] + 8. * F12[ tx ] + 8. * F13[ tx ] + 8. * F14[ tx ] + 8. * F15[ tx ] + 8. * F16[ tx ] + 8. * F17[ tx ] + 8. * F18[ tx ];
		MF2 [ tx ] = 12. * F0[ tx ] - 4. * F1[ tx ] - 4. * F2[ tx ] - 4. * F3[ tx ] - 4. * F4[ tx ] - 4. * F5[ tx ] - 4. * F6[ tx ] + F7[ tx ] + F8[ tx ] + F9[ tx ] + F10[ tx ] + F11[ tx ] + F12[ tx ] + F13[ tx ] + F14[ tx ] + F15[ tx ] + F16[ tx ] + F17[ tx ] + F18[ tx ];
		MF3 [ tx ] = F1[ tx ] - F2[ tx ] + F7[ tx ] - F8[ tx ] + F9[ tx ] - F10[ tx ] + F11[ tx ] - F12[ tx ] + F13[ tx ] - F14[ tx ];
		MF4 [ tx ] =  - 4. * F1[ tx ] + 4. * F2[ tx ] + F7[ tx ] - F8[ tx ] + F9[ tx ] - F10[ tx ] + F11[ tx ] - F12[ tx ] + F13[ tx ] - F14[ tx ];
		MF5 [ tx ] = F3[ tx ] - F4[ tx ] + F7[ tx ] - F8[ tx ] - F9[ tx ] + F10[ tx ] + F15[ tx ] - F16[ tx ] + F17[ tx ] - F18[ tx ];
		MF6 [ tx ] =  - 4. * F3[ tx ] + 4. * F4[ tx ] + F7[ tx ] - F8[ tx ] - F9[ tx ] + F10[ tx ] + F15[ tx ] - F16[ tx ] + F17[ tx ] - F18[ tx ];
		MF7 [ tx ] = F5[ tx ] - F6[ tx ] + F11[ tx ] - F12[ tx ] - F13[ tx ] + F14[ tx ] + F15[ tx ] - F16[ tx ] - F17[ tx ] + F18[ tx ];
		MF8 [ tx ] =  - 4. * F5[ tx ] + 4. * F6[ tx ] + F11[ tx ] - F12[ tx ] - F13[ tx ] + F14[ tx ] + F15[ tx ] - F16[ tx ] - F17[ tx ] + F18[ tx ];
		MF9 [ tx ] = 2. * F1[ tx ] + 2. * F2[ tx ] - F3[ tx ] - F4[ tx ] - F5[ tx ] - F6[ tx ] + F7[ tx ] + F8[ tx ] + F9[ tx ] + F10[ tx ] + F11[ tx ] + F12[ tx ] + F13[ tx ] + F14[ tx ] - 2. * F15[ tx ] - 2. * F16[ tx ] - 2. * F17[ tx ] - 2. * F18[ tx ];
		MF10[ tx ] =  - 4. * F1[ tx ] - 4. * F2[ tx ] + 2. * F3[ tx ] + 2. * F4[ tx ] + 2. * F5[ tx ] + 2. * F6[ tx ] + F7[ tx ] + F8[ tx ] + F9[ tx ] + F10[ tx ] + F11[ tx ] + F12[ tx ] + F13[ tx ] + F14[ tx ] - 2. * F15[ tx ] - 2. * F16[ tx ] - 2. * F17[ tx ] - 2. * F18[ tx ];
		MF11[ tx ] = F3[ tx ] + F4[ tx ] - F5[ tx ] - F6[ tx ] + F7[ tx ] + F8[ tx ] + F9[ tx ] + F10[ tx ] - F11[ tx ] - F12[ tx ] - F13[ tx ] - F14[ tx ];
		MF12[ tx ] =  - 2. * F3[ tx ] - 2. * F4[ tx ] + 2. * F5[ tx ] + 2. * F6[ tx ] + F7[ tx ] + F8[ tx ] + F9[ tx ] + F10[ tx ] - F11[ tx ] - F12[ tx ] - F13[ tx ] - F14[ tx ];
		MF13[ tx ] = F7[ tx ] + F8[ tx ] - F9[ tx ] - F10[ tx ];
		MF14[ tx ] = F15[ tx ] + F16[ tx ] - F17[ tx ] - F18[ tx ];
		MF15[ tx ] = F11[ tx ] + F12[ tx ] - F13[ tx ] - F14[ tx ];
		MF16[ tx ] = F7[ tx ] - F8[ tx ] + F9[ tx ] - F10[ tx ] - F11[ tx ] + F12[ tx ] - F13[ tx ] + F14[ tx ];
		MF17[ tx ] =  - F7[ tx ] + F8[ tx ] + F9[ tx ] - F10[ tx ] + F15[ tx ] - F16[ tx ] + F17[ tx ] - F18[ tx ];
		MF18[ tx ] = F11[ tx ] - F12[ tx ] - F13[ tx ] + F14[ tx ] - F15[ tx ] + F16[ tx ] + F17[ tx ] - F18[ tx ];

		MF0 [ tx ] = MF0 [ tx ] + S[ 0  ] * ( meq( 0 , _rho, _ux, _uy, _uz ) - MF0 [ tx ] ) + ( 1 - 0.5f * S[ 0  ] )	* mForceGenerator( 0 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF1 [ tx ] = MF1 [ tx ] + S[ 1  ] * ( meq( 1 , _rho, _ux, _uy, _uz ) - MF1 [ tx ] ) + ( 1 - 0.5f * S[ 1  ] )	* mForceGenerator( 1 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF2 [ tx ] = MF2 [ tx ] + S[ 2  ] * ( meq( 2 , _rho, _ux, _uy, _uz ) - MF2 [ tx ] ) + ( 1 - 0.5f * S[ 2  ] )	* mForceGenerator( 2 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF3 [ tx ] = MF3 [ tx ] + S[ 3  ] * ( meq( 3 , _rho, _ux, _uy, _uz ) - MF3 [ tx ] ) + ( 1 - 0.5f * S[ 3  ] )	* mForceGenerator( 3 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF4 [ tx ] = MF4 [ tx ] + S[ 4  ] * ( meq( 4 , _rho, _ux, _uy, _uz ) - MF4 [ tx ] ) + ( 1 - 0.5f * S[ 4  ] )	* mForceGenerator( 4 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF5 [ tx ] = MF5 [ tx ] + S[ 5  ] * ( meq( 5 , _rho, _ux, _uy, _uz ) - MF5 [ tx ] ) + ( 1 - 0.5f * S[ 5  ] )	* mForceGenerator( 5 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF6 [ tx ] = MF6 [ tx ] + S[ 6  ] * ( meq( 6 , _rho, _ux, _uy, _uz ) - MF6 [ tx ] ) + ( 1 - 0.5f * S[ 6  ] )	* mForceGenerator( 6 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF7 [ tx ] = MF7 [ tx ] + S[ 7  ] * ( meq( 7 , _rho, _ux, _uy, _uz ) - MF7 [ tx ] ) + ( 1 - 0.5f * S[ 7  ] )	* mForceGenerator( 7 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF8 [ tx ] = MF8 [ tx ] + S[ 8  ] * ( meq( 8 , _rho, _ux, _uy, _uz ) - MF8 [ tx ] ) + ( 1 - 0.5f * S[ 8  ] )	* mForceGenerator( 8 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF9 [ tx ] = MF9 [ tx ] + S[ 9  ] * ( meq( 9 , _rho, _ux, _uy, _uz ) - MF9 [ tx ] ) + ( 1 - 0.5f * S[ 9  ] )	* mForceGenerator( 9 , _ux, _uy, _uz, EXF, EYF, EZF );
		MF10[ tx ] = MF10[ tx ] + S[ 10 ] * ( meq( 10, _rho, _ux, _uy, _uz ) - MF10[ tx ] ) + ( 1 - 0.5f * S[ 10 ] )	* mForceGenerator( 10, _ux, _uy, _uz, EXF, EYF, EZF );
		MF11[ tx ] = MF11[ tx ] + S[ 11 ] * ( meq( 11, _rho, _ux, _uy, _uz ) - MF11[ tx ] ) + ( 1 - 0.5f * S[ 11 ] )	* mForceGenerator( 11, _ux, _uy, _uz, EXF, EYF, EZF );
		MF12[ tx ] = MF12[ tx ] + S[ 12 ] * ( meq( 12, _rho, _ux, _uy, _uz ) - MF12[ tx ] ) + ( 1 - 0.5f * S[ 12 ] )	* mForceGenerator( 12, _ux, _uy, _uz, EXF, EYF, EZF );
		MF13[ tx ] = MF13[ tx ] + S[ 13 ] * ( meq( 13, _rho, _ux, _uy, _uz ) - MF13[ tx ] ) + ( 1 - 0.5f * S[ 13 ] )	* mForceGenerator( 13, _ux, _uy, _uz, EXF, EYF, EZF );
		MF14[ tx ] = MF14[ tx ] + S[ 14 ] * ( meq( 14, _rho, _ux, _uy, _uz ) - MF14[ tx ] ) + ( 1 - 0.5f * S[ 14 ] )	* mForceGenerator( 14, _ux, _uy, _uz, EXF, EYF, EZF );
		MF15[ tx ] = MF15[ tx ] + S[ 15 ] * ( meq( 15, _rho, _ux, _uy, _uz ) - MF15[ tx ] ) + ( 1 - 0.5f * S[ 15 ] )	* mForceGenerator( 15, _ux, _uy, _uz, EXF, EYF, EZF );
		MF16[ tx ] = MF16[ tx ] + S[ 16 ] * ( meq( 16, _rho, _ux, _uy, _uz ) - MF16[ tx ] ) + ( 1 - 0.5f * S[ 16 ] )	* mForceGenerator( 16, _ux, _uy, _uz, EXF, EYF, EZF );
		MF17[ tx ] = MF17[ tx ] + S[ 17 ] * ( meq( 17, _rho, _ux, _uy, _uz ) - MF17[ tx ] ) + ( 1 - 0.5f * S[ 17 ] )	* mForceGenerator( 17, _ux, _uy, _uz, EXF, EYF, EZF );
		MF18[ tx ] = MF18[ tx ] + S[ 18 ] * ( meq( 18, _rho, _ux, _uy, _uz ) - MF18[ tx ] ) + ( 1 - 0.5f * S[ 18 ] )	* mForceGenerator( 18, _ux, _uy, _uz, EXF, EYF, EZF );

		f0 [ k ] = MF0[ tx ] / 19. - 5. * MF1[ tx ] / 399. + MF2[ tx ] / 21.;
		f2 [ k ] = MF0[ tx ] / 19. - 11. * MF1[ tx ] / 2394. - MF2[ tx ] / 63. + MF3[ tx ] / 10. - MF4[ tx ] / 10. + MF9[ tx ] / 18. - MF10[ tx ] / 18.;
		f1 [ k ] = MF0[ tx ] / 19. - 11. * MF1[ tx ] / 2394. - MF2[ tx ] / 63. - MF3[ tx ] / 10. + MF4[ tx ] / 10. + MF9[ tx ] / 18. - MF10[ tx ] / 18.;
		f4 [ k ] = MF0[ tx ] / 19. - 11. * MF1[ tx ] / 2394. - MF2[ tx ] / 63. + MF5[ tx ] / 10. - MF6[ tx ] / 10. - MF9[ tx ] / 36. + MF10[ tx ] / 36. + MF11[ tx ] / 12. - MF12[ tx ] / 12.;
		f3 [ k ] = MF0[ tx ] / 19. - 11. * MF1[ tx ] / 2394. - MF2[ tx ] / 63. - MF5[ tx ] / 10. + MF6[ tx ] / 10. - MF9[ tx ] / 36. + MF10[ tx ] / 36. + MF11[ tx ] / 12. - MF12[ tx ] / 12.;
		f6 [ k ] = MF0[ tx ] / 19. - 11. * MF1[ tx ] / 2394. - MF2[ tx ] / 63. + MF7[ tx ] / 10. - MF8[ tx ] / 10. - MF9[ tx ] / 36. + MF10[ tx ] / 36. - MF11[ tx ] / 12. + MF12[ tx ] / 12.;
		f5 [ k ] = MF0[ tx ] / 19. - 11. * MF1[ tx ] / 2394. - MF2[ tx ] / 63. - MF7[ tx ] / 10. + MF8[ tx ] / 10. - MF9[ tx ] / 36. + MF10[ tx ] / 36. - MF11[ tx ] / 12. + MF12[ tx ] / 12.;
		f8 [ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. + MF3[ tx ] / 10. + MF4[ tx ] / 40. + MF5[ tx ] / 10. + MF6[ tx ] / 40. + MF9[ tx ] / 36. + MF10[ tx ] / 72. + MF11[ tx ] / 12. + MF12[ tx ] / 24. + MF13[ tx ] / 4. + MF16[ tx ] / 8. - MF17[ tx ] / 8.;
		f7 [ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. - MF3[ tx ] / 10. - MF4[ tx ] / 40. - MF5[ tx ] / 10. - MF6[ tx ] / 40. + MF9[ tx ] / 36. + MF10[ tx ] / 72. + MF11[ tx ] / 12. + MF12[ tx ] / 24. + MF13[ tx ] / 4. - MF16[ tx ] / 8. + MF17[ tx ] / 8.;
		f10[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. + MF3[ tx ] / 10. + MF4[ tx ] / 40. - MF5[ tx ] / 10. - MF6[ tx ] / 40. + MF9[ tx ] / 36. + MF10[ tx ] / 72. + MF11[ tx ] / 12. + MF12[ tx ] / 24. - MF13[ tx ] / 4. + MF16[ tx ] / 8. + MF17[ tx ] / 8.;
		f9 [ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. - MF3[ tx ] / 10. - MF4[ tx ] / 40. + MF5[ tx ] / 10. + MF6[ tx ] / 40. + MF9[ tx ] / 36. + MF10[ tx ] / 72. + MF11[ tx ] / 12. + MF12[ tx ] / 24. - MF13[ tx ] / 4. - MF16[ tx ] / 8. - MF17[ tx ] / 8.;
		f12[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. + MF3[ tx ] / 10. + MF4[ tx ] / 40. + MF7[ tx ] / 10. + MF8[ tx ] / 40. + MF9[ tx ] / 36. + MF10[ tx ] / 72. - MF11[ tx ] / 12. - MF12[ tx ] / 24. + MF15[ tx ] / 4. - MF16[ tx ] / 8. + MF18[ tx ] / 8.;
		f11[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. - MF3[ tx ] / 10. - MF4[ tx ] / 40. - MF7[ tx ] / 10. - MF8[ tx ] / 40. + MF9[ tx ] / 36. + MF10[ tx ] / 72. - MF11[ tx ] / 12. - MF12[ tx ] / 24. + MF15[ tx ] / 4. + MF16[ tx ] / 8. - MF18[ tx ] / 8.;
		f14[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. + MF3[ tx ] / 10. + MF4[ tx ] / 40. - MF7[ tx ] / 10. - MF8[ tx ] / 40. + MF9[ tx ] / 36. + MF10[ tx ] / 72. - MF11[ tx ] / 12. - MF12[ tx ] / 24. - MF15[ tx ] / 4. - MF16[ tx ] / 8. - MF18[ tx ] / 8.;
		f13[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. - MF3[ tx ] / 10. - MF4[ tx ] / 40. + MF7[ tx ] / 10. + MF8[ tx ] / 40. + MF9[ tx ] / 36. + MF10[ tx ] / 72. - MF11[ tx ] / 12. - MF12[ tx ] / 24. - MF15[ tx ] / 4. + MF16[ tx ] / 8. + MF18[ tx ] / 8.;
		f16[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. + MF5[ tx ] / 10. + MF6[ tx ] / 40. + MF7[ tx ] / 10. + MF8[ tx ] / 40. - MF9[ tx ] / 18. - MF10[ tx ] / 36. + MF14[ tx ] / 4. + MF17[ tx ] / 8. - MF18[ tx ] / 8.;
		f15[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. - MF5[ tx ] / 10. - MF6[ tx ] / 40. - MF7[ tx ] / 10. - MF8[ tx ] / 40. - MF9[ tx ] / 18. - MF10[ tx ] / 36. + MF14[ tx ] / 4. - MF17[ tx ] / 8. + MF18[ tx ] / 8.;
		f18[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. + MF5[ tx ] / 10. + MF6[ tx ] / 40. - MF7[ tx ] / 10. - MF8[ tx ] / 40. - MF9[ tx ] / 18. - MF10[ tx ] / 36. - MF14[ tx ] / 4. + MF17[ tx ] / 8. + MF18[ tx ] / 8.;
		f17[ k ] = MF0[ tx ] / 19. + 4. * MF1[ tx ] / 1197. + MF2[ tx ] / 252. - MF5[ tx ] / 10. - MF6[ tx ] / 40. + MF7[ tx ] / 10. + MF8[ tx ] / 40. - MF9[ tx ] / 18. - MF10[ tx ] / 36. - MF14[ tx ] / 4. - MF17[ tx ] / 8. - MF18[ tx ] / 8.;
	}
}

__global__ void LBProp( int nx, int ny, int nz, int * flag,
					    double * f0, double * f1, double * f2, double * f3, double * f4, double * f5, double * f6, double * f7, double * f8,
					    double * f9, double * f10, double * f11, double * f12, double * f13, double * f14, double * f15, double * f16, double * f17, double * f18 )
{
	int tx = threadIdx.x;
	int k = blockIdx.y * nx * ny + blockIdx.x * nx + tx;

	if( flag[ k ] == 'F' ) {

		Swap( f1 [ k + 1 ],				f2 [ k ] );
		Swap( f3 [ k + nx ],			f4 [ k ] );
		Swap( f5 [ k + nx * ny ],		f6 [ k ] );
		Swap( f7 [ k + nx + 1 ],		f8 [ k ] );
		Swap( f9 [ k - nx + 1 ],		f10[ k ] );
		Swap( f11[ k + nx * ny + 1 ],	f12[ k ] );
		Swap( f13[ k - nx * ny + 1 ],	f14[ k ] );
		Swap( f15[ k + nx * ny + nx ],	f16[ k ] );
		Swap( f17[ k - nx * ny + nx ],	f18[ k ] );
	}

	if( flag[ k ] == 'B' ) {

		Swap( f5 [ k + nx * ny ],		f6 [ k ] );
		Swap( f11[ k + nx * ny + 1 ],	f12[ k ] );
		Swap( f15[ k + nx * ny + nx ],	f16[ k ] );
	}

	if( flag[ k ] == 'T' ) {

		Swap( f13[ k - nx * ny + 1 ],	f14[ k ] );
		Swap( f17[ k - nx * ny + nx ],	f18[ k ] );
	}

	if( flag[ k ] == 'J' ) {

		Swap( f3 [ k + nx ],			f4 [ k ] );
		Swap( f7 [ k + nx + 1 ],		f8 [ k ] );
		Swap( f15[ k + nx * ny + nx ],	f16[ k ] );
		Swap( f17[ k - nx * ny + nx ],	f18[ k ] );
	}

	if( flag[ k ] == 'I' ) {

		Swap( f9 [ k - nx + 1 ],		f10[ k ] );
	}

	if( flag[ k ] == 'L' ) {

		Swap( f1 [ k + 1 ],				f2 [ k ] );
		Swap( f7 [ k + nx + 1 ],		f8 [ k ] );
		Swap( f9 [ k - nx + 1 ],		f10[ k ] );
		Swap( f11[ k + nx * ny + 1 ],	f12[ k ] );
		Swap( f13[ k - nx * ny + 1 ],	f14[ k ] );
	}

	if( flag[ k ] == 'Q' ) Swap( f15[ k + nx * ny + nx ],	f16[ k ] );
	if( flag[ k ] == 'Y' ) Swap( f17[ k - nx * ny + nx ],	f18[ k ] );
	if( flag[ k ] == 'A' ) Swap( f11[ k + nx * ny + 1 ],	f12[ k ] );
	if( flag[ k ] == 'D' ) Swap( f13[ k - nx * ny + 1 ],	f14[ k ] );
	if( flag[ k ] == 'Z' ) Swap( f7 [ k + nx + 1 ],			f8 [ k ] );
	if( flag[ k ] == 'C' ) Swap( f9 [ k - nx + 1 ],			f10[ k ] );
}



__global__ void LBBC(int nx, int ny, int nz, int* flag, double* rho, double* ux, double* uy, double* uz,
	double* f0, double* f1, double* f2, double* f3, double* f4, double* f5, double* f6, double* f7, double* f8,
	double* f9, double* f10, double* f11, double* f12, double* f13, double* f14, double* f15, double* f16, double* f17, double* f18)
{
	int tx = threadIdx.x;
	int k = blockIdx.y * nx * ny + blockIdx.x * nx + tx;

	double _rho;
	double _ux;
	double _uy;
	double _uz;

	if (flag[k] == 'B') {

		_rho = rho[k + nx * ny];
		_ux = ux[k + nx * ny];
		_uy = uy[k + nx * ny];
		_uz = uz[k + nx * ny];

		rho[k] = _rho;

		f6[k] = feq(5, _rho, 0, 0, 0) + f6[k + nx * ny] - feq(5, _rho, _ux, _uy, _uz);
		f12[k] = feq(11, _rho, 0, 0, 0) + f12[k + nx * ny] - feq(11, _rho, _ux, _uy, _uz);
		f13[k] = feq(14, _rho, 0, 0, 0) + f13[k + nx * ny] - feq(14, _rho, _ux, _uy, _uz);
		f16[k] = feq(15, _rho, 0, 0, 0) + f16[k + nx * ny] - feq(15, _rho, _ux, _uy, _uz);
		f17[k] = feq(18, _rho, 0, 0, 0) + f17[k + nx * ny] - feq(18, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'T') {

		_rho = rho[k - nx * ny];
		_ux = ux[k - nx * ny];
		_uy = uy[k - nx * ny];
		_uz = uz[k - nx * ny];

		rho[k] = _rho;
		ux[k] = 0.1;

		f5[k] = feq(6, _rho, 0.1, 0, 0) + f5[k - nx * ny] - feq(6, _rho, _ux, _uy, _uz);
		f14[k] = feq(13, _rho, 0.1, 0, 0) + f14[k - nx * ny] - feq(13, _rho, _ux, _uy, _uz);
		f11[k] = feq(12, _rho, 0.1, 0, 0) + f11[k - nx * ny] - feq(12, _rho, _ux, _uy, _uz);
		f18[k] = feq(17, _rho, 0.1, 0, 0) + f18[k - nx * ny] - feq(17, _rho, _ux, _uy, _uz);
		f15[k] = feq(16, _rho, 0.1, 0, 0) + f15[k - nx * ny] - feq(16, _rho, _ux, _uy, _uz);

		//f5 [ k ] = f5 [ k - nx * ny ];
		//f14[ k ] = f14[ k - nx * ny ];
		//f11[ k ] = f11[ k - nx * ny ];
		//f18[ k ] = f18[ k - nx * ny ];
		//f15[ k ] = f15[ k - nx * ny ];
	}

	if (flag[k] == 'J') {

		_rho = rho[k + nx];
		_ux = ux[k + nx];
		_uy = uy[k + nx];
		_uz = uz[k + nx];

		rho[k] = _rho;

		f4[k] = feq(3, _rho, 0, 0, 0) + f4[k + nx] - feq(3, _rho, _ux, _uy, _uz);
		f8[k] = feq(7, _rho, 0, 0, 0) + f8[k + nx] - feq(7, _rho, _ux, _uy, _uz);
		f9[k] = feq(10, _rho, 0, 0, 0) + f9[k + nx] - feq(10, _rho, _ux, _uy, _uz);
		f16[k] = feq(15, _rho, 0, 0, 0) + f16[k + nx] - feq(15, _rho, _ux, _uy, _uz);
		f18[k] = feq(17, _rho, 0, 0, 0) + f18[k + nx] - feq(17, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'I') {

		_rho = rho[k - nx];
		_ux = ux[k - nx];
		_uy = uy[k - nx];
		_uz = uz[k - nx];

		rho[k] = _rho;

		f3[k] = feq(4, _rho, 0, 0, 0) + f3[k - nx] - feq(4, _rho, _ux, _uy, _uz);
		f10[k] = feq(9, _rho, 0, 0, 0) + f10[k - nx] - feq(9, _rho, _ux, _uy, _uz);
		f7[k] = feq(8, _rho, 0, 0, 0) + f7[k - nx] - feq(8, _rho, _ux, _uy, _uz);
		f17[k] = feq(18, _rho, 0, 0, 0) + f17[k - nx] - feq(18, _rho, _ux, _uy, _uz);
		f15[k] = feq(16, _rho, 0, 0, 0) + f15[k - nx] - feq(16, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'L') {

		_rho = rho[k + 1];
		_ux = ux[k + 1];
		_uy = uy[k + 1];
		_uz = uz[k + 1];

		rho[k] = _rho;

		f2[k] = feq(1, _rho, 0, 0, 0) + f2[k + 1] - feq(1, _rho, _ux, _uy, _uz);
		f8[k] = feq(7, _rho, 0, 0, 0) + f8[k + 1] - feq(7, _rho, _ux, _uy, _uz);
		f10[k] = feq(9, _rho, 0, 0, 0) + f10[k + 1] - feq(9, _rho, _ux, _uy, _uz);
		f12[k] = feq(11, _rho, 0, 0, 0) + f12[k + 1] - feq(11, _rho, _ux, _uy, _uz);
		f14[k] = feq(13, _rho, 0, 0, 0) + f14[k + 1] - feq(13, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'R') {

		_rho = rho[k - 1];
		_ux = ux[k - 1];
		_uy = uy[k - 1];
		_uz = uz[k - 1];

		rho[k] = _rho;

		f1[k] = feq(2, _rho, 0, 0, 0) + f1[k - 1] - feq(2, _rho, _ux, _uy, _uz);
		f7[k] = feq(8, _rho, 0, 0, 0) + f7[k - 1] - feq(8, _rho, _ux, _uy, _uz);
		f9[k] = feq(10, _rho, 0, 0, 0) + f9[k - 1] - feq(10, _rho, _ux, _uy, _uz);
		f11[k] = feq(12, _rho, 0, 0, 0) + f11[k - 1] - feq(12, _rho, _ux, _uy, _uz);
		f13[k] = feq(14, _rho, 0, 0, 0) + f13[k - 1] - feq(14, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'Q') {

		_rho = rho[k + nx * ny + nx];
		_ux = ux[k + nx * ny + nx];
		_uy = uy[k + nx * ny + nx];
		_uz = uz[k + nx * ny + nx];

		rho[k] = _rho;

		f16[k] = feq(15, _rho, 0, 0, 0) + f16[k + nx * ny + nx] - feq(15, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'W') {

		_rho = rho[k + nx * ny - nx];
		_ux = ux[k + nx * ny - nx];
		_uy = uy[k + nx * ny - nx];
		_uz = uz[k + nx * ny - nx];

		rho[k] = _rho;

		f17[k] = feq(18, _rho, 0, 0, 0) + f17[k + nx * ny - nx] - feq(18, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'E') {

		_rho = rho[k - nx * ny - nx];
		_ux = ux[k - nx * ny - nx];
		_uy = uy[k - nx * ny - nx];
		_uz = uz[k - nx * ny - nx];

		rho[k] = _rho;
		ux[k] = 0.1;

		f15[k] = feq(16, _rho, 0.1, 0, 0) + f15[k - nx * ny - nx] - feq(16, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'Y') {

		_rho = rho[k - nx * ny + nx];
		_ux = ux[k - nx * ny + nx];
		_uy = uy[k - nx * ny + nx];
		_uz = uz[k - nx * ny + nx];

		rho[k] = _rho;
		ux[k] = 0.1;

		f18[k] = feq(17, _rho, 0.1, 0, 0) + f18[k - nx * ny + nx] - feq(17, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'A') {

		_rho = rho[k + nx * ny + 1];
		_ux = ux[k + nx * ny + 1];
		_uy = uy[k + nx * ny + 1];
		_uz = uz[k + nx * ny + 1];

		rho[k] = _rho;

		f12[k] = feq(11, _rho, 0, 0, 0) + f12[k + nx * ny + 1] - feq(11, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'S') {

		_rho = rho[k + nx * ny - 1];
		_ux = ux[k + nx * ny - 1];
		_uy = uy[k + nx * ny - 1];
		_uz = uz[k + nx * ny - 1];

		rho[k] = _rho;

		f13[k] = feq(14, _rho, 0, 0, 0) + f13[k + nx * ny - 1] - feq(14, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'D') {

		_rho = rho[k - nx * ny + 1];
		_ux = ux[k - nx * ny + 1];
		_uy = uy[k - nx * ny + 1];
		_uz = uz[k - nx * ny + 1];

		rho[k] = _rho;
		ux[k] = 0.1;

		f14[k] = feq(13, _rho, 0.1, 0, 0) + f14[k - nx * ny + 1] - feq(13, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'G') {

		_rho = rho[k - nx * ny - 1];
		_ux = ux[k - nx * ny - 1];
		_uy = uy[k - nx * ny - 1];
		_uz = uz[k - nx * ny - 1];

		rho[k] = _rho;
		ux[k] = 0.1;

		f11[k] = feq(12, _rho, 0.1, 0, 0) + f11[k - nx * ny - 1] - feq(12, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'Z') {

		_rho = rho[k + nx + 1];
		_ux = ux[k + nx + 1];
		_uy = uy[k + nx + 1];
		_uz = uz[k + nx + 1];

		rho[k] = _rho;

		f8[k] = feq(7, _rho, 0, 0, 0) + f8[k + nx + 1] - feq(7, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'X') {

		_rho = rho[k + nx - 1];
		_ux = ux[k + nx - 1];
		_uy = uy[k + nx - 1];
		_uz = uz[k + nx - 1];

		rho[k] = _rho;

		f9[k] = feq(10, _rho, 0, 0, 0) + f9[k + nx - 1] - feq(10, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'C') {

		_rho = rho[k - nx + 1];
		_ux = ux[k - nx + 1];
		_uy = uy[k - nx + 1];
		_uz = uz[k - nx + 1];

		rho[k] = _rho;

		f10[k] = feq(9, _rho, 0, 0, 0) + f10[k - nx + 1] - feq(9, _rho, _ux, _uy, _uz);
	}

	if (flag[k] == 'V') {

		_rho = rho[k - nx - 1];
		_ux = ux[k - nx - 1];
		_uy = uy[k - nx - 1];
		_uz = uz[k - nx - 1];

		rho[k] = _rho;

		f7[k] = feq(8, _rho, 0, 0, 0) + f7[k - nx - 1] - feq(8, _rho, _ux, _uy, _uz);
	}
}