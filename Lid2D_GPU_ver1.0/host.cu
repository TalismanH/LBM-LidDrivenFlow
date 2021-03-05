
/**
GPU ver1.0
1. 分配内存
2. 错误处理
3. 宏观量计算并行：计算时长617s（过于频繁的内存迁移）
4. 碰撞步骤并行
5. 迁移步骤并行（还能改进）
6. 边界条件步骤并行
	若程序编译报错“MSB3721，返回代码255”，则是因为 .cu 文件中未找到所调用的设备函数
	比如 LBBoundary.cu 和 LBCollandProp.cu 两个文件中，设备函数 feq 只在后者中定义了，那么编译时 LBBoundary.cu 会找不到 feq
	解决方法：
	(1) 两个 .cu 文件合并为一个
	(2) 更改两个 .cu 文件的属性：右键 .cu 文件 → 属性 →
		将 CUDA C/C++: Common 中的 "Generate Relocatable Device Code" 从否改为 "是(-rdc=true)"

若程序运行时报错 "cudaErrorLaunchOutOfResources"，则需要适当 更改 线程 或 线程块 的数量

128 * 128 网格: 并行计算后，计算时长为 44s，约为CPU计算的5.7倍
256 * 256 网格: dim3 threads_per_block(16, 16, 1); 在Release模式下，计算时间为60s；在Debug模式下，计算时间为217s
				dim3 threads_per_block(32, 32, 1); 在Release模式下，计算时间为68s
**/

//CUDA头文件
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// C/C++头文件
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <omp.h>
//自定义头文件
#include "basicVar.cuh"

//声明变量
double* rho;												//密度
double* ux, * uy;											//速度
double* u0x, * u0y;											//保存上一时步的速度
double* f0, * f1, * f2, * f3, * f4, * f5, * f6, * f7, * f8;	//分布函数
double* F0, * F1, * F2, * F3, * F4, * F5, * F6, * F7, * F8;	//保存上一时步的分布函数
int* Flag;												//标记


//CUDA错误处理
inline cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));

		memoryfinalize();
		cudaDeviceReset();

		assert(result == cudaSuccess);
	}
	
	return result;
}

int main()
{
	initializer();
	
	double error;
	double st = omp_get_wtime();

	for (int n = 0;; n++)
	{
		dim3 threads_per_block(32, 32, 1);
		dim3 number_of_blocks((NX / threads_per_block.x) + 1, (NY / threads_per_block.y) + 1, 1);

		LBColl << <number_of_blocks, threads_per_block >> > (rho, ux, uy, f0, f1, f2, f3, f4, f5, f6, f7, f8, Flag, tau_f);
		checkCuda(cudaGetLastError());
		//cudaDeviceSynchronize();

		LBBoundary << < number_of_blocks, threads_per_block >> > (NX, NY, rho, ux, uy, f0, f1, f2, f3,
																f4, f5, f6, f7, f8, Flag);
		checkCuda(cudaGetLastError());
		//evolution(rho, ux, uy,
		//	f0, f1, f2, f3, f4, f5, f6, f7, f8);

		LBProp << < number_of_blocks, threads_per_block >> > (NX, NY, f0, f1, f2, f3, f4, f5, f6, f7, f8,
															F0, F1, F2, F3, F4, F5, F6, F7, F8, Flag);
		LBUpgrade << < number_of_blocks, threads_per_block >> > (f0, f1, f2, f3, f4, f5, f6, f7, f8,
															F0, F1, F2, F3, F4, F5, F6, F7, F8, Flag);

		LBmacro_kernel << <number_of_blocks, threads_per_block >> > (rho, ux, uy, u0x, u0y, f0, f1, f2, f3, f4, f5, f6, f7, f8, Flag);
		checkCuda(cudaGetLastError());
		cudaDeviceSynchronize();

		if (n % 100 == 0)
		{
			error = Error(ux, uy, u0x, u0y);
			cout << "The" << n << "th computation result:" << endl << "The u,v of point(NX/2,NY/2) is:" << setprecision(6) << ux[(NY / 2) * NX + (NX / 2)] << "," << uy[(NY / 2) * NX + (NX / 2)] << endl;
			cout << "The max relative error of uv is:" << setiosflags(ios::scientific) << error << endl;
			if (n >= 1000)
			{
				if (n % 1000 == 0) output(n);
				if (error < 3.0e-5) break;
			}
		}
	}

	double runtime = omp_get_wtime() - st;

	cout << "Total time is " << runtime << endl;

	memoryfinalize();

	checkCuda(cudaDeviceReset());

	return 0;
}

void initializer() //初始化
{
	cout << "tau_f = " << tau_f << endl;

	memoryInitiate();

	for (int i = 0; i < NX; i++) //初始化
		for (int j = 0; j < NY; j++)
		{
			ux[j * NX + i] = 0;
			uy[j * NX + i] = 0;
			//u0x[j * NX + i] = 0;
			//u0y[j * NX + i] = 0;

			rho[j * NX + i] = rho0;

			ux[(NY - 1) * NX + i] = U; //顶部速度

			//初始平衡状态
			f0[j * NX + i] = feq(0, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f1[j * NX + i] = feq(1, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f2[j * NX + i] = feq(2, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f3[j * NX + i] = feq(3, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f4[j * NX + i] = feq(4, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f5[j * NX + i] = feq(5, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f6[j * NX + i] = feq(6, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f7[j * NX + i] = feq(7, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);
			f8[j * NX + i] = feq(8, rho[j * NX + i], ux[j * NX + i], uy[j * NX + i]);

			//初始化标记
			Flag[j * NX + i] = 'F';
			//Flag[j * NX] = 'L';
			//Flag[j * NX + NX - 1] = 'R';
			//Flag[i] = 'B';
			//Flag[(NY - 1) * NX + i] = 'T';
		}
	
	for (int i = 1; i < NX - 1; i++) {
		Flag[i] = 'B';
		Flag[(NY - 1) * NX + i] = 'T';
	}
	for (int j = 1; j < NY - 1; j++) {
		Flag[j * NX] = 'L';
		Flag[j * NX + NX - 1] = 'R';
	}

	Flag[0] = 'M'; Flag[NX - 1] = 'N'; Flag[(NY - 1) * NX] = 'Q'; Flag[NY * NX - 1] = 'P';
}

void memoryInitiate() {

	size_t size_double = NX * NY * sizeof(double);
	size_t size_int	   = NX * NY * sizeof(int);

	cudaMallocManaged(&rho, size_double);
	cudaMallocManaged(&ux,  size_double);   cudaMallocManaged(&uy,  size_double);
	cudaMallocManaged(&u0x, size_double);   cudaMallocManaged(&u0y, size_double);
	cudaMallocManaged(&f0,  size_double);   cudaMallocManaged(&F0,  size_double);
	cudaMallocManaged(&f1,  size_double);   cudaMallocManaged(&F1,  size_double);
	cudaMallocManaged(&f2,  size_double);   cudaMallocManaged(&F2,  size_double);
	cudaMallocManaged(&f3,  size_double);   cudaMallocManaged(&F3,  size_double);
	cudaMallocManaged(&f4,  size_double);   cudaMallocManaged(&F4,  size_double);
	cudaMallocManaged(&f5,  size_double);   cudaMallocManaged(&F5,  size_double);
	cudaMallocManaged(&f6,  size_double);   cudaMallocManaged(&F6,  size_double);
	cudaMallocManaged(&f7,  size_double);   cudaMallocManaged(&F7,  size_double);
	cudaMallocManaged(&f8,  size_double);   cudaMallocManaged(&F8,  size_double);

	cudaMallocManaged(&Flag, size_int);
}


void memoryfinalize() {

	cudaFree(rho);
	cudaFree(ux);		cudaFree(uy);
	cudaFree(u0x);		cudaFree(u0y);

	cudaFree(f0);		cudaFree(F0);
	cudaFree(f1);		cudaFree(F1);
	cudaFree(f2);		cudaFree(F2);
	cudaFree(f3);		cudaFree(F3);
	cudaFree(f4);		cudaFree(F4);
	cudaFree(f5);		cudaFree(F5);
	cudaFree(f6);		cudaFree(F6);
	cudaFree(f7);		cudaFree(F7);
	cudaFree(f8);		cudaFree(F8);

	cudaFree(Flag);
}


void output(int m) //输出
{
	ostringstream name;
	name << "cavity_" << m << ".dat";
	ofstream out(name.str().c_str());
	out << "Title=\"LBM Lid Driven Flow\"\n" << "VARIABLES = \"X\",\"Y\",\"U\",\"V\",\"P\"\n" << "ZONE T= \"BOX\",I= " << NX << ",J= " << NY << ",F=POINT" << endl;
	for (int j = 0; j < NY; j++)
		for (int i = 0; i < NX; i++)
		{
			out << i << " " << j << " " << ux[j * NX + i] << " " << uy[j * NX + i] << " " << rho[j * NX + i] / 3. << endl;
		}
}

double Error(double* ux, double* uy, double* u0x, double* u0y)
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