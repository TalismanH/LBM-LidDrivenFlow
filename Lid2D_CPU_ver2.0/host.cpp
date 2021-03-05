
/*
CPUver2.0，多文件编译为多个源代码文件 + 一个头文件
使用数组作为形参进行传值
*/
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>
#include "basicVar.h"

double w[Q] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //权系数
double rho[NX + 1][NY + 1], u[NX + 1][NY + 1][2], u0[NX + 1][NY + 1][2], f[NX + 1][NY + 1][Q], F[NX + 1][NY + 1][Q];
//密度，n+1时层速度，n时层速度，演化前密度分布函数，演化后密度分布函数
//int i, j, k, ip, jp, n; //k即为alpha
double error;

int main()
{
	init();

	double st = omp_get_wtime();

	for (int n = 0;; n++)
	{
		evolution(rho, u, u0, f, F);
		if (n % 100 == 0)
		{
			Error();
			cout << "The" << n << "th computation result:" << endl << "The u,v of point(NX/2,NY/2) is:" << setprecision(6) << u[NX / 2][NY / 2][0] << "," << u[NX / 2][NY / 2][1] << endl;
			cout << "The max relative error of uv is:" << setiosflags(ios::scientific) << error << endl;
			if (n >= 1000)
			{
				if (n % 1000 == 0) output(n);
				if (error < 1.0e-6) break;
			}
		}
	}

	double runtime = omp_get_wtime() - st;

	cout << "Total time is " << runtime << endl;

	return 0;
}

void init() //初始化
{
	cout << "tau_f = " << tau_f << endl;

	for (int i = 0; i <= NX; i++) //初始化
		for (int j = 0; j <= NY; j++)
		{
			u[i][j][0] = 0;
			u[i][j][1] = 0;
			rho[i][j] = rho0;
			u[i][NY][0] = U; //顶部速度
			for (int k = 0; k < Q; k++)
			{
				f[i][j][k] = feq(k, rho[i][j], u[i][j]); //初始平衡状态
			}
		}
}

double feq(int k, double rho, double u[2]) //计算平衡态分布函数
{
	double eu, uv, feq;
	eu = e[k][0] * u[0] + e[k][1] * u[1]; //e_alpha*u
	uv = u[0] * u[0] + u[1] * u[1]; //u^2
	feq = w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
	return feq;
}



void output(int m) //输出
{
	ostringstream name;
	name << "cavity_" << m << ".dat";
	ofstream out(name.str().c_str());
	out << "Title=\"LBM Lid Driven Flow\"\n" << "VARIABLES = \"X\",\"Y\",\"U\",\"V\",\"P\"\n" << "ZONE T= \"BOX\",I= " << NX + 1 << ",J= " << NY + 1 << ",F=POINT" << endl;
	for (int j = 0; j <= NY; j++)
		for (int i = 0; i <= NX; i++)
		{
			out << i << " " << j << " " << u[i][j][0] << " " << u[i][j][1] << " " << rho[i][j] / 3. << endl;
		}
}

void Error()
{
	double temp1, temp2;
	temp1 = 0;
	temp2 = 0;
	for (int i = 1; i < NX; i++)
		for (int j = 1; j < NY; j++)
		{
			temp1 += ((u[i][j][0] - u0[i][j][0]) * (u[i][j][0] - u0[i][j][0]) + (u[i][j][1] - u0[i][j][1]) * (u[i][j][1] - u0[i][j][1]));
			temp2 += (u[i][j][0] * u[i][j][0] + u[i][j][1] * u[i][j][1]);
		}
	temp1 = sqrt(temp1);
	temp2 = sqrt(temp2);
	error = temp1 / (temp2 + 1e-30);
}
