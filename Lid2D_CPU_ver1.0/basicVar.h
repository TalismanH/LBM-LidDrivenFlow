//#pragma once
#ifndef BASICVAR_H_ //文件开头
	#define BASICVAR_H_ 

using namespace std; //命名空间

const int Q = 9; //D2Q9模型
const int e[Q][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} }; //离散速度张量
const int NX = 128; //x方向，节点数NX+1,NY+1
const int NY = 128; //y方向
const double U = 0.1; //顶盖速度
const double rho0 = 1.0;
const int Re = 1000;
const double niu = U * NX / Re;
const double tau_f = 3.0 * niu + 0.5;

double w[Q] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 }; //权系数
double rho[NX][NY], u[NX][NY][2], u0[NX][NY][2], f[NX][NY][Q], F[NX][NY][Q];
//密度，n+1时层速度，n时层速度，演化前密度分布函数，演化后密度分布函数
//int i, j, k, ip, jp, n; //k即为alpha
double error;

void init(); //初始化函数
double feq(int k, double rho, double u[2]); //平衡态函数
void evolution();
//演化函数
void output(int m); //输出函数
void Error(); //误差函数

#endif