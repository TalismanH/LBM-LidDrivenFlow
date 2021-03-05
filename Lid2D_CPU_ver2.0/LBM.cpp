
#include "basicVar.h"

void evolution(double rho[NX + 1][NY + 1], double u[NX + 1][NY + 1][2], double u0[NX + 1][NY + 1][2], double f[NX + 1][NY + 1][Q], double F[NX + 1][NY + 1][Q])
{
	for (int i = 1; i < NX; i++) //除去左右边界
		for (int j = 1; j < NY; j++) //除去上下边界
			for (int k = 0; k < Q; k++)
			{
				int ip = i - e[k][0];
				int jp = j - e[k][1];
				F[i][j][k] = f[ip][jp][k] + (feq(k, rho[ip][jp], u[ip][jp]) - f[ip][jp][k]) / tau_f;
			}

	for (int i = 1; i < NX; i++) //计算宏观量
		for (int j = 1; j < NY; j++)
		{
			u0[i][j][0] = u[i][j][0];
			u0[i][j][1] = u[i][j][1];
			rho[i][j] = 0;
			u[i][j][0] = 0;
			u[i][j][1] = 0;
			for (int k = 0; k < Q; k++)
			{
				f[i][j][k] = F[i][j][k];
				rho[i][j] += f[i][j][k]; //宏观密度
				u[i][j][0] += e[k][0] * f[i][j][k];
				u[i][j][1] += e[k][1] * f[i][j][k];
			}
			u[i][j][0] /= rho[i][j]; //宏观速度
			u[i][j][1] /= rho[i][j];
		}
	//边界处理，非平衡态外推格式
	for (int j = 1; j < NY; j++) //左右边界
		for (int k = 0; k < Q; k++)
		{
			rho[NX][j] = rho[NX - 1][j]; //边界附近一点值替代,右边界
			f[NX][j][k] = feq(k, rho[NX][j], u[NX][j]) + f[NX - 1][j][k] - feq(k, rho[NX - 1][j], u[NX - 1][j]);
			rho[0][j] = rho[1][j]; //左边界
			f[0][j][k] = feq(k, rho[0][j], u[0][j]) + f[1][j][k] - feq(k, rho[1][j], u[1][j]);
		}
	for (int i = 0; i <= NX; i++) //上下边界
		for (int k = 0; k < Q; k++)
		{
			rho[i][0] = rho[i][1]; //下边界
			f[i][0][k] = feq(k, rho[i][0], u[i][0]) + f[i][1][k] - feq(k, rho[i][1], u[i][1]);
			rho[i][NY] = rho[i][NY - 1]; //上边界
			u[i][NY][0] = U;
			f[i][NY][k] = feq(k, rho[i][NY], u[i][NY]) + f[i][NY - 1][k] - feq(k, rho[i][NY - 1], u[i][NY - 1]);
		}
}