#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define Q 9
#define NX 512
#define NY 512
#define U 0.1
#define RHO 1.0
#define Re 1000
#define ERROR 1.0e-4

int e[Q][2] = { {0,0},{1,0},{0,1},{-1,0},{0,-1},{1,1},{-1,1},{-1,-1},{1,-1} };
double w[Q] = { 4.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 9,1.0 / 36,1.0 / 36,1.0 / 36,1.0 / 36 };
double rho[NX + 1][NY + 1], u[NX + 1][NY + 1][2], u0[NX + 1][NY + 1][2], f[NX + 1][NY + 1][Q], F[NX + 1][NY + 1][Q];

double tau_f, niu;
double eu, uv, feq1, feq2;
void initializer();
void track(int, double);
//void output(int);
//double feq(int k, double rho, double u[2]);

int main() {

	int i, j, k;
	int ip, jp;
	int iter = 1;
	double error = 10;
	double temp1, temp2;

	clock_t lidBegin = clock();

	initializer();
#pragma acc data copy(u,u0,rho,f), create(F)
	while (error > ERROR) {
#pragma acc kernels
		for (i = 1; i < NX; i++)
			for (j = 1; j < NY; j++)
				for (k = 0; k < Q; k++) {
					ip = i - e[k][0];
					jp = j - e[k][1];

					eu = e[k][0] * u[ip][jp][0] + e[k][1] * u[ip][jp][1];
					uv = u[ip][jp][0] * u[ip][jp][0] + u[ip][jp][1] * u[ip][jp][1];
					feq1 = w[k] * rho[ip][jp] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

					F[i][j][k] = f[ip][jp][k] + (feq1 - f[ip][jp][k]) / tau_f;
				}
#pragma acc kernels
		for (i = 1; i < NX; i++)
			for (j = 1; j < NY; j++) {
				u0[i][j][0] = u[i][j][0];
				u0[i][j][1] = u[i][j][1];
				rho[i][j] = 0;
				u[i][j][0] = 0;
				u[i][j][1] = 0;
				for (k = 0; k < Q; k++)
				{
					f[i][j][k] = F[i][j][k];
					rho[i][j] += f[i][j][k];
					u[i][j][0] += e[k][0] * f[i][j][k];
					u[i][j][1] += e[k][1] * f[i][j][k];
				}
				u[i][j][0] /= rho[i][j];
				u[i][j][1] /= rho[i][j];
			}
#pragma acc kernels
		for (j = 1; j < NY; j++)
			for (k = 0; k < Q; k++) {

				rho[NX][j] = rho[NX - 1][j];
				eu = e[k][0] * u[NX][j][0] + e[k][1] * u[NX][j][1];
				uv = u[NX][j][0] * u[NX][j][0] + u[NX][j][1] * u[NX][j][1];
				feq1 = w[k] * rho[NX][j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

				eu = e[k][0] * u[NX - 1][j][0] + e[k][1] * u[NX - 1][j][1];
				uv = u[NX - 1][j][0] * u[NX - 1][j][0] + u[NX - 1][j][1] * u[NX - 1][j][1];
				feq2 = w[k] * rho[NX - 1][j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

				f[NX][j][k] = feq1 + f[NX - 1][j][k] - feq2;

				rho[0][j] = rho[1][j];
				eu = e[k][0] * u[0][j][0] + e[k][1] * u[0][j][1];
				uv = u[0][j][0] * u[0][j][0] + u[0][j][1] * u[0][j][1];
				feq1 = w[k] * rho[0][j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

				eu = e[k][0] * u[1][j][0] + e[k][1] * u[1][j][1];
				uv = u[1][j][0] * u[1][j][0] + u[1][j][1] * u[1][j][1];
				feq2 = w[k] * rho[1][j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
				f[0][j][k] = feq1 + f[1][j][k] - feq2;
			}
#pragma acc kernels
		for (i = 0; i <= NX; i++)
			for (k = 0; k < Q; k++) {
				rho[i][0] = rho[i][1];
				eu = e[k][0] * u[i][0][0] + e[k][1] * u[i][0][1];
				uv = u[i][0][0] * u[i][0][0] + u[i][0][1] * u[i][0][1];
				feq1 = w[k] * rho[i][0] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

				eu = e[k][0] * u[i][1][0] + e[k][1] * u[i][1][1];
				uv = u[i][1][0] * u[i][1][0] + u[i][1][1] * u[i][1][1];
				feq2 = w[k] * rho[i][1] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
				f[i][0][k] = feq1 + f[i][1][k] - feq2;

				rho[i][NY] = rho[i][NY - 1];
				u[i][NY][0] = U;
				eu = e[k][0] * u[i][NY][0] + e[k][1] * u[i][NY][1];
				uv = u[i][NY][0] * u[i][NY][0] + u[i][NY][1] * u[i][NY][1];
				feq1 = w[k] * rho[i][NY] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

				eu = e[k][0] * u[i][NY - 1][0] + e[k][1] * u[i][NY - 1][1];
				uv = u[i][NY - 1][0] * u[i][NY - 1][0] + u[i][NY - 1][1] * u[i][NY - 1][1];
				feq2 = w[k] * rho[i][NY - 1] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);
				f[i][NY][k] = feq1 + f[i][NY - 1][k] - feq2;
			}
		temp1 = 0.0;
		temp2 = 0.0;
#pragma acc kernels
		for (i = 1; i < NX; i++)
			for (j = 1; j < NY; j++) {
				temp1 += ((u[i][j][0] - u0[i][j][0]) * (u[i][j][0] - u0[i][j][0]) + (u[i][j][1] - u0[i][j][1]) * (u[i][j][1] - u0[i][j][1]));
				temp2 += (u[i][j][0] * u[i][j][0] + u[i][j][1] * u[i][j][1]);
			}
		temp1 = sqrt(temp1);
		temp2 = sqrt(temp2);
		error = temp1 / (temp2 + 1e-30);

		if (iter % 100 == 0) {
#pragma acc update host(u)
			track(iter, error);
		}

		iter++;
	}

	clock_t lidEnd = clock();
	double lidTime = ((double)(lidEnd - lidBegin)) / CLOCKS_PER_SEC;

	printf("\nThe iteration is %d.\n", iter);
	printf("\nThe total time is %f seconds.\n", lidTime);

	return 0;
}

//double feq(int k, double rho, double u[2]) {
//	
//	eu = e[k][0] * u[0] + e[k][1] * u[1];
//	uv = u[0] * u[0] + u[1] * u[1];
//	return (w[k] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv));
//	
//}

void initializer() {
	int i, j, k;

	niu = U * NX / Re;
	tau_f = 3.0 * niu + 0.5;
	printf("tau_f = %f\n", tau_f);

	for (i = 0; i <= NX; i++)
		for (j = 0; j <= NY; j++) {
			u[i][j][0] = 0;
			u[i][j][1] = 0;
			u0[i][j][1] = 0;
			u0[i][j][0] = 0;
			rho[i][j] = RHO;
			u[i][NY][0] = U;
			u0[i][NY][0] = U;
			for (k = 0; k < Q; k++) {

				eu = e[k][0] * u[i][j][0] + e[k][1] * u[i][j][1];
				uv = u[i][j][0] * u[i][j][0] + u[i][j][1] * u[i][j][1];
				feq1 = w[k] * rho[i][j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv);

				f[i][j][k] = feq1;
			}
		}
}

void track(int iter, double error) {
	printf("-------------------------------------------------------------\n");
	printf("The %d th result: the u, v of point(NX/2,NY/2) are: %.10f, %.10f\n", iter, u[NX / 2][NY / 2][0], u[NX / 2][NY / 2][1]);
	printf("The %d th result: the u, v of point(NX/2,NY-1) are: %.10f, %.10f\n", iter, u[NX / 2][NY - 1][0], u[NX / 2][NY - 1][1]);
	printf("The relative error of uv is: %f\n", error);
}