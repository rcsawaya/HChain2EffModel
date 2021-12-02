#include <math.h>
#include "nrutil.h"
#include "nrutil.c"

// Line search hyperparameters
#define ALF 1.0e-4   // Learning rate
#define TOLX_lin 1.0e-7  // Convergence criterion

// BFGS hyperparameters
#define ITMAX 200
#define EPS 3.0e-8
#define TOLX (4 * EPS)
#define STPMX 100.0

#define FREEALL free_vector(xi, 1, n); free_vector(pnew, 1, n); \
	free_matrix(hessin, 1, n, 1, n); free_vector(hdg, 1, n); free_vector(g, 1, n); \
	free_vector(dg, 1, n);


// --------------------- //
// Line Search Algorithm // 
// --------------------- //
// n           : number of dimensions
// xold[1...n] : starting point
// fold        : function value at xold
// g[1...n]    : grad(f)
// p[1...n]    : direction to do line search
// x[1...n]    : new point
// *f          : new function value
// stpmax      : limits the step size
// *check      : convergence flag
// *func       : function to minimize
// --------------------- //

void lnsrch(int n, float xold[], float fold, float g[], float p[], float x[], 
		float *f, float stpmax, int *check, float (*func)(float []))
	{
	int i;
	float a, alam, alam2, alamin, b, disc, f2, rhs1, rhs2, slope, sum, temp, test, tmplam;

	*check = 0;
	// Check requested step size
	for (sum = 0.0, i = 1; i <= n; i++) sum += p[i] * p[i];
	if (sum > stpmax)
		{
		for (i = 1; i <= n; i++) p[i] *= stpmax / sum;
		}

	// Calculate gradient along the line at current position
	for (slope = 0.0, i = 1; i <= n; i++) slope += g[i] * p[i];
	if (slope >= 0.0) nrerror("Roundoff problem in lnsrch.");
	
	// Compute lambda_min
	test = 0.0;
	for (i = 1; i <= n; i++)
		{
		temp = fabs(p[i]) / FMAX(fabs(xold[i]), 1.0);
		if (temp > test) test = temp;
		}
	alamin = TOLX_lin / test;

	// Line search
	alam = 1.0;
	for (;;)
		{
		for (i = 1; i <= n; i++) x[i] = xold[i] + (alam * p[i]);
		*f = (*func)(x);

		// Check convergence
		if (alam < alamin)
			{
			for (i = 1; i <= n; i++) x[i] = xold[i];
			*check = 1;
			return;
			}
		else if (*f <= fold + (ALF * alam * slope)) return;
		else
			{
			// Start backtracking
			if (alam == 1.0) tmplam = -slope / (2.0 * (*f - fold - slope));
			else
				{
				rhs1 = *f - fold - (alam * slope);
				rhs2 = f2 - fold - (alam2 * slope);
				a = ((rhs1 / (alam * alam)) - (rhs2 / (alam2 * alam2))) / (alam - alam2);
				b = (((-alam2 * rhs1) / (alam * alam)) + ((alam * rhs2) / (alam2 * alam2))) / (alam - alam2);

				if (a == 0.0) tmplam = -slope / (2.0 * b);
				else
					{
					disc = (b * b) - (3.0 * a * slope);
					if (disc < 0.0) tmplam = 0.5 * alam;
					else if (b <= 0.0) tmplam = (-b + sqrt(disc)) / (3.0 * a);
					else tmplam = -slope / (b + sqrt(disc));
					}
				if (tmplam > (0.5 * alam)) tmplam = 0.5 * alam;
				}
			}
		alam2 = alam;
		f2 = *f;
		alam = FMAX(tmplam, 0.1 * alam);
		}
	}


// -------------- //
// BFGS Algorithm //
// -------------- //
// p[1...n] : starting point 
// n        : number of dimensions
// gtol     : value beyond which to zero the gradient
// *iter    : number of iterations performed
// *fret    : minimum value of the function
// *func    : function to minimize
// *dfunc   : gradient of function
// -------------- //

void BFGS(float p[], int n, float gtol, int *iter, float *fret,
		float (*func)(float []), void (*dfunc)(float, float [], float []))
	{
	int check, i, its, j;
	float den, fac, fad, fae, fp, stpmax, sum=0.0, sumdg, sumxi, temp, test;
	float *dg, *g, *hdg, **hessin, *pnew, *xi;
	FILE *f, *f_param, *f_grad;
	f = fopen("BFGSConvergence.txt", "w");
	f_param = fopen("BFGSParamConvergence.txt", "w");
	f_grad = fopen("BFGSGradConvergence.txt", "w");

	dg = vector(1, n);
	g = vector(1, n);
	hdg = vector(1, n);
	hessin = matrix(1, n, 1, n);
	pnew = vector(1, n);
	xi = vector(1, n);
	
	auto print_vec = [n](FILE* ftmp, float* tmp_vec)
		{
		for (int ii = 1; ii <= n; ii++) fprintf(ftmp, "%0.5f\t", tmp_vec[ii]);
		fprintf(ftmp, "\n");
		};

	// Calculate starting function value and gradients
	fp = (*func)(p);
	(*dfunc)(fp, p, g);
	fprintf(f, "%0.5f\n", fp);
	print_vec(f_param, p);
	print_vec(f_grad, g);

	// Initialize the inverse Hessian to be the identity
	for (i = 1; i <= n; i++)
		{
		for (j = 1; j <= n; j++) hessin[i][j] = 0.0;
		hessin[i][i] = 1.0;
		xi[i] = -g[i];
		sum += p[i] * p[i];
		}
	stpmax = STPMX * FMAX(sqrt(sum), (float) n);

	// Main Loop
	for (its = 1; its <= ITMAX; its++)
		{
		*iter = its;
		
		// Find minimum along line
		lnsrch(n, p, fp, g, xi, pnew, fret, stpmax, &check, func);

		// Update current position
		fp = *fret;
		for (i = 1; i <= n; i++)
			{
			xi[i] = pnew[i] - p[i];
			p[i] = pnew[i];
			}
		fprintf(f, "%0.5f\n", fp);
		print_vec(f_param, p);
		print_vec(f_grad, g);
		
		// Check for convergence
		test = 0.0;
		for (i = 1; i <= n; i++)
			{
			temp = fabs(xi[i]) / FMAX(fabs(p[i]), 1.0);
			if (temp > test) test = temp;
			}
		if (test < TOLX)
			{
			fclose(f);
			fclose(f_param);
			fclose(f_grad);
			FREEALL
			return;
			}
		
		// Get new gradient and check for convergence
		for (i = 1; i <= n; i++) dg[i] = g[i];
		(*dfunc)(fp, p, g);
		
		test = 0.0;
		den = FMAX(*fret, 1.0);
		for (i = 1; i <= n; i++)
			{
			temp = (fabs(g[i]) * FMAX(fabs(p[i]), 1.0)) / den;
			if (temp > test) test = temp;
			}
		if (test < gtol)
			{
			fclose(f);
			fclose(f_param);
			fclose(f_grad);
			FREEALL
			return;
			}

		// Update the inverse Hessian
		for (i = 1; i <= n; i++) dg[i] = g[i] - dg[i];
		for (i = 1; i <= n; i++)
			{
			hdg[i] = 0.0;
			for (j = 1; j <= n; j++) hdg[i] += hessin[i][j] * dg[j];
			}

		fac = fae = sumdg = sumxi = 0.0;
		for (i = 1; i <= n; i++)
			{
			fac += dg[i] * xi[i];
			fae += dg[i] * hdg[i];
			sumdg += SQR(dg[i]);
			sumxi += SQR(xi[i]);
			}
		if (fac > sqrt(EPS * sumdg * sumxi))
			{
			fac = 1.0 / fac;
			fad = 1.0 / fae;

			for (i = 1; i <= n; i++) dg[i] = (fac * xi[i]) - (fad * hdg[i]);
			
			for (i = 1; i <= n; i++)
			for (j = i; j <= n; j++)
				{
				hessin[i][j] += (fac * xi[i] * xi[j]) - (fad * hdg[i] * hdg[j]) + (fae * dg[i] * dg[j]);
				hessin[j][i] = hessin[i][j];
				}
			}
		for (i = 1; i <= n; i++)
			{
			xi[i] = 0.0;
			for (j = 1; j <= n; j++) xi[i] -= hessin[i][j] * g[j];
			}
		}
	fclose(f);
	fclose(f_param);
	fclose(f_grad);
	nrerror("too many iterations in BFGS");
	FREEALL

	}



























