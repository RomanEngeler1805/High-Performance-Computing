#include <cassert>
#include <chrono>
#include <random>
#include <string>
#include <iostream>
#include <algorithm> 
#include <math.h> 
#include <cstdlib>     // posix_memalign
#include <emmintrin.h> // SSE/SSE2 intrinsics header
#include <cblas.h>	   // CBLAS
#include <fstream>
//#include <cstudio>
#include <omp.h>
using namespace std;


static inline void power_method_sequential(const double *A, double *b_new, int N, double &lambda, int &iter, double tol= 1e-12)
{
	// Power Method ---------------------------
	// initial guess
	double* b_old = new double[N]();
	double eps = 1.0;
	double bi;

	while(eps> tol){
		iter++;

		// b_old = b_new
		for(int i= 0; i< N; i++) b_old[i] = b_new[i];

		// A* b
		for(int i= 0; i< N; i++){
			bi = 0.0;
			for(int j= 0; j< N; j++) bi+= A[i*N+ j]* b_old[j];
			b_new[i] = bi;
		}

		// ||A* b||
		double Ab_abs = 0.0;
		for(int i= 0; i< N; i++) Ab_abs+= pow(b_new[i], 2);
		Ab_abs= sqrt(Ab_abs);

		// b_(k+1)
		for(int i= 0; i< N; i++) b_new[i]/= Ab_abs;

		// convergence
		eps = 0.0;
		for(int i= 0; i< N; i++) eps+= pow(b_new[i]- b_old[i], 2);
		eps = sqrt(eps);

	}

	// calculate lambda
	// A* v= lambda* v -> A[0, :]* v[:] = lambda* v[0]

	double v1 [N] = { };
	for(int i= 0; i< N; i++){
		for(int j= 0; j< N; j++){
			v1[i]+= A[i*N+ j]* b_new[j];
		}
		v1[i]/= b_new[i];

		lambda+= v1[i];
	}

	lambda /= N;

	delete[] b_old;
}



static inline void power_method_parallel(const double *A, double *b_new, int N, double &lambda, int &iter, double tol= 1e-12)
{
	typedef chrono::steady_clock Clock;

	// Power Method ---------------------------
	// initial guess
	double* b_old = new double[N]();
	double eps = 1.0;

	const int simd_width = 16/ sizeof(double);

	auto t1_opt = Clock::now();
	while(eps> tol){
		iter++;

		// b_old = b_new
		for(int i=0; i<N; i+=simd_width){
			const __m128d x2 = _mm_load_pd(b_new+ i);
			_mm_store_pd(b_old+ i, x2);
		}

		// A* b
		for(int i= 0; i< N; i++){
			__m128d sum = _mm_set_pd(0.0, 0.0);
			for(int j= 0; j< N; j+= simd_width){ //thread access sometimes same memory location for q -> first touch
				const __m128d A2 = _mm_load_pd(A+ i*N+ j);
				const __m128d x2 = _mm_load_pd(b_old+ j);
				const __m128d prod = _mm_mul_pd(A2, x2);
				sum = _mm_add_pd(sum, prod);
			}
			b_new[i] = sum[0]+ sum[1];
		}

		// ||A* b||
		__m128d sum = _mm_set_pd(0.0, 0.0);
		for(int i= 0; i< N; i+= simd_width){
			const __m128d x2 = _mm_load_pd(b_new+ i);
			const __m128d prod = _mm_mul_pd(x2, x2);
			sum = _mm_add_pd(sum, prod);
		}

		double Ab_abs = sqrt(sum[0]+ sum[1]);
		double *ptr_Ab_abs = &Ab_abs;

		// b_(k+1)
		for(int i= 0; i< N; i+= simd_width){
			__m128d x2 = _mm_load_pd(b_new+ i);
			__m128d x_abs = _mm_load_pd1(ptr_Ab_abs);
			x2 = _mm_div_pd(x2, x_abs);
			_mm_store_pd(b_new+ i, x2);
		}

		// convergence
		__m128d sum2 = _mm_set_pd(0.0, 0.0);

		for(int i= 0; i< N; i+= simd_width){
			const __m128d xold = _mm_load_pd(b_old+ i);
			const __m128d xnew = _mm_load_pd(b_new+ i);
			const __m128d x_sub = _mm_sub_pd(xold, xnew);
			__m128d prod = _mm_mul_pd(x_sub, x_sub);
			sum2 = _mm_add_pd(sum2, prod);
		}

		eps = sqrt(sum[0]+ sum[1]);

	}

	// calculate lambda
	// A* v= lambda* v -> A[0, :]* v[:] = lambda* v[0]

	__m128d v1 = _mm_set_pd(0.0, 0.0);
	for(int i= 0; i< N; i+=simd_width){
		const __m128d A2 = _mm_load_pd(A+ i);
		const __m128d x2 = _mm_load_pd(b_new+ i);
		const __m128d prod = _mm_mul_pd(A2, x2);
		v1 = _mm_add_pd(v1, prod);
	}

	lambda = (v1[0]+ v1[1])/ b_new[0];


	auto t2_opt = Clock::now();
	const double t_opt = chrono::duration_cast<chrono::nanoseconds>(t2_opt - t1_opt).count();
	cout<<t_opt<<endl;

	delete[] b_old;
}


static inline void power_method_cblas(const double* A, double *b, int N, double &lambda, int &iter, double tol= 1e-12)
{
	double *v = new double[N], *res = new double [N];
 
	double eps = 1.0;

	while(eps> tol){
		iter++;

		cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, 1.0, A, N, b, 1, 0.0, v, 1);
		double nrm = cblas_dnrm2(N, v, 1);
		cblas_dscal(N, 1.0/ nrm, v, 1);

		cblas_dcopy(N, v, 1, res, 1);
		cblas_daxpy(N, -1.0, b, 1, res, 1);
		cblas_dcopy(N, v, 1, b, 1);

		double error = cblas_dnrm2(N, res, 1);
		if(error < N* tol)
			break;
	}

	double v1 = 0.0;
	for(int i= 0; i< N; i++) v1+= A[i]* b[i];

	lambda = v1/ b[0];

	delete[] v;
	delete[] res;
}



int main(void)
{
	// alphas
	int n_alpha = 9;
	float alpha[n_alpha] = {0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0};

	// eigenvalues
	double eig_own [n_alpha] = {};
	double eig_own2 [n_alpha] = {};
	double eig_cblas [n_alpha] = {};

	// iteration counter
	int iter_own [n_alpha] = {};
	int iter_own2 [n_alpha] = {};
	int iter_cblas [n_alpha] = {};

	// time
	typedef chrono::steady_clock Clock;
	double time_own [n_alpha] = {};
	double time_cblas [n_alpha] = {};

	// fill matrix
	int N= 1024;
	double* A;
	posix_memalign((void**)&A, 16, N*N*sizeof(double));

  	default_random_engine gen(0);
  	uniform_real_distribution<double> u;

  	auto t1 = Clock::now();
  	auto t2 = Clock::now();

  	for(int k= 0; k< n_alpha ; k++){
		// A
		for(int i= 0; i< N; i++){
			A[i*N+ i] = alpha[k]* i;

			for(int j= i+1; j< N; j++){
				A[i*N+ j] = u(gen);
				A[j*N+ i] = A[i*N+ j];
			}
		}

		// own sequential --------------------------------------------------
	  	double* b_o = new double [N]();
		b_o[0] = 1.0;
		t1 = Clock::now();
		power_method_sequential(A, b_o, N, eig_own[k], iter_own[k]);
		t2 = Clock::now();
		time_own[k] = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

		// own parallel
	  	double* b_o2 = new double [N]();
		b_o2[0] = 1.0;
		//power_method_parallel(A, b_o2, N, eig_own2[k], iter_own2[k]);

		// cblas sequential
	  	double* b_c = new double [N]();
		b_c[0] = 1.0;
		t1 = Clock::now();
		power_method_cblas(A, b_c, N, eig_cblas[k], iter_cblas[k]);
		t2 = Clock::now();
		time_cblas[k] = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
	}

	// Eigenvalues --------------------------------------------------------
	ofstream myev;
  	myev.open ("MyEigenvalue.txt");
	myev <<"Implementation    alpha    Eigenvalue"<<endl;
	for(int k= 0; k< n_alpha ; k++){
		myev <<"Own  		  "  <<alpha[k] <<"	   "<< eig_own[k]<< endl;
		myev <<"CBLAS 		  " <<alpha[k] << "	   "<< eig_cblas[k]<< endl;
	}
	myev.close();

	// Iterations ---------------------------------------------------------
	// own method
	int min_own = *min_element(iter_own, iter_own+ n_alpha);
	int argmin_own = min_element(iter_own, iter_own+ n_alpha)- iter_own;
	int max_own = *max_element(iter_own, iter_own+ n_alpha);
	int argmax_own = max_element(iter_own, iter_own+ n_alpha)- iter_own;

	// BLAS
	int min_cblas = *min_element(iter_cblas, iter_cblas+ n_alpha);
	int argmin_cblas = min_element(iter_cblas, iter_cblas+ n_alpha)- iter_cblas;
	int max_cblas = *max_element(iter_cblas, iter_cblas+ n_alpha);
	int argmax_cblas = max_element(iter_cblas, iter_cblas+ n_alpha)- iter_cblas;

	ofstream myiter;
  	myiter.open ("MyIter.txt");
  	myiter << "Implementation    	alpha    iter"<<endl;
  	myiter << "Own 		min 	" << alpha[argmin_own] << "	 "<<min_own<<endl;
  	myiter << "Own 		max 	" << alpha[argmax_own] << "	 "<<max_own<<endl;
  	myiter << "CBLAS		min 	" << alpha[argmin_cblas] << "	 "<<min_cblas<<endl;
  	myiter << "CBLAS		max 	" << alpha[argmax_cblas] << "	 "<<max_cblas<<endl;
  	myiter.close();

	// Time -----------------------------------------------------------------
	ofstream mytime;
  	mytime.open ("MyOwnTime.txt");
  	for(int k= 0; k< n_alpha ; k++){
  		mytime << alpha[k]<<" "<< time_own[k]<< endl ;
  	}
  	mytime.close();

  	mytime.open ("MyCblasTime.txt");
  	for(int k= 0; k< n_alpha ; k++){
  		mytime << alpha[k]<<" "<< time_cblas[k]<< endl ;
  	}
  	mytime.close();

	return 0;
}