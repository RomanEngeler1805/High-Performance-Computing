#include <iostream>
#include <cmath>
#include <vector>
#include <chrono>
#include <ctime>
#include <ratio>
#include <fstream>

using namespace std;

// g++ ex1_diffusion_roengele.cpp -O2 -std=c++11 -o ex01

int main()
{
	/* varialbe choices */
        int L = 1000, N = 1000;
        float alpha = 1e-4;

	/* number of iterations */
	int M = 20;

	/* calculate derived variables*/
        float dx = (float)L / (float) N;
        float dt = pow(dx, 2) / alpha;

	float lam = dt* alpha/ pow(dx, 2);
	float lam2 = 1.0- 2.0* lam;
	
	float u0;

	/* to flush the cache */
	const size_t bigger_than_cachesize = 10 * 1024 * 1024; /* 10M* 8B = 80MB */
	double *p = new double[bigger_than_cachesize];

	/* for time taking */
	auto t0 = std::chrono::milliseconds(0);
	chrono::duration<double> time_span_avg = chrono::duration_cast<chrono::duration<double>>(t0);

	/* initialize diffusion vector*/
	vector<float> u(N, 0.0f);

	for (int i = 0; i < N; ++i){
		u[i] = sin(2* M_PI/ (float) L * i* dx);
	}

	for(int j = 0; j < M; ++j){
		/* flush cache */
		for(int i = 0; i < bigger_than_cachesize; ++i){
			p[i] = rand();
		}

		chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

		/* iterate over diffusion vector */
		for (int i = 1; i < N-1; ++i){
			u[i] = lam2* u[i] + lam* (u[i-1]+ u[i+1]);
		}

        	/* boundary conditions */
		u0 = u[0];
		u[0] = u[N];
		u[N] = u[0];
	
		chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
		chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2- t1);
		time_span_avg += time_span;
	}
	
	time_span_avg /= M;
	cout << time_span_avg.count() << endl;

	std::ofstream file("ex1_output_roengele.txt");
	file << "Number of Grid points: " << N << endl;
   	file << "Average elapsed Time: " << time_span_avg.count();
        
	return 0;
}
