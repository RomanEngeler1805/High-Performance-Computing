#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

const int MIN_N = 1000 / sizeof(int);      // From 1 KB
const int MAX_N = 20000000 / sizeof(int);  // to 20 MB.
const int NUM_SAMPLES = 100;
const int M = 100000000;    // Operations per sample.
int a[MAX_N];               // Permutation array.


void sattolo(int *p, int N) {
    /*
     * Generate a random single-cycle permutation using Satollo's algorithm.
     *
     * https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle#Sattolo's_algorithm
     */
    for (int i = 0; i < N; ++i)
        p[i] = i;
    for (int i = 0; i < N - 2; ++i)
        std::swap(p[i], p[i + 1 + rand() % (N - i - 1)]);
}

double measure(int N, int mode) {
	int *p = new int[N];

    if (mode == 0) {
        // TODO: Question 1b: Use the sattolo function to generate a random one-cycle permutation.
    	sattolo(p, N);

    } else if (mode == 1) {
        // TODO: Question 1c: Initialize the permutation such that k jumps by 1 item every step (cyclically).
    	for(int k = 0; k< N; ++k)
    		p[k] = (k+ 1)% N;

    } else if (mode == 2) {
        // TODO: Question 1d: Initialize the permutation such that k jumps by 64 bytes (cyclically).
    	for(int k = 0; k< N; ++k)
    		p[k] = (k+ 64/ 4)% N;

    }

    // TODO: Question 1b: Traverse the list (make M jumps, starting from k = 0) and measure the execution time.
    int idx = 0;

	std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

   	for(int k = 0; k< M; ++k)
 		idx = p[idx];

 	std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2- t1);

    // TODO: Question 1b: Return execution time in seconds.
    return time_span.count();
}

void run_mode(int mode) {
    /*
     * Run the measurement for many different values of N and output in a
     * format compatible with the plotting script.
     */
    printf("%9s  %9s  %7s  %7s\n", "N", "size[kB]", "t[s]", "op_per_sec[10^9]");
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        // Generate N in a logarithmic scale. gitlab.ethz.ch:hpcse18/lecture
        int N = (int)(MIN_N * std::pow((double)MAX_N / MIN_N,
                                       (double)i / (NUM_SAMPLES - 1)));
        double t = measure(N, mode);
        printf("%9d  %9.1lf  %7.5lf  %7.6lf\n",
               N, N * sizeof(int) / 1024., t, M / t * 1e-9);
        fflush(stdout);
    }
    printf("\n\n");
}

int main() {
    // Question 1b:
    run_mode(0);   // Random.

    // TODO: Enable for Question 1c:
    run_mode(1);   // Sequential (jump by sizeof(int) bytes).

    // TODO: Enable for Question 1d:
    run_mode(2);   // Sequential (jump by cache line size, i.e. 64 bytes).

    return 0;
}
