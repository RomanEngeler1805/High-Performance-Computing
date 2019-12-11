#include <cstdio>
#include <chrono>

void measure_flops(int N, int K) {
    // TODO: Question 2b: Allocate the buffer of N * K elements.
    // int buffer[N* K]
    double* arr = new double[N* K];

    // TODO: Question 2b: Repeat `repeat` times a traversal of arrays and
    //                    measure total execution time.
    int repeat = 500 / K;

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    for(int r= 0; r< repeat; ++r){
        for(int n= 0; n< N; ++n){
            for(int k= 0; k< K; ++k){
                arr[n+ k* N]+= 1.0;
            }
        }

    }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2- t1);

    // TODO: Question 2b: Deallocate.
    delete[ ] arr;

    // Report.
    double time = time_span.count();  /* TODO: Question 2b: time in seconds */
    double flops = (double)repeat * N * K / time;
    printf("%d  %2d  %.4lf\n", N, K, flops * 1e-9);
    fflush(stdout);
}

void run(int N) {
    printf("      N   K  GFLOPS\n");
    for (int K = 1; K <= 40; ++K)
        measure_flops(N, K);
    printf("\n\n");
}

int main() {
    // Array size. Must be a multiple of a large power of two.
    const int N = 1 << 20;

    // Power of two size --> bad.
    run(N);

    // Non-power-of-two size --> better.
    // TODO: Enable for Question 2c:
    run(N + 64 / sizeof(double));

    return 0;
}

