#include <atomic>
#include <chrono>
#include <omp.h>
#include <random>
#include <thread>
#include <fstream>

const int MAX_T = 100;  // Maximum number of threads.
auto start_time = std::chrono::high_resolution_clock::now();

class ALock {
private:
    // TODO: Question 3a: Member variables.
    // MyAtomicCounter tail;
    std::atomic_int tail;
    int* slot = new int[MAX_T];
    volatile bool* flag = new bool[MAX_T];

public:
    ALock() {
        // TODO: Question 3a: Initial values.
        tail = 0;
        flag[0] = true;
    }

    void lock(int tid) {
        // TODO: Question 3a
        //*tail = tid;

        //slot[tid] = tail%MAX_T;
        slot[tid]  = atomic_fetch_add(&tail, 1);
        while(!flag[slot[tid]%MAX_T]);
        //tid.wait(flag[slot[tid]])
        // tail.count();
    }

    void unlock(int tid) {
        // TODO: Question 3a
        flag[slot[tid]%MAX_T] = false;
        flag[(slot[tid]+ 1)%MAX_T] = true;
    }
};


/*
 * Print the thread ID and the current time.
 */
void log(int tid, const char *info) {
    // TODO: Question 3a: Print the event in the format:
    //  tid     info    time_since_start
    //
    // Note: Be careful with a potential race condition here.
    ALock lock;
    lock.lock(tid);

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_passed = std::chrono::duration_cast<std::chrono::duration<double>>(end_time- start_time);

    // fprintf(stderr, "tid     info     time_since_start\n");
    fprintf(stdout, "%d  %s  %f\n", tid, info, time_passed.count());
    /*

    std::ofstream myfile;
    myfile.open ("results.txt");
    myfile << tid << info << start_time.count() << endl;
    myfile.close();
    */

    lock.unlock(tid);
}


/*
 * Sleep for `ms` milliseconds.
 */
void suspend(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


void emulate() {
    ALock lock;

#pragma omp parallel
    {
        // Begin parallel region.
        int tid = omp_get_thread_num();  // Thread ID.
        int sus_time1;
        int sus_time2;

        // TODO: Question 3b: Repeat multiple times
        //      - log(tid, "BEFORE")
        //      - lock
        //      - log(tid, "INSIDE")
        //      - Winside
        //      - unlock
        //      - log(tid, "AFTER")
        //      - Woutside
        //
        // NOTE: Make sure that:
        //      - there is no race condition in the random number generator
        //      - each thread computes different random numbers

        // End parallel region.
        for(int i= 0; i< 5; ++i){
            
            log(tid, "BEFORE");
            lock.lock(tid);
            log(tid, "INSIDE");

            sus_time1 = rand() % 150+ 50;
            suspend(sus_time1);

            lock.unlock(tid);
            log(tid, "AFTER");

            sus_time2 = rand() % 150+ 50;
            suspend(sus_time2);
        }

    }
}


/*
 * Test that a lock works properly by executing some calculations.
 */
void test_alock() {
    const int N = 1000000, A[2] = {2, 3};
    int result = 0, curr = 0;
    ALock lock;

#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();  // Thread ID.
        lock.lock(tid);

        // Something not as trivial as a single ++x.
        result += A[curr = 1 - curr];

        lock.unlock(tid);
    }

    int expected = (N / 2) * A[0] + (N - N / 2) * A[1];
    if (expected == result) {
        fprintf(stderr, "Test OK!\n");
    } else {
        fprintf(stderr, "Test NOT OK: %d != %d\n", result, expected);
        exit(1);
    //  tid     info    time_since_start
    //  tid     info    time_since_start
    }
}


int main() {
    test_alock();

    // TODO: Question 3b:
    emulate();

    return 0;
}
