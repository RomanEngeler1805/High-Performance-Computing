#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>

inline long exact(const long N){
    // TODO b): Implement the analytical solution.
    long sum = N* (N+ 1)/ 2;
    return sum;
}

void reduce_mpi(const int rank, long& sum){
    // TODO e): Perform the reduction using blocking collectives.
    long total_sum = 0;
    MPI_Reduce(&sum, &total_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    sum = total_sum;
}

// PRE: size is a power of 2 for simplicity
void reduce_manual(int rank, int size, long& sum){
    // TODO f): Implement a tree based reduction using blocking point-to-point communication.
	// levels of reductions
    int level = size/2;

    while(rank< level && level>0){
    	// receive message
    	long msg;
		MPI_Recv(&msg, 1, MPI_LONG, MPI_ANY_SOURCE, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    sum+= msg;

	    // next lower level
    	level/= 2;
	}

    // send message
    if(rank> 0){
	    int rec = rank- level;
	    MPI_Send(&sum, 1, MPI_LONG, rec, rec, MPI_COMM_WORLD);
	}

}


int main(int argc, char** argv){
    const long N = 1000000;
    
    // Initialize MPI
    int rank, size;
    // TODO c): Initialize MPI and obtain the rank and the number of processes (size)
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // -------------------------
    // Perform the local sum:
    // -------------------------
    long sum = 0;

    // Determine work load per rank
    long N_per_rank = N / size;
    
    // TODO d): Determine the range of the subsum that should be calculated by this rank.
    long N_start = rank* N_per_rank;
    long N_end = std::min(N_start+ N_per_rank, N);
    
    // N_start + (N_start+1) + ... + (N_start+N_per_rank-1)
    for(long i = N_start+ 1; i <= N_end; ++i){
        sum += i;
    }

    // -------------------------
    // Reduction
    // -------------------------
    //reduce_mpi(rank, sum);
    reduce_manual(rank, size, sum);
    
    // -------------------------
    // Print the result
    // -------------------------
    if(rank == 0){
        std::cout << std::left << std::setw(25) << "Final result (exact): " << exact(N) << std::endl;
        std::cout << std::left << std::setw(25) << "Final result (MPI): " << sum << std::endl;
    }
    // Finalize MPI
    // TODO c): Finalize MPI

    MPI_Finalize();
    
    return 0;
}
