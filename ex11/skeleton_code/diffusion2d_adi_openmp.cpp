#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "timer.hpp"

// TODO
// Include OpenMP header
#include <omp.h>



struct Diagnostics
{
    double time;
    double heat;

    Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2D
{
public:
    Diffusion2D(const double D,
                const double L,
                const int N,
                const double dt,
                const int rank)
            : D_(D), L_(L), N_(N), dt_(dt), rank_(rank)
    {
        // Real space grid spacing.
        dr_ = L_ / (N_ - 1);

        // Actual dimension of a row (+2 for the ghost cells).
        real_N_ = N + 2;

        // Total number of cells.
        Ntot_ = (N_ + 2)* (N_ + 2);

        rho_.resize(Ntot_, 0.);
        rho_tmp_.resize(Ntot_, 0.);

        // Initialize field on grid
        initialize_rho();

        R_ = 2.*dr_*dr_ / dt_;
        // TODO:
        // Initialize diagonals of the coefficient
        // matrix A, where Ax=b is the corresponding
        // system to be solved

        // define coefficient
        double lam = D_* dt_/ (2* dr_* dr_);
        // fill diagonals (a: lower, b: diag, c: upper)

        a_.resize(Ntot_-2, lam);
        b_.resize(Ntot_-2, -1.- 2.*lam);
        c_.resize(Ntot_-2, lam);

        // modify diagonal according to thomas algorithm
        for (int i = 1; i < Ntot_- 1; ++i)
        	b_[i] = b_[i]- c_[i-1]* a_[i]/ b_[i-1];

    }


    void advance()
    {
        // TODO:
        // Implement the ADI scheme for diffusion
        // and parallelize with OpenMP
        // ...

 
        // ADI Step 1: Update ROWS at half timestep
        // Solve implicit system with Thomas algorithm
        // ...
        double lam = D_* dt_/ (2* pow(dr_, 2));

        // i -- >
        // j
        // |
        // |
        // v
        // i: fast running, j: slow 

        // rows
        #pragma omp parallel for
        for (int j = 1; j < real_N_- 1; ++j){
        	// columns
        	for (int i= 1; i< real_N_- 1; ++i){
        		// create RHS v by summing weighted fields (FE in y -> real_N_ elements apart)
        		rho_tmp_[i+ j*real_N_] = -lam* rho_[i+ (j- 1)*real_N_]+  (-1.0+ 2.0* lam)* rho_[i+ j*real_N_]- lam* rho_[i+ (j+1)* real_N_];
        		// first step Thomas algorithm for RHS
        		rho_tmp_[i+ j*real_N_] = rho_tmp_[i+ j*real_N_]- rho_tmp_[i+ j*real_N_- 1]* a_[i+ j*real_N_]/ b_[i+ j*real_N_- 1];

        	}
        }

        // initialize last column in support
        //#pragma omp parallel for
        #pragma omp parallel for
        for (int j = 1; j< real_N_- 1; ++j)
            rho_[Ntot_- j* real_N_- 1] = rho_tmp_[Ntot_- j* real_N_- 1]/ b_[Ntot_- j*real_N_- 1];

        // loop over rows
        #pragma omp parallel for
        for (int j = 1; j< real_N_- 1; ++j){
        	// loop over second last: first columns
        	for (int i = real_N_- 2; i>= 1; --i){
        		rho_[i+ j* real_N_] = 1/ b_[i+ j* real_N_]* (rho_tmp_[i+ j* real_N_]- c_[i+ j* real_N_]* rho_[i+ j* real_N_+ 1]);
        	}
        }

        // ADI: Step 2: Update COLUMNS at full timestep
        // Solve implicit system with Thomas algorithm
        // ...

        // rows
        #pragma omp parallel for
        for (int j = 1; j < real_N_- 1; ++j){
            // columns
            for (int i= 1; i< real_N_- 1; ++i){
                // create RHS v by summing weighted fields (FE in y -> real_N_ elements apart)
                rho_tmp_[i+ j*real_N_] = -lam* rho_[i+ j*real_N_- 1]+  (-1.0+ 2.0* lam)* rho_[i+ j*real_N_]- lam* rho_[i+ j* real_N_+ 1];
                // first step Thomas algorithm for RHS
                rho_tmp_[i+ j*real_N_] = rho_tmp_[i+ j* real_N_]- rho_tmp_[i+ (j- 1)*real_N_]* a_[i+ j*real_N_]/ b_[i+ (j- 1)*real_N_];

            }
        }

        // initialize last row in support
        #pragma omp parallel for
        for (int i = 1; i< real_N_- 1; ++i)
            rho_[Ntot_- real_N_- i] = rho_tmp_[Ntot_- real_N_- i]/ b_[Ntot_- real_N_- i];

        // loop over rows
        #pragma omp parallel for
        for (int j = real_N_- 2; j<=1; --j){
            // loop over second last: first columns
            for (int i = 1; i< real_N_- 1; ++i){
                rho_[i+ j* real_N_] = 1/ b_[i+ j* real_N_]* (rho_tmp_[i+ j* real_N_]- c_[i+ j* real_N_]* rho_[i+ (j+ 1)* real_N_]);
            }
        }


    }


    void compute_diagnostics(const double t)
    {
        double heat = 0.0;
        for (int i = 1; i <= real_N_; ++i)
            for (int j = 1; j <= real_N_; ++j)
                heat += dr_ * dr_ * rho_[i * real_N_ + j];

#if DEBUG
        std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
        diag_.push_back(Diagnostics(t, heat));
    }


    void write_diagnostics(const std::string &filename) const
    {
        std::ofstream out_file(filename, std::ios::out);
        for (const Diagnostics &d : diag_)
            out_file << d.time << '\t' << d.heat << '\n';
        out_file.close();
    }


private:

    void initialize_rho()
    {
        /* Initialize rho(x, y, t=0) */

        double bound = 0.25 * L_;

        // TODO:
        // Initialize field rho based on the
        // prescribed initial conditions
        // and parallelize with OpenMP
        // ...
        
        #pragma omp parallel for
        for (int j = 1; j <= N_; ++j)
        for (int i = 1; i <= N_; ++i) {
            // x coord
            if(bound < i* dr_ && i*dr_ < 3.0* bound){
               	// y coord
             	if(bound < j* dr_ && j*dr_ < 3.0* bound){
            		rho_[i+ j* real_N_] = 1.0;
            	}
            }
        }
  
    }



    double D_, L_;
    int N_, Ntot_, real_N_;
    double dr_, dt_;
    double R_;
    int rank_;
    std::vector<double> rho_, rho_tmp_;
    std::vector<Diagnostics> diag_;
    std::vector<double> a_, b_, c_;
};



int main(int argc, char* argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D L N dt\n";
        return 1;
    }

#pragma omp parallel
    {
#pragma omp master
        std::cout << "Running with " << omp_get_num_threads() << " threads\n";
    }

    const double D = std::stod(argv[1]);  //diffusion constant
    const double L = std::stod(argv[2]);  //domain side size
    const int N = std::stoul(argv[3]);    //number of grid points per dimension
    const double dt = std::stod(argv[4]); //timestep

    Diffusion2D system(D, L, N, dt, 0);

    timer t;
    t.start();
    for (int step = 0; step < 1000; ++step) {
        system.advance();
#ifndef _PERF_
        system.compute_diagnostics(dt * step);
#endif
    }
    t.stop();

    std::cout << "Timing: " << N << ' ' << t.get_timing() << '\n';

#ifndef _PERF_
    system.write_diagnostics("diagnostics_openmp.dat");
#endif

    return 0;
}
