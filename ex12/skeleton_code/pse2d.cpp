#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <iostream>

constexpr int SQRT_N = 32;           // Number of particles per dimension.
constexpr int N = SQRT_N * SQRT_N;   // Number of particles.
constexpr double DOMAIN_SIZE = 1.0;  // Domain side length.
constexpr double eps = 0.20;         // Epsilon.
constexpr double nu = 1.0;           // Diffusion constant.
constexpr double dt = 0.00001;       // Time step.

// Particle storage.
double x[N];
double y[N];
double phi[N];

// Helper function.
double sqr(double x) { return x * x; }

/*
 * Initialize the particle positions and values.
 */
void init(void) {
    for (int i = 0; i < SQRT_N; ++i)
    for (int j = 0; j < SQRT_N; ++j) {
        int k = i * SQRT_N + j;
        // Put particles on the lattice with up to 10% random displacement from
        // the lattice points.
        x[k] = DOMAIN_SIZE / SQRT_N * (i + 0.2 / RAND_MAX * rand() - 0.1 + 0.5);
        y[k] = DOMAIN_SIZE / SQRT_N * (j + 0.2 / RAND_MAX * rand() - 0.1 + 0.5);

        // Initial condition are two full disks with value 1.0, everything else
        // with value 0.0.
        phi[k] = sqr(x[k] - 0.3) + sqr(y[k] - 0.7) < 0.06
              || sqr(x[k] - 0.4) + sqr(y[k] - 0.2) < 0.04 ? 1.0 : 0.0;
    }
}

double nuf(double x, double y) {return 4.0/ M_PI* exp(-sqr(x)- sqr(y));}
double nuf_eps(double x, double y) {return 1.0/sqr(eps)* nuf(x/eps, y/eps);}
double min_dist(double xi, double xj){return fmin(fabs(xj- xi), DOMAIN_SIZE- fabs(xj- xi));}

/*
 * Perform a single timestep. Compute RHS of Eq. (2) and update values of phi_i.
 */
void timestep(void) {
    const double volume = sqr(DOMAIN_SIZE) / N;

    // Placeholder, remove this.
    for (int i = 0; i < N; ++i){

    // TODO 1b: Implement the RHS of Eq. (2), and the Eq. (3).
        double update = 0.0;
        // loop over neighbors
        for (int j = 0; j< N; ++j){
            double xmin = min_dist(x[i], x[j]);
            double ymin = min_dist(y[i], y[j]);
            update += volume* (phi[j]- phi[i])* nuf_eps(xmin, ymin);
        }


    // TODO 1c: Implement the forward Euler update of phi_i, as defined in Eq. (2).
        phi[i] = phi[i]+ dt* nu/ sqr(eps)* update;
    }
}

/*
 * Store the particles into a file.
 */
void save(int k) {
    char filename[32];
    sprintf(filename, "output/plot2d_%03d.txt", k);
    FILE *f = fopen(filename, "w");
    fprintf(f, "x  y  phi\n");
    for (int i = 0; i < N; ++i)
        fprintf(f, "%lf %lf %lf\n", x[i], y[i], phi[i]);
    fclose(f);
}

int main(void) {
    init();
    save(0);
    for (int i = 1; i <= 100; ++i) {
        // Every 10 time steps == 1 frame of the animation.
        fprintf(stderr, ".");
        for (int j = 0; j < 10; ++j)
            timestep();
        save(i);
    }
    fprintf(stderr, "\n");

    return 0;
}