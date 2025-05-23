#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define DX 0.15
#define BETA_STEP 0.01

#define NUM_THREADS 16

// Device functions for wavefunction calculations
__device__ double euclidean_norm(double r[3]) {
    return sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
}

__device__ double HF_wavefunction(double r1[3], double r2[3]) {
    const double pi = 3.141592653589793;
    const double Z = 2.0;
    return (pow(Z, 3)/pi) * exp(-Z*(euclidean_norm(r1) + euclidean_norm(r2)));
}

__device__ double Jastrow(double beta, double r1[3], double r2[3]) {
    double r12_vec[3] = {r1[0]-r2[0], r1[1]-r2[1], r1[2]-r2[2]};
    double r12 = euclidean_norm(r12_vec);
    return exp(r12/(2.0*(1.0 + beta*r12)));
}

__device__ double vmc_WF(double beta, double r1[3], double r2[3]) {
    return HF_wavefunction(r1, r2) * Jastrow(beta, r1, r2);
}


__device__ double dot_product(double* v1, double* v2) {
    double sum = 0.0;
    for (int i = 0; i < 3; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

__device__ double analytical_loc_en(double beta, double r1[3], double r2[3]){
    const double Z = 2.0;

    double r12_vec[3];
    for (int i = 0; i<3; i++) {
        r12_vec[i] = r1[i] - r2[i];
    }

    double r12 = euclidean_norm(r12_vec);
    double r1_norm = euclidean_norm(r1);
    double r2_norm = euclidean_norm(r2);

    double term1 = 1.0/r12 - Z*Z;
    double den = 1.0 + beta*r12;
    double factor = 1.0/(2.0*(den*den));
    double factor_with_dot = 1.0 - dot_product(r1, r2)/(r1_norm*r2_norm);
    double term2 = factor*((Z*(r1_norm + r2_norm)/r12)*factor_with_dot - factor - 2.0/r12 + 2.0*beta/(1+beta*r12));
    return term1 + term2;
}

__global__ void monte_carlo_kernel(
    double beta,
    double step_size,
    int n_steps,
    int n_walkers,
    int n_eq,
    double* energies,
    unsigned long seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_walkers) return;

    curandState_t state;
    curand_init(seed, tid, 0, &state);  // Unique seed per thread

    double r1[3] = {0.5, 0.5, 0.5};
    double r2[3] = {-0.5, -0.5, -0.5};
    double energy_sum = 0.0;

    for (int step = 0; step < n_steps; step++) {
        int electron = (curand_uniform_double(&state) > 0.5) ? 1 : 2;
        double new_r[3];
        double p;

        if (electron == 1) {
            for (int j = 0; j < 3; j++) {
                new_r[j] = r1[j] + (2.0 * curand_uniform_double(&state) * step_size - step_size);
            }
            p = vmc_WF(beta, new_r, r2) / vmc_WF(beta, r1, r2);
        } else {
            for (int j = 0; j < 3; j++) {
                new_r[j] = r2[j] + (2.0 * curand_uniform_double(&state) * step_size - step_size);
            }
            p = vmc_WF(beta, r1, new_r) / vmc_WF(beta, r1, r2);
        }

        p = fmin(p * p, 1.0);
        if (curand_uniform_double(&state) <= p) {
            if (electron == 1) memcpy(r1, new_r, 3*sizeof(double));
            else memcpy(r2, new_r, 3*sizeof(double));
        }

        if (step >= n_eq) {
            energy_sum += analytical_loc_en(beta, r1, r2);
        }
    }

    energies[tid] = energy_sum / (n_steps - n_eq);
}

double* execute_mc(
    double* betas,
    int n_betas,
    double step_size,
    int n_walkers,
    int n_steps,
    int n_eq,
    unsigned long seed
) {
    double *answers = (double*)malloc(n_betas * sizeof(double));
    double *d_energies;

    cudaMalloc(&d_energies, n_walkers * sizeof(double));

    dim3 threads(NUM_THREADS);
    dim3 blocks((n_walkers + threads.x - 1) / threads.x);

    for (int i = 0; i < n_betas; i++) {
        cudaMemset(d_energies, 0, n_walkers * sizeof(double));
        
        monte_carlo_kernel<<<blocks, threads>>>(
            betas[i],
            step_size,
            n_steps,
            n_walkers,
            n_eq,
            d_energies,
            seed
        );
        cudaDeviceSynchronize();

        double *h_energies = (double*)malloc(n_walkers * sizeof(double));
        cudaMemcpy(h_energies, d_energies, n_walkers * sizeof(double), cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int j = 0; j < n_walkers; j++) sum += h_energies[j];
        answers[i] = sum / n_walkers;
        
        free(h_energies);
    }

    cudaFree(d_energies);
    return answers;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <num_steps> <n_eq> <num_trajectories>\n", argv[0]);
        return 1;
    }

    int num_steps = atoi(argv[1]);
    int n_eq = atoi(argv[2]);
    int num_trajectories = atoi(argv[3]);
    double step_size = DX;

    int n_betas = (0.5 - 0.1)  / BETA_STEP;
    printf("N_STEPS: %d. N_EQ: %d. N_TRAJ: %d\n", num_steps, n_eq, num_trajectories);
    double betas[n_betas];
    for(int i = 0; i < n_betas; i++) {
        betas[i] = 0.1 + i * BETA_STEP;
    }

    printf("Executing MC\n");
    double* energies = execute_mc(betas, n_betas, step_size, 
                                 num_trajectories, num_steps, n_eq, time(NULL));

    FILE *fp = fopen("answers.txt", "w");
    if(!fp) {
        fprintf(stderr, "Error opening output file\n");
        return 1;
    }
    
    for(int i = 0; i < n_betas; i++) {
        fprintf(fp, "%.4f\t%.8f\n", betas[i], energies[i] * 27.2114);
    }
    fclose(fp);

    double min_energy = energies[0];
    for(int i = 1; i < n_betas; i++) {
        if(energies[i] < min_energy) {
            min_energy = energies[i];
        }
    }
    printf("Minimum energy: %.6f eV\n", min_energy * 27.2114);

    free(energies);
    return 0;
}