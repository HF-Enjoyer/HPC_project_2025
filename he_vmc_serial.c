#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N_STEPS 2000            /* Number of MC steps for each electron */
#define N_EQ 800                /* Equilibration (warm up) steps */
#define N_TRJS 1024              /* 100 trajectories to reduce statistical variance */
#define DX 0.15                 /* Step size also defined */

double euclidean_norm(double v[3]) {
    double sum = 0.0;
    for (int i = 0; i < 3; i++) {
        sum += v[i]*v[i];
    }
    return sqrt(sum);
}

double dot_product(double v1[3], double v2[3]) {
    double sum = 0.0;
    for (int i = 0; i < 3; i++) {
        sum +=v1[i]*v2[i];
    }
    return sum;
}

/* Wavefunction block starts*/

/* He atom Hartree-Fock wavefunction (just product of two 1s-orbitals): */
double HF_wavefunction(double r1[3], double r2[3]) {
    const double pi = 3.141592653589793;
    const double Z = 2.0;

    double r1_norm = euclidean_norm(r1);
    double r2_norm = euclidean_norm(r2);

    return (pow(Z, 3)/pi)*exp(-Z*(r1_norm + r2_norm));
}

/* Jastrow factor in its simplest form: */
double Jastrow(double beta, double r1[3], double r2[3]) {
    double r12_vec[3];
    for (int i; i<3; i++) {
        r12_vec[i] = r1[i]-r2[i];
    }
    double r12 = euclidean_norm(r12_vec);
    double u12 = r12/(2.0*(1.0+beta*r12));

    return exp(u12);
}

/* VMC wavefunction = HF*Jastrow: */
double vmc_WF(double beta, double r1[3], double r2[3]) {
    return HF_wavefunction(r1, r2)*Jastrow(beta, r1, r2);
}

/* Wavefunction block ends */

/* Energy block starts */
double analytical_loc_en(double beta, double r1[3], double r2[3]) {
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

double variational_energy(double beta, double tr1[N_STEPS + 1][3], double tr2[N_STEPS + 1][3], int N_eq) {
    double E_loc_sum = 0.0;
    int count = 0;

    for (int i = N_eq; i<(N_STEPS+1); i++) {
        E_loc_sum += analytical_loc_en(beta, tr1[i], tr2[i]);
        count++;
    }
    return E_loc_sum / count; /* Add condition that count != 0 */
}
/* Energy block ends */
/* Monte-Carlo block starts */
/* Check this carefully */
void random_walker(double beta, double step_size, int n_steps, double trajectory1[(N_STEPS+1)][3], double trajectory2[(N_STEPS+1)][3]) {
    /* Initialize random number generator */
    srand(time(NULL));
    
    /* Initialize electron positions */
    double r1[3] = {0.5, 0.5, 0.5};
    double r2[3] = {-0.5, -0.5, -0.5};
    
    /* Append initial positions to trajectories */
    for (int j = 0; j < 3; j++) {
        trajectory1[0][j] = r1[j];
        trajectory2[0][j] = r2[j];
    }
    
    /* Monte Carlo Loop */
    for (int step = 0; step < n_steps; step++) {
        /* Choose electron (1 or 2) */
        int electron = rand() % 2 + 1;
        
        double new_r[3];
        double p;
        
        if (electron == 1) {
            /* Propose new position for r1 */
            for (int j = 0; j < 3; j++) {
                new_r[j] = r1[j] + (2.0*rand()/(double)RAND_MAX*step_size - step_size);
            }
            /* Compute acceptance probability */
            p = vmc_WF(beta, new_r, r2) / vmc_WF(beta, r1, r2);
            p = p * p; /* Square of ratio */
            if (p > 1.0) p = 1.0; /* min(1, p^2) */
            
            /* Accept or reject */
            if (p >= rand() / (double)RAND_MAX) {
                for (int j = 0; j < 3; j++) {
                    r1[j] = new_r[j];
                }
            }
        } else {
            /* Propose new position for r2 */
            for (int j = 0; j < 3; j++) {
                new_r[j] = r2[j] + (2.0*rand()/(double)RAND_MAX*step_size - step_size);
            }
            /* Compute acceptance probability */
            p = vmc_WF(beta, r1, new_r) / vmc_WF(beta, r1, r2);
            p = p * p; /* Square of ratio */
            if (p > 1.0) p = 1.0; /* min(1, p^2) */
            
            /* Accept or reject */
            if (p >= rand() / (double)RAND_MAX) {
                for (int j = 0; j < 3; j++) {
                    r2[j] = new_r[j];
                }
            }
        }
        
        /* Store current positions in trajectories */
        for (int j = 0; j < 3; j++) {
            trajectory1[step + 1][j] = r1[j];
            trajectory2[step + 1][j] = r2[j];
        }
    }
}

double average_over_trajectories(double beta, int n_trajectories, double step_size, int N_steps, int N_eq) {
    double energies[n_trajectories];
    double energy_sum = 0.0;
    double tr1[(N_STEPS+1)][3];
    double tr2[(N_STEPS+1)][3];
    int count = 0;

    /* unsigned int base_seed = (unsigned int)time(NULL); */
    for (int i=0; i<n_trajectories; i++) {

        random_walker(beta, step_size, N_steps, tr1, tr2);
        energies[i] = variational_energy(beta, tr1, tr2, N_eq);
        energy_sum += energies[i];
        count++;
    }
    return energy_sum/count;
}

int main() {
    double beta = 0.225;
    double avg_energy = average_over_trajectories(beta, N_TRJS, DX, N_STEPS, N_EQ);
    printf("Average total energy, eV: %f\n", 27.2114*avg_energy);
    return 0;
}
/* Monte-Carlo block ends */