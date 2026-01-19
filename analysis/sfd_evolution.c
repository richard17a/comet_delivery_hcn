/* 
This script is adapted and modified from Jess Rigley's zodiacal cloud model presented in 
"Comet fragmentation as a source of the zodiacal cloud" (https://doi.org/10.1093/mnras/stab3482).

Simulate size-frequency distribution (SFD) evolution for comet populations using the Di Sisto 
(2009) and Jewitt (2021) fragmentation models:
 *   - MODEL == 0: Di Sisto et al. (2009)-style behaviour
 *   - MODEL == 1: Jewitt et al. (2021)-style (rotational fragmentation)

Simulation outputs (the comets' SFD as a function of time are written to ./data/sfd_evolution/<directory-name>
as .dat files)

Key compile-time options / macros
--------------------------------
 * - N_OPT:    threshold count for averaged evolution
 * - COM_NPER: number of orbits simulated per evolution step
 * - MODEL:    select fragmentation model (0 = Di Sisto, 1 = Jewitt)

Dependencies
------------
 * - GSL (GNU Scientific Library)
 * - math library (libm)

Example compile
---------------
 * gcc sfd_evolution.c -o sfd_evolution -I/opt/homebrew/opt/gsl/include -L/opt/homebrew/opt/gsl/lib -lgsl -lgslcblas -lm

Author
------
Richard Anslow (Institute of Astronomy, University of Cambridge)

*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define N_OPT 100000      // threshold count for averaged evolution
#define COM_NPER 1    // number of orbits simulated per evolution step
#define MODEL 0 // if 0: Di Sisto+ 2009, if 1: Jewitt+ 2021 (rotational fragmentation)


double sum_array(double a[], int num_elements)
{
  int i;
  double sum=0;
   for (i=1; i<num_elements; i++)
   {
	 sum = sum + a[i];
   }
   return(sum);
}


double* logspace(double r_min, double r_max, int n_points) {
    double *rbin = malloc(n_points * sizeof(double));
    if (rbin == NULL) {
        fprintf(stderr, "Error allocating memory for rbin.\n");
        exit(EXIT_FAILURE);
    }
    double log_r_min = log10(r_min);
    double log_r_max = log10(r_max);
    double dlog = (log_r_max - log_r_min) / (n_points - 1);
    
    for (int i = 0; i < n_points; i++) {
        rbin[i] = pow(10, log_r_min + i * dlog);
    }
    
    return rbin;
}


void init_sizedist(int nr, double *rbin, double mean_nr[nr]) {

    int ir, n_pb;
    n_pb = 1;

    /* limits and slopes of size distribution; radii that changes in slope occur at */
    double rb1, rb2, rb3;
    double alph1, alph2, alph3, alph4;
    double k1, k2, k3, k4;
    double r_l, r_h;
    double density_n, min_norm, r1, r2;

    density_n = 0.6;
    min_norm = 8.0801e19;
    
    r1 = 1.0;
    r2 = 10.0;
    
    alph1 = 2.25;
    alph2 = 3.0;
    alph3 = 6.0;
    alph4 = 3.5;

    rb1 = 1.0;
    rb2 = 50.;
    rb3 = 150.;

    /* convert to mass per parent body, to draw from when cloning */
    min_norm /= (double) n_pb;

    /* find K2 using normalisation */
    if (alph2 == 4.0) {
        printf("Slope = 4, so taking into account with normalisation of mass... \n");
        k2 = 3. / (4. * M_PI * density_n * 1e15) * min_norm / (log(r2) - log(r1));
    } else {
        k2 = 3. * (4. - alph2) / (4. * M_PI * density_n * 1e15) * min_norm / (pow(r2, 4. - alph2) - pow(r1, 4. - alph2));
    }

    /* find other K's using continuity of size distribution*/
    k1 = k2 * pow(rb1, alph1 - alph2);
    k3 = k2 * pow(rb2, alph3 - alph2);
    k4 = k3 * pow(rb3, alph4 - alph3);

    /* Integrate N(R) to get numbers in each bin */
    for (ir=0; ir<nr; ir++) {
        /* Find mean number of particles in given bin - doesn't have to be an integer as it's an average value. */
        r_l = rbin[ir];
        r_h = rbin[ir + 1];

        if (r_l < rb1) {
            if (r_h < rb1) {
                /* use first slope */
                mean_nr[ir] = k1 / (alph1 - 1.) * (pow(r_l, 1. - alph1) - pow(r_h, 1. - alph1));
            } else {
                /* bin crosses over break in slopes*/
                mean_nr[ir] = k1 / (alph1 - 1.) * (pow(r_l, 1. - alph1) - pow(rb1, 1. - alph1)) +
                        k2 / (alph2 - 1.) * (pow(rb1, 1. - alph2) - pow(r_h, 1. - alph2));
            }

        } else if (r_l < rb2) {
            if (r_h < rb2) {
                /* use second slope */
                mean_nr[ir] = k2 / (alph2 - 1.) * (pow(r_l, 1. - alph2) - pow(r_h, 1. - alph2));
            } else {
                /* bin crosses over break in slopes */
                mean_nr[ir] = k2 / (alph2 - 1.) * (pow(r_l, 1. - alph2) - pow(rb2, 1. - alph2)) +
                        k3 / (alph3 - 1.) * (pow(rb2, 1. - alph3) - pow(r_h, 1. - alph3));
            }
        } else if (r_l < rb3) {
            if (r_h < rb3) {
                /* use third slope */
                mean_nr[ir] = k3 / (alph3 - 1.) * (pow(r_l, 1. - alph3) - pow(r_h, 1. - alph3));
            } else {
                /* bin crosses over third break */
                mean_nr[ir] = k3 / (alph3 - 1.) * (pow(r_l, 1. - alph3) - pow(rb3, 1. - alph3)) +
                        k4 / (alph4 - 1.) * (pow(rb3, 1. - alph4) - pow(r_h, 1. - alph4));
            }
        } else {
            /* use fourth slope */
            mean_nr[ir] = k4 / (alph4 - 1.) * (pow(r_l, 1. - alph4) - pow(r_h, 1. - alph4));
        }

    }

    /* Write the number of bodies in each bin to a .dat file */
    FILE *fp = fopen("./data/sfd_evolution/size_distribution.dat", "w");
    if (fp == NULL) {
        perror("1Error opening file");
        return;
    }

    /* Print header (optional) */
    fprintf(fp, "# idx R N_R\n");

    for (ir = 0; ir < nr; ir++) {
        double avg_radius = 0.5 * (rbin[ir] + rbin[ir + 1]);
        fprintf(fp, "%d %e %e\n", ir, avg_radius, mean_nr[ir]);
    }
    fclose(fp);

    printf("Size distribution initialised with slopes %.2f %.2f %.2f %.2f, with breaks at %.2f, %.2f, and %.2f km. \n", alph1,
            alph2, alph3, alph4, rb1, rb2, rb3);
}


void gen_sizedist(int nr, double *rbin, double mean_nr[nr], double n_r[nr], double r[nr], gsl_rng *rand_gen) {

    int ir_n, ir;
    unsigned int k;

    /* randomly draw number in each radius bin using Poisson distribution. */
    for (ir_n=0; ir_n<nr; ir_n++) {

        k = gsl_ran_poisson(rand_gen, mean_nr[ir_n]);
        n_r[ir_n] = (int) k;
        r[ir_n] = 0.5 * (rbin[ir_n] + rbin[ir_n + 1]);

    }

    /* Write the number of bodies in each bin to a .dat file */
    FILE *fp = fopen("./data/sfd_evolution/gen_size_dist.dat", "w");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    /* Print header (optional) */
    fprintf(fp, "# idx R N_R\n");

    for (ir = 0; ir < nr; ir++) {
        fprintf(fp, "%d %e %f\n", ir, r[ir], n_r[ir]);
    }
    fclose(fp);

}


void RW_fig3(int nr, double *rbin, double mean_nr[nr], double n_r[nr], double r[nr], double *rmax_out, gsl_rng *rand_gen) {

    int ir_n;
    unsigned int k;

    for (ir_n=0; ir_n<nr; ir_n++) {
        k = gsl_ran_poisson(rand_gen, mean_nr[ir_n]);
        n_r[ir_n] = (int) k;
        r[ir_n] = 0.5 * (rbin[ir_n] + rbin[ir_n + 1]);
    }

    double rmax = 0.0;
    for (ir_n = nr - 1; ir_n >= 0; --ir_n) {
        if (n_r[ir_n] > 0) {
            rmax = r[ir_n];
            break;
        }
    }

    *rmax_out = rmax;
}


void RW_fig3_wrap(int nr, double *rbin, double mean_nr[nr], double n_r[nr], double r[nr], gsl_rng *rand_gen) {

    int num_runs = 1e6;
    double rmax_values[num_runs];

    for (int i = 0; i < num_runs; i++) {

        printf("%f\r", 100 * (double) i / num_runs);

        double rmax;
        RW_fig3(nr, rbin, mean_nr, n_r, r, &rmax, rand_gen);
        rmax_values[i] = rmax;
    }

    FILE *fp = fopen("./data/sfd_evolution/RW22_fig3.dat", "w");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    fprintf(fp, "# idx R_max\n");

    for (int ir = 0; ir < num_runs; ir++) {
        fprintf(fp, "%d %e\n", ir, rmax_values[ir]);
    }
    fclose(fp);

}


double frag_step(double prob, double m) {
    /* returns fractional mass loss, s*/

    double p, s, r_c;
    double density_n = 0.6;
    
    double r0 = 10;
    double s0 = 0.007;

    p = (double) rand() / (double) RAND_MAX;  /* random no. from 0 to 1 */
    if (p < prob) {

        r_c = pow(3 * m / (4 * M_PI * density_n), 1./3.) / 1e5;
        s = s0 / (r_c / r0);

        if (s > 1.) s = 1.;  /* avoid negative mass */

    } else s = 0;

    return s;
}


int find_bin_index(int nr, double rad, double r[nr]) {

    for (int i = 0; i < nr; i++) {
        if (rad >= r[i] && rad < r[i+1]) {
            return i;
        }
    }
    return -1; // outside range
}


void evol_sizedist(int nr, double n_r[nr], double r[nr], double tmax, double delta_t, double mean_nr[nr], gsl_rng *rand_gen) {

    double curr_nr[nr];
    double t, rmax, r_n, m_n, s_c, dm;
    int j, ir, ir_n, nc, bool_opt, norb, idx;

    unsigned int k;

    double density_n = 0.6;

    double n_rt[nr];
    double f_c = (1./3.) / 2.;

    for (ir = 0; ir < nr; ir++) {
        n_rt[ir] = 0.0;
    }

    for (ir=0; ir<nr; ir++) {

        if (n_r[ir] > 0) {
            r_n = r[ir];
            double r_n0 = r[ir];
            m_n = 4 * M_PI * density_n * pow(r_n * 1e5, 3) / 3;  /* initial mass in g */
            
            for (j=0; j<n_r[ir]; j++) {

                norb = 0;  /* no. of orbits at given location */
                dm = 0;
                while (norb < COM_NPER) {
                    s_c = frag_step(f_c, m_n);
                    dm += s_c * m_n;
                    m_n = (1. - s_c) * m_n;  /* update comet mass */
                    norb++;
                    if (m_n == 0.) {
                        break;
                    }
                }

                r_n = pow(3. * m_n / (4. * M_PI * density_n), 1./3.) / 1.e5;
                idx = find_bin_index(nr, r_n, r);
                if (idx > 0) {
                    n_rt[idx] += 1.;
                }
            }
        }

    }

    for (ir = 0; ir < nr; ir++) {

        n_r[ir] = n_rt[ir];
    }

}


void evol_sizedist_jewitt(int nr, double n_r[nr], double r[nr], double tmax, double delta_t, double mean_nr[nr], gsl_rng *rand_gen) {

    double curr_nr[nr];
    double t, rmax, r_n, m_n, s_c, dm, f_c;
    int j, ir, ir_n, nc, bool_opt, norb, idx;

    unsigned int k1;
    unsigned int k2;

    double density_n = 0.6;

    double n_rt[nr];

    for (ir = 0; ir < nr; ir++) {
        n_rt[ir] = 0.0;
    }

    for (ir=0; ir<nr; ir++) {

        if (n_r[ir] > 0) {
            r_n = r[ir];
            double r_n0 = r[ir];
            m_n = 4 * M_PI * density_n * pow(r_n * 1e5, 3) / 3;  /* initial mass in g */

            f_c = (delta_t / COM_NPER) / (100 * pow(r_n, 2));
            if (f_c > 1.) {
                f_c = 1.;
            }
            
            for (j=0; j<n_r[ir]; j++) {

                norb = 0;  /* no. of orbits at given location */
                dm = 0;
                while (norb < COM_NPER) {
                    double p = (double) rand() / (double) RAND_MAX;  /* random no. from 0 to 1 */
                    
                    if (p < f_c) {
                        m_n = 0.5 * m_n;  /* update comet mass */
                        r_n = pow(3. * m_n / (4. * M_PI * density_n), 1./3.) / 1.e5;
                        break;
                    }

                }

                if (r_n < r_n0) {
                    idx = find_bin_index(nr, r_n, r);
                    if (idx > 0) {
                        n_rt[idx] += 2.;
                    }
                } else {
                    n_rt[ir] += 1.;
                }

            }
        }

    }

    for (ir = 0; ir < nr; ir++) {
        n_r[ir] = n_rt[ir];

        if (n_r[ir] < 0) {
            n_r[ir] = 0;
        }
    }

}


void update_sizedist(int nr, double n_r[nr], double mean_nr[nr], gsl_rng *rand_gen) {

    int ir;
    unsigned int k1;
    unsigned int k2;

    for (ir = 0; ir < nr; ir++) {

        k1 = gsl_ran_poisson(rand_gen, mean_nr[ir]);
        k2 = gsl_ran_poisson(rand_gen, mean_nr[ir]);
        n_r[ir] += (int) k1;
        n_r[ir] -= (int) k2;

        if (n_r[ir] < 0) {
            n_r[ir] = 0;
        }

    }

}


int main(void) {

    int nr = 81;
    double mean_nr[nr], n_r[nr], r[nr];
    
    double tmax = 1e8;
    double delta_t = 1e0;
    double t_curr = 0;

    double r_min = 1e-1;
    double r_max = 1e3;
    double *rbin = logspace(r_min, r_max, nr + 1);

    const gsl_rng_type *T;
    gsl_rng *rand_gen;

    gsl_rng_env_setup();
    T = gsl_rng_default;
    rand_gen = gsl_rng_alloc(T);
    unsigned long seed = (unsigned long) time(NULL);
    gsl_rng_set(rand_gen, seed);

    init_sizedist(nr, rbin, mean_nr);
    gen_sizedist(nr, rbin, mean_nr, n_r, r, rand_gen);

    // RW_fig3_wrap(nr, rbin, mean_nr, n_r, r, rand_gen);

    int file_counter = 1;
    int iter = 0;

    if (MODEL == 0) {

        while (t_curr < tmax) {

            printf("t_curr= %e\r", t_curr);
            fflush(stdout);

            evol_sizedist(nr, n_r, r, tmax, delta_t, mean_nr, rand_gen);

            t_curr += delta_t;
            iter++;

            if (iter % 12000 == 0) {
                update_sizedist(nr, n_r, mean_nr, rand_gen);
            }

            if (iter % 100 == 0) {
                char filename[256];
                sprintf(filename, "./data/sfd_evolution/nrt_disisto_12e3/nrt_%d.dat", file_counter);
                FILE *fp = fopen(filename, "w");
                if (fp == NULL) {
                    perror("Error opening file");
                    break;
                }
                fprintf(fp, "# idx R N_R (at time = %g years)\n", t_curr);
                for (int ir = 0; ir < nr; ir++) {
                    fprintf(fp, "%d %e %f\n", ir, r[ir], n_r[ir]);
                }
                fclose(fp);
                file_counter++;
            }

        }
    } else {

        while (t_curr < tmax) {

            printf("t_curr= %e\r", t_curr);
            fflush(stdout);

            evol_sizedist_jewitt(nr, n_r, r, tmax, delta_t, mean_nr, rand_gen);

            t_curr += delta_t;
            iter++;

            if (iter % 12000 == 0) {
                update_sizedist(nr, n_r, mean_nr, rand_gen);
            }

            if (iter % 100 == 0) {
                char filename[256];
                sprintf(filename, "./data/sfd_evolution/nrt_jewitt_12e3/nrt_%d.dat", file_counter);
                FILE *fp = fopen(filename, "w");
                if (fp == NULL) {
                    perror("Error opening file");
                    break;
                }
                fprintf(fp, "# idx R N_R (at time = %g years)\n", t_curr);
                for (int ir = 0; ir < nr; ir++) {
                    fprintf(fp, "%d %e %f\n", ir, r[ir], n_r[ir]);
                }
                fclose(fp);
                file_counter++;
            }

        }

    }

    gsl_rng_free(rand_gen);

    return 0;
}
