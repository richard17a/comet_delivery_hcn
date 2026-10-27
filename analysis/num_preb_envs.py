"""
This module containts the Monte Carlo simulations of cometary HCN delivery to
the early Earth.

This module combines the specified impact bombardment chronology with comet size, impact
velocity and impcat angle distributions to determine the number of successful cometary
impacts through time.

These simulations are used to calculate the number and timing of successful cometary impacts. 
Alternative impact bombardment chronologies can also be used to test the sensitivity of 
the results to this model assumption
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from cmcrameri import cm
from scipy.ndimage import uniform_filter1d
import scipy
import multiprocessing as mp
import atmos_entry_interp, hcn_surv_interp
from impactor_distributions import get_population, get_population_21kms, get_population_DS09
from nesvorny2023 import convert_sfd
import h5py
import time


matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
        width_pt = width_pt / 72.27
    elif width == 'beamer':
        width_pt = 307.28987
        width_pt = width_pt / 72.27
    else:
        width_pt = width

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2


    fig_width_in = width_pt * fraction
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def read_guokorenaga2020(t, plot=False):

    time = np.loadtxt('./data/t_values.dat')
    Mc_25 = np.loadtxt('./data/Mc_25.dat')
    Mc_50 = np.loadtxt('./data/Mc_50.dat')
    Mc_75 = np.loadtxt('./data/Mc_75.dat')

    Mc_50 = Mc_50 / Mc_50[-1]
    Mc_25 = Mc_25 / Mc_25[-1]
    Mc_75 = Mc_75 / Mc_75[-1]

    Mc_25 = np.interp(
                t,
                time,
                Mc_25
            )
    
    Mc_50 = np.interp(
                t,
                time,
                Mc_50
            )

    Mc_75 = np.interp(
                t,
                time,
                Mc_75
            )
    
    t = t[0] - t

    if plot:

        _ = plt.figure(figsize=set_size('thesis'))

        plt.plot(t, Mc_50, label='Guo & Korenaga (2020)')
        plt.fill_between(t, Mc_25, Mc_75, alpha=0.25)

        plt.plot(t, 0.16 * (t[-1] - t), label='McCulloch & Bennett (1993)')

        plt.xlim(3.7, 4.5)
        plt.ylim(0, 1)

        plt.xlabel('Time [Gya]', fontsize=13)
        plt.ylabel('Normalised crustal mass', fontsize=13)

        plt.minorticks_on()

        plt.legend(frameon=False, loc='upper left')

        plt.gca().invert_xaxis()

        plt.gca().tick_params(axis='both', which='both', top=True, bottom=True, left=True, right=True, direction='in')

        plt.savefig('./src/figures/GK_MB_comparison.pdf', bbox_inches='tight', format='pdf')

        plt.show()
    
    return Mc_50


def req_delivery_efficiency(density, molar_mass_hcn, rads):
    
    volumes = (4/3) * np.pi * (rads ** 3)
    hcn_concentrations = [1e-4, 1e-3, 10e-3, 30e-3]
    hcn_amounts = [(conc * density * volumes * molar_mass_hcn / 1e3) for conc in hcn_concentrations]
    hcn_init = (4 * np.pi * density * (rads ** 3) / 3) * 1e-3
    
    return [np.mean(hcn / hcn_init * 100) for hcn in hcn_amounts]


def popoulation_survival(t, F, delta_t, hcn_interp, m_frag_interp, req_eff):

  N_vals = np.zeros((4, len(t)))
  m = np.pi * 0.6e3 * (100 ** 3) / 6

  dF = np.diff(F, prepend=0.0)

  for i, time in enumerate(t):

      nbodies = np.random.poisson(lam=dF[i])

      if nbodies > 0:
          vels0, radii0, angles0 = get_population(nbodies)
          valid = ((15 <= angles0) & (angles0 <= 90) &
                  (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
                  (11.19e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
          queries = np.vstack([angles0[valid], radii0[valid] * 1e3, vels0[valid] * 1e3]).T

          HCN_surv = hcn_interp(queries)
          masses = m_frag_interp(queries)

          mask = masses >= m
          for j, req in enumerate(req_eff):
              N_vals[j, i] = len(masses[mask][HCN_surv[mask] > req])

  return N_vals


def popoulation_survival_DS09(t, F, delta_t, hcn_interp, m_frag_interp, req_eff):

    N_vals = np.zeros((4, len(t)))
    m = np.pi * 0.6e3 * (100 ** 3) / 6

    dF = np.diff(F, prepend=0.0)
    
    for i, time in enumerate(t):

        nbodies = np.random.poisson(lam=dF[i])

        if nbodies > 0:
            vels0, radii0, angles0 = get_population_DS09(nbodies)
            valid = ((15 <= angles0) & (angles0 <= 90) &
                     (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
                     (11.19e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
            queries = np.vstack([angles0[valid], radii0[valid] * 1e3, vels0[valid] * 1e3]).T
            
            HCN_surv = hcn_interp(queries)
            masses = m_frag_interp(queries)
            
            mask = masses >= m  
            for j, req in enumerate(req_eff):
                N_vals[j, i] = len(masses[mask][HCN_surv[mask] > req])
    
    return N_vals


def popoulation_survival_21kms(t, F, delta_t, hcn_interp, m_frag_interp, req_eff):

    N_vals = np.zeros((4, len(t)))
    m = np.pi * 0.6e3 * (100 ** 3) / 6

    dF = np.diff(F, prepend=0.0)
    
    for i, time in enumerate(t):

        nbodies = np.random.poisson(lam=dF[i])

        if nbodies > 0:
            vels0, radii0, angles0 = get_population_21kms(nbodies)
            valid = ((15 <= angles0) & (angles0 <= 90) &
                     (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
                     (11.19e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
            queries = np.vstack([angles0[valid], radii0[valid] * 1e3, vels0[valid] * 1e3]).T
            
            HCN_surv = hcn_interp(queries)
            masses = m_frag_interp(queries)
            
            mask = masses >= m  
            for j, req in enumerate(req_eff):
                N_vals[j, i] = len(masses[mask][HCN_surv[mask] > req])
    
    return N_vals


def mass_to_ncom(M_tot, 
                 alphas = (1.78, 2.34, 5.78, 3.5),
                 rho = 600,
                 D_edges = (0.2e3, 2e3, 100e3, 300e3, 2000e3)):

    alpha1, alpha2, alpha3, alpha4 = alphas
    D0, D1, D2, D3, D4 = D_edges

    I1_mass = (D1 ** (4 - alpha1) - D0 ** (4 - alpha1)) / (4 - alpha1)
    I2_mass = (D2 ** (4 - alpha2) - D1 ** (4 - alpha2)) / (4 - alpha2)
    I3_mass = (D3 ** (4 - alpha3) - D2 ** (4 - alpha3)) / (4 - alpha3)
    I4_mass = (D4 ** (4 - alpha4) - D3 ** (4 - alpha4)) / (4 - alpha4)

    I1_num = (D1 ** (1 - alpha1) - D0 ** (1 - alpha1)) / (1 - alpha1)
    I2_num = (D2 ** (1 - alpha2) - D1 ** (1 - alpha2)) / (1 - alpha2)
    I3_num = (D3 ** (1 - alpha3) - D2 ** (1 - alpha3)) / (1 - alpha3)
    I4_num = (D4 ** (1 - alpha4) - D3 ** (1 - alpha4)) / (1 - alpha4)

    r2 = (D1)**(-(alpha1 - alpha2))
    r3 = r2 * (D2)**(-(alpha2 - alpha3))
    r4 = r3 * (D3)**(-(alpha3 - alpha4))

    C = (np.pi / 6.0) * rho

    denom_scalar = C * (I1_mass + r2 * I2_mass + r3 * I3_mass + r4 * I4_mass)

    K1_arr = M_tot / denom_scalar

    K1 = K1_arr
    K2 = r2 * K1_arr
    K3 = r3 * K1_arr
    K4 = r4 * K1_arr

    N_com = K1 * I1_num + K2 * I2_num + K3 * I3_num + K4 * I4_num

    return N_com, (K1, K2, K3, K4)


def N_comet_brasser():

  t = np.linspace(4570, 10, 4570)

  a1 = 0.1511
  a2 = 0.6782
  a3 = 0.2340

  tau1 = 0.919
  tau2 = 6.440
  tau3 = 41.82

  M_E = 5.97e24

  Mc = 4 * M_E * 3.1e-8 * (a1 * np.exp((t - 4500) / tau1) + a2 * np.exp((t - 4500) / tau2) + a3 * np.exp((t - 4500) / tau3))

  t[:70] = np.nan
  Mc[:70] = np.nan

  dM = np.diff(Mc, prepend=0.0)

  N_comets = []
  for i, time in enumerate(t):

    lam = np.abs(dM[i])

    if not np.isfinite(lam):
        N_comets.append(np.nan)
        continue

    dM_t = np.random.poisson(lam=lam)

    N_com, Ks = mass_to_ncom(dM_t)
    
    N_comets = np.append(N_comets, N_com)

  return t, N_comets


def popoulation_survival_brasser(hcn_interp, m_frag_interp, req_eff):

    t, N_comets = N_comet_brasser()

    N_vals = np.zeros((4, len(t)))
    m = np.pi * 0.6e3 * (100 ** 3) / 6

    for i, time in enumerate(t):

        if np.isnan(N_comets[i]):
            continue

        nbodies = np.random.poisson(lam=N_comets[i])

        if nbodies > 0:
            vels0, radii0, angles0 = get_population_21kms(nbodies)
            valid = ((15 <= angles0) & (angles0 <= 90) &
                     (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
                     (11.19e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
            queries = np.vstack([angles0[valid], radii0[valid] * 1e3, vels0[valid] * 1e3]).T
            
            HCN_surv = hcn_interp(queries)
            masses = m_frag_interp(queries)

            mask = masses >= m  
            for j, req in enumerate(req_eff):

                N_vals[j, i] = len(masses[mask][HCN_surv[mask] > req])

    return N_vals 


def load_simulation_data(concentration):

    filename = f"{OUTPUT_DIR}/N1_{concentration}.h5"
    
    with h5py.File(filename, 'r') as f:
        data = f['N'][:]
    
    return np.vstack(data) if data.ndim == 3 else data


def load_simulation_data_brasser(concentration):

    filename = f"{OUTPUT_DIR}/N1_{concentration}.h5"
    
    with h5py.File(filename, 'r') as f:
        data = f['N'][:]
    
    data = np.vstack(data) if data.ndim == 3 else data
    return data[:, ::-1]


# *************** Precompute constants and interpolators once ***************
DENSITY = 0.6e3
MOLAR_MASS_HCN = 27.03
RADS = np.logspace(2, 4, 1000)
REQ_EFF = req_delivery_efficiency(DENSITY, MOLAR_MASS_HCN, RADS)
HCN_CONC = 1e-3

C_S = 1./6.
F1, F2, F3 = 6.5e3, 6.5e3, 4e-3
ALPHA1, ALPHA2 = 1, 0.44
TAU1, TAU2 = 7, 13

T = np.linspace(4570, 10, 4570)
DELTA_T = (T[0] - T[-1]) / len(T)

F10 = C_S * (F1 * np.exp(-(T / TAU1) ** ALPHA1)
             + F2 * np.exp(-(T / TAU2) ** ALPHA2)
             + F3 * (4570 - T))
F = convert_sfd(F10)

HCN_INTERP = hcn_surv_interp.generate_hcn_interpolator("./data/simulation_results.pkl")
MFRAG_INTERP, _, _ = atmos_entry_interp.generate_interpolator("./data/simulation_results.pkl")

# HCN_INTERP = hcn_surv_interp.generate_hcn_interpolator("./data/simulation_results_01.pkl")
# MFRAG_INTERP, _, _ = atmos_entry_interp.generate_interpolator("./data/simulation_results_01.pkl")

CONCENTRATIONS = ["01mM", "1mM", "10mM", "30mM"]

# OUTPUT_DIR = "./data/num_preb_envs_100m"
# OUTPUT_DIR = "./data/num_preb_envs_100m_21kms"
# OUTPUT_DIR = "./data/num_preb_envs_100m_DS09"
# OUTPUT_DIR = "./data/num_preb_envs_100m_Brasser"
OUTPUT_DIR = "./data/num_preb_envs_100m_01"


def run_simulation(num_runs):

    files = {
        conc: h5py.File(f"{OUTPUT_DIR}/N1_{conc}.h5", 'a')
        for conc in CONCENTRATIONS
    }

    for conc, f in files.items():
        if "N" not in f:
            f.create_dataset(
                "N",
                shape=(0, len(T)),
                maxshape=(None, len(T)),
                chunks=True,
                compression="gzip"
            )

    start_row = files[CONCENTRATIONS[0]]["N"].shape[0]

    start_time = time.time()

    for run_id in range(0, num_runs):

        N_vals = popoulation_survival(
            T, F, DELTA_T, HCN_INTERP, MFRAG_INTERP, REQ_EFF
        )
        # N_vals = popoulation_survival_21kms(
        #     T, F, DELTA_T, HCN_INTERP, MFRAG_INTERP, REQ_EFF
        # )
        # N_vals = popoulation_survival_DS09(
        #     T, F, DELTA_T, HCN_INTERP, MFRAG_INTERP, REQ_EFF
        # )
        # N_vals = popoulation_survival_brasser(
        #     HCN_INTERP, MFRAG_INTERP, REQ_EFF
        # )

        row_index = start_row + run_id

        for idx, conc in enumerate(CONCENTRATIONS):
            dset = files[conc]["N"]
            dset.resize((row_index + 1, dset.shape[1]))
            dset[row_index] = N_vals[idx]

        if (run_id + 1) % 100 == 0:
            completed = run_id + 1
            elapsed = time.time() - start_time
            avg_time_per_run = elapsed / completed
            remaining_runs = num_runs - completed
            eta_seconds = remaining_runs * avg_time_per_run

            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            pct = round(100 * completed / num_runs, 1)

            print(
                f"Progress: {pct:.1f}% ({completed}/{num_runs}) | "
                f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
                f"ETA: {eta_str}  \r",
                end='',
                flush=True
            )

    for f in files.values():
        f.close()


if __name__ == "__main__":

    if True:
        
        num_runs = 100000
        run_simulation(num_runs)
    
    # graphic_mainplot()

    print("\n====  num_preb_envs.py  ====\n")
