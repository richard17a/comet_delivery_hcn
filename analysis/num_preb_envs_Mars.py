import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cmcrameri import cm
from scipy.ndimage import uniform_filter1d
import scipy
import multiprocessing as mp
import atmos_entry_interp, hcn_surv_interp
from impactor_distributions import get_population_Mars, calc_surface_distributions
from global_HCN_delivery import size_to_mass_conversion
from nesvorny2023 import convert_sfd
import time, h5py

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
            vels0, radii0, angles0 = get_population_Mars(nbodies)
            valid = ((15 <= angles0) & (angles0 <= 90) &
                     (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
                     (5.03e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
            queries = np.vstack([angles0[valid], radii0[valid] * 1e3, vels0[valid] * 1e3]).T
            
            HCN_surv = hcn_interp(queries)
            masses = m_frag_interp(queries)
            
            mask = masses >= m  
            for j, req in enumerate(req_eff):
                N_vals[j, i] = len(masses[mask][HCN_surv[mask] > req])
    
    return N_vals


# *************** Precompute constants and interpolators once ***************
DENSITY = 0.6e3
MOLAR_MASS_HCN = 27.03
RADS = np.logspace(2, 4, 1000)
REQ_EFF = req_delivery_efficiency(DENSITY, MOLAR_MASS_HCN, RADS)
HCN_CONC = 1e-3

C_S = 1./4.9
F1, F2, F3 = 6.5e3, 6.5e3, 4e-3
ALPHA1, ALPHA2 = 1, 0.44
TAU1, TAU2 = 7, 13

T = np.linspace(4570, 10, 4570)
DELTA_T = (T[0] - T[-1]) / len(T)

F10 = C_S * (F1 * np.exp(-(T / TAU1) ** ALPHA1)
             + F2 * np.exp(-(T / TAU2) ** ALPHA2)
             + F3 * (4570 - T))
F = convert_sfd(F10, D_new=0.1)

F = F / 5

# HCN_INTERP = hcn_surv_interp.generate_hcn_interpolator("./data/simulation_results_Mars.pkl")
# MFRAG_INTERP, _, _ = atmos_entry_interp.generate_interpolator("./data/simulation_results_Mars.pkl")

# ******* here i am testing the sensitivity of our results to Mars' early surface density
# and assume instead a 1bar early atmosphere (indirect climatic evidence for this)
HCN_INTERP = hcn_surv_interp.generate_hcn_interpolator("./data/simulation_results_Mars_1bar.pkl")
MFRAG_INTERP, _, _ = atmos_entry_interp.generate_interpolator("./data/simulation_results_Mars_1bar.pkl")

CONCENTRATIONS = ["01mM", "1mM", "10mM", "30mM"]

OUTPUT_DIR = "./data/num_preb_envs_Mars_1bar"
# OUTPUT_DIR = "./data/num_preb_envs_Mars"

def avg_hcn_surv():

    m = np.pi * 0.6e3 * (100 ** 3) / 6

    nbodies = int(F[-1])
    
    vels0, radii0, angles0 = get_population_Mars(nbodies)
    masses0 = np.pi * 0.6e3 * ((radii0 * 1e3) ** 3) / 6
    tot_comet_mass = np.sum(masses0)

    valid = ((15 <= angles0) & (angles0 <= 90) &
            (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
            (5.03e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
    queries = np.vstack([angles0[valid], radii0[valid] * 1e3, vels0[valid] * 1e3]).T
    
    HCN_surv = HCN_INTERP(queries)
    masses = MFRAG_INTERP(queries)

    masses = masses[~np.isnan(HCN_surv)]
    HCN_surv = HCN_surv[~np.isnan(HCN_surv)]
    
    mask = masses >= m

    hcn_delivered = np.sum(masses[mask] * HCN_surv[mask] * HCN_CONC / 100)
    hcn_frac = np.mean(HCN_surv[mask])

    return tot_comet_mass, hcn_delivered, hcn_frac


def population_statistics():

    n_trials = 10000
    tot_masses = np.empty(n_trials)
    hcn_delivered = np.empty(n_trials)
    hcn_frac = np.empty(n_trials)

    for i in range(n_trials):

        print(round(100 * i / n_trials, 2), end='%\r')
        tot_masses[i], hcn_delivered[i], hcn_frac[i] = avg_hcn_surv()

    mu_mass, std_mass = scipy.stats.norm.fit(tot_masses / 1e16)
    mu_hcn, std_hcn = scipy.stats.norm.fit(hcn_delivered / 1e13)
    mu_hcn_frac, std_hcn_frac = scipy.stats.norm.fit(hcn_frac)

    print(f"Total cometary mass: mu={mu_mass:.4f}, std={std_mass:.4f}")
    print(f"HCN Delivered: mu={mu_hcn:.4f}, std={std_hcn:.4f}")
    print(f"Average HCN survival: mu={mu_hcn_frac:.4f}, std={std_hcn_frac:.4f}")

    x = np.linspace(min(tot_masses / 1e16), max(tot_masses / 1e16), 1000)
    pdf_mass = scipy.stats.norm.pdf(x, mu_mass, std_mass)

    _ = plt.figure(figsize=set_size('thesis'))

    plt.hist(tot_masses / 1e16, bins=50, density=True, histtype='step', color=cm.bamako(0.2))
    # plt.plot(x, pdf_mass, '--', c=cm.bamako(0.8))

    plt.xlabel(r'$M_{\rm tot}$ [$10^{16}\,$kg]', fontsize=13)
    plt.ylabel(r'Frequency (normalised)', fontsize=13)

    plt.minorticks_on()

    # plt.savefig('./src/figures/comet_mass_dist_Mars.pdf', bbox_inches='tight', format='pdf')

    plt.close()

    x = np.linspace(min(hcn_delivered / 1e13), max(hcn_delivered / 1e13), 1000)
    pdf_hcn = scipy.stats.norm.pdf(x, mu_hcn, std_hcn)

    _ = plt.figure(figsize=set_size('thesis'))

    plt.hist(hcn_delivered / 1e13, bins=50, histtype='step', density=True, color=cm.bamako(0.2))
    # plt.plot(x, pdf_hcn, '--', c=cm.bamako(0.8))

    plt.xlabel(r'$M_{\rm HCN}$ [$10^{13}\,$kg]', fontsize=13)
    plt.ylabel(r'Frequency (normalised)', fontsize=13)

    plt.minorticks_on()

    # plt.savefig('./src/figures/HCN_mass_dist_Mars.pdf', bbox_inches='tight', format='pdf')

    plt.close()

    _ = plt.figure(figsize=set_size('thesis'))

    plt.hist(hcn_frac, bins=50, histtype='step', density=True, color=cm.bamako(0.2))

    plt.xlabel(r'Average HCN survival [%]', fontsize=13)
    plt.ylabel(r'Frequency (normalised)', fontsize=13)

    plt.minorticks_on()

    # plt.savefig('./src/figures/HCN_surv_dist_Mars.pdf', bbox_inches='tight', format='pdf')

    plt.close()

    ks_stat_mass, p_mass = scipy.stats.kstest(tot_masses / 1e16, 'norm', args=(mu_mass, std_mass))
    ks_stat_hcn, p_hcn = scipy.stats.kstest(hcn_delivered / 1e13, 'norm', args=(mu_hcn, std_hcn))
    ks_stat_hcn_frac, p_hcn_frac = scipy.stats.kstest(hcn_frac, 'norm', args=(mu_hcn_frac, std_hcn_frac))

    print(f"Total Mass KS test: D={ks_stat_mass:.4f}, p={p_mass:.4f}")
    print(f"HCN Delivered KS test: D={ks_stat_hcn:.4f}, p={p_hcn:.4f}")
    print(f"HCN Delivered KS test: D={ks_stat_hcn_frac:.4f}, p={p_hcn_frac:.4f}")

    # np.savez('./data/Mars_pop/tot_masses.npz', tot_masses)
    # np.savez('./data/Mars_pop/hcn_delivered.npz', hcn_delivered)
    # np.savez('./data/Mars_pop/hcn_frac.npz', hcn_frac)


def run_simulation(num_runs):

    files = {
        conc: h5py.File(f"{OUTPUT_DIR}/N1_{conc}.h5", 'a')
        for conc in CONCENTRATIONS
    }

    for conc, f in files.items():
        if "N" not in f:
            f.create_dataset(
                "N",
                shape=(num_runs, len(T)),
                maxshape=(None, len(T)),
                chunks=True,
                compression="gzip"
            )

    start_time = time.time()
    for run_id in range(0, num_runs):

        N_vals = popoulation_survival(
            T, F, DELTA_T, HCN_INTERP, MFRAG_INTERP, REQ_EFF
        )

        for idx, conc in enumerate(CONCENTRATIONS):
            files[conc]["N"][run_id] = N_vals[idx]

        if (run_id + 1) % 100 == 0: 
            completed = run_id + 1
            elapsed = time.time() - start_time
            avg_time_per_run = elapsed / completed
            remaining_runs = num_runs - completed
            eta_seconds = remaining_runs * avg_time_per_run

            # Format ETA as HH:MM:SS or MM:SS
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

    # population_statistics()

    print("\n====  num_preb_envs_Mars.py  ====\n")
