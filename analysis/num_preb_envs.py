# pylint: disable=C0103,E1101,R0913,R0917,R0914,W0621,R0915

"""
Docstring for num_preb_envs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
import scipy
import atmos_entry_interp
import hcn_surv_interp
from impactor_distributions import get_population, get_population_21kms, get_population_DS09
from nesvorny2023 import convert_sfd
import h5py
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def req_delivery_efficiency(density, molar_mass_hcn, rads):
    """
    Docstring for req_delivery_efficiency
    
    :param density: Description
    :param molar_mass_hcn: Description
    :param rads: Description
    """

    volumes = (4/3) * np.pi * (rads ** 3)
    hcn_concentrations = [1e-4, 1e-3, 10e-3, 30e-3]
    hcn_amounts = [(conc * density * volumes * molar_mass_hcn / 1e3) for conc in hcn_concentrations]
    hcn_init = (4 * np.pi * density * (rads ** 3) / 3) * 1e-3

    return [np.mean(hcn / hcn_init * 100) for hcn in hcn_amounts]


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

HCN_INTERP = hcn_surv_interp.generate_hcn_interpolator(
                "./data/atmospheric_entry/simulation_results.pkl"
            )
MFRAG_INTERP, _, _ = atmos_entry_interp.generate_interpolator(
                    "./data/atmospheric_entry/simulation_results.pkl"
                    )

CONCENTRATIONS = ["01mM", "1mM", "10mM", "30mM"]

OUTPUT_DIR = "./data/num_preb_envs_100m"
# OUTPUT_DIR = "./data/num_preb_envs_100m_21kms"
# OUTPUT_DIR = "./data/num_preb_envs_100m_DS09"
# OUTPUT_DIR = "./data/num_preb_envs_100m_Brasser"


def read_guokorenaga2020(t, plot=False):
    """
    Docstring for read_guokorenaga2020
    
    :param t: Description
    :param plot: Description
    """

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

        plt.gca().tick_params(axis='both', which='both', top=True, bottom=True,
                              left=True, right=True, direction='in')

        plt.savefig('./src/figures/GK_MB_comparison.pdf', bbox_inches='tight', format='pdf')

        plt.show()

    return Mc_50


def popoulation_survival(t, F, hcn_interp, m_frag_interp, req_eff):
    """
    Docstring for popoulation_survival
    
    :param t: Description
    :param F: Description
    :param hcn_interp: Description
    :param m_frag_interp: Description
    :param req_eff: Description
    """

    N_vals = np.zeros((4, len(t)))
    m = np.pi * 0.6e3 * (100 ** 3) / 6

    dF = np.diff(F, prepend=0.0)

    for i, _ in enumerate(t):

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


def popoulation_survival_DS09(t, F, hcn_interp, m_frag_interp, req_eff):
    """
    Docstring for popoulation_survival_DS09
    
    :param t: Description
    :param F: Description
    :param hcn_interp: Description
    :param m_frag_interp: Description
    :param req_eff: Description
    """

    N_vals = np.zeros((4, len(t)))
    m = np.pi * 0.6e3 * (100 ** 3) / 6

    dF = np.diff(F, prepend=0.0)

    for i, _ in enumerate(t):

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


def popoulation_survival_21kms(t, F, hcn_interp, m_frag_interp, req_eff):
    """
    Docstring for popoulation_survival_21kms
    
    :param t: Description
    :param F: Description
    :param hcn_interp: Description
    :param m_frag_interp: Description
    :param req_eff: Description
    """

    N_vals = np.zeros((4, len(t)))
    m = np.pi * 0.6e3 * (100 ** 3) / 6

    dF = np.diff(F, prepend=0.0)

    for i, _ in enumerate(t):

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
                 D_edges = (0.1e3, 1e3, 50e3, 150e3, 1000e3)):
    """
    Docstring for mass_to_ncom
    
    :param M_tot: Description
    :param alphas: Description
    :param rho: Description
    :param D_edges: Description
    """

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

    r2 = (D1)*(-(alpha1 - alpha2))
    r3 = (D2)**(-(alpha2 - alpha3))
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
    """
    Docstring for N_comet_brasser
    """

    t = np.linspace(4570, 10, 4570)

    a1 = 0.1511
    a2 = 0.6782
    a3 = 0.2340

    tau1 = 0.919
    tau2 = 6.440
    tau3 = 41.82

    M_E = 5.97e24

    Mc = 4 * M_E * 3.1e-8 * (a1 * np.exp((t - 4500) / tau1) + a2 * np.exp((t - 4500) / tau2) +\
                             a3 * np.exp((t - 4500) / tau3))

    t[:70] = np.nan
    Mc[:70] = np.nan

    dM = np.diff(Mc, prepend=0.0)

    N_comets = []
    for i, _ in enumerate(t):

        lam = np.abs(dM[i])

        if not np.isfinite(lam):
            N_comets.append(np.nan)
            continue

        dM_t = np.random.poisson(lam=lam)

        N_com, _ = mass_to_ncom(dM_t)

        N_comets = np.append(N_comets, N_com)

    return t, N_comets


def popoulation_survival_brasser(hcn_interp, m_frag_interp, req_eff):
    """
    Docstring for popoulation_survival_brasser
    
    :param hcn_interp: Description
    :param m_frag_interp: Description
    :param req_eff: Description
    """

    t, N_comets = N_comet_brasser()

    N_vals = np.zeros((4, len(t)))
    m = np.pi * 0.6e3 * (100 ** 3) / 6

    for i, _ in enumerate(t):

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
    """
    Docstring for load_simulation_data
    
    :param concentration: Description
    """

    filename = f"{OUTPUT_DIR}/N1_{concentration}.h5"
    print(filename)

    with h5py.File(filename, 'r') as f:
        data = f['N'][:]

    return np.vstack(data) if data.ndim == 3 else data


def load_simulation_data_brasser(concentration):
    """
    Docstring for load_simulation_data_brasser
    
    :param concentration: Description
    """

    filename = f"{OUTPUT_DIR}/N1_{concentration}.h5"

    with h5py.File(filename, 'r') as f:
        data = f['N'][:]

    data = np.vstack(data) if data.ndim == 3 else data
    return data[:, ::-1]


def plot_combined():
    """
    Docstring for plot_combined
    """

    t = np.linspace(4.57, 0, 4570)[1:]

    Mc_50 = read_guokorenaga2020(t)

    f_crust_lo = 0.16 * 0.41 * t
    f_crust_hi = np.maximum(Mc_50 * 0.41, 0.0016)
    # this assumes an oceanic island landmass at early times following Bada & Korenaga 2021 (Life)

    f_crust = f_crust_lo

    concentrations = ["01mM", "1mM", "10mM", "30mM"]
    colors = [cm.bamako(0.2), cm.bamako(0.5), cm.bamako(0.8), cm.bamako(0.9)]
    labels = [r'$c_{\rm HCN} > 0.1\,{\rm mM}$',
              r'$c_{\rm HCN} > 1\,{\rm mM}$',
              r'$c_{\rm HCN} > 10\,{\rm mM}$',
              r'$c_{\rm HCN} > 30\,{\rm mM}$']

    all_runs = {}
    for _, conc in enumerate(concentrations):

        sim = load_simulation_data(conc)
        # need to use mulation_data_Brasser for brasser simulations...
        all_runs[conc] = sim[:, 1:]
        print(f"conc={conc}", f"num_runs={np.shape(all_runs[conc])[0]}")

    mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.5)

    _ = plt.figure(figsize=set_size('thesis'))

    print('c_HCN |', 'Guo & Korenaga (2020) |', 'McCulloch & Bennett (1993)')
    for conc, color, label in zip(concentrations, colors, labels):
        runs = all_runs[conc]

        if conc != "30mM":
            mean_curve = runs.mean(axis=0)
            plt.plot(4.57 - t, mean_curve * f_crust,
                    label=label, color=color, zorder=10)

        hi_events = runs[:, mask] * f_crust_hi[mask]
        lo_events = runs[:, mask] * f_crust_lo[mask]

        totals_hi = hi_events.sum(axis=1)
        totals_lo = lo_events.sum(axis=1)

        median_hi = np.median(totals_hi)
        median_lo = np.median(totals_lo)

        print(conc, median_hi, median_lo)

    plt.axhline(1, c='tab:gray', alpha=0.2, zorder=0)

    plt.ylim(1e-4, 1e0)

    plt.yscale('log')

    plt.xlabel('Time [Gya]', fontsize=13)
    plt.ylabel('Number of impacts per Myr', fontsize=13)

    plt.minorticks_on()

    plt.legend(frameon=False, fontsize=11, loc='upper left')

    plt.savefig('./src/figures/num_impacts_conc.pdf', bbox_inches='tight', format='pdf')

    plt.show()

    _ = plt.figure(figsize=set_size('thesis'))

    # mask = t < 1.07
    mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.5)
    x_vals = 4.57 - t[mask]
    for conc, color, label in zip(concentrations, colors, labels):
        runs = all_runs[conc]
        if conc != "30mM":

            mean_curve = np.cumsum(runs.mean(axis=0)[mask] * f_crust[mask])
            plt.plot(x_vals, mean_curve, label=label, color=color, zorder=10)

        totals = (runs[:, mask] * f_crust[mask]).sum(axis=1)
        median_total = np.median(totals)
        print(conc, f"median_total_impacts = {median_total}")

        cumperrun = np.cumsum(runs[:, mask] * f_crust[mask], axis=1)
        idxs = np.argmax(cumperrun >= 1, axis=1)
        never = cumperrun.max(axis=1) < 1
        idxs[never] = -1

        times = np.full_like(idxs, np.nan, dtype=float)
        valid = idxs >= 0
        times[valid] = x_vals[idxs[valid]]

        median_cross = np.nanmedian(times)

        plt.axvline(x=median_cross, color=color, linestyle="--", zorder=5)
        print(f"{conc:>5s} last crossing at t={median_cross:.3f} Gya")

    plt.xlim(3.5, 4.5)
    plt.ylim(0, 40)

    plt.xticks([3.5, 3.7, 3.9, 4.1, 4.3, 4.5])

    plt.xlabel('Time [Gya]', fontsize=13)
    plt.ylabel('Cumulative number of impacts', fontsize=13)

    plt.minorticks_on()

    plt.legend(frameon=False, fontsize=11, loc='upper right')

    plt.savefig('./src/figures/num_impacts_conc_cumulative.pdf', bbox_inches='tight', format='pdf')

    plt.show()


def avg_hcn_surv():
    """
    Docstring for avg_hcn_surv
    """

    m = np.pi * 0.6e3 * (100 ** 3) / 6

    nbodies = int(F[-1])

    vels0, radii0, angles0 = get_population(nbodies)
    masses0 = np.pi * 0.6e3 * ((radii0 * 1e3) ** 3) / 6
    tot_comet_mass = np.sum(masses0)

    valid = ((15 <= angles0) & (angles0 <= 90) &
            (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e8) &
            (11.19e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
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
    """
    Docstring for population_statistics
    """

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

    # x = np.linspace(min(tot_masses / 1e18), max(tot_masses / 1e18), 1000)
    # pdf_mass = scipy.stats.norm.pdf(x, mu_mass, std_mass)

    _ = plt.figure(figsize=set_size('thesis'))

    plt.hist(tot_masses / 1e18, bins=50, density=True, histtype='step', color=cm.bamako(0.2))
    # plt.plot(x, pdf_mass, '--', c=cm.bamako(0.8))

    plt.xlabel(r'$M_{\rm tot}$ [$10^{18}\,$kg]', fontsize=13)
    plt.ylabel(r'Frequency (normalised)', fontsize=13)

    plt.minorticks_on()

    plt.savefig('./src/figures/comet_mass_dist.pdf', bbox_inches='tight', format='pdf')

    plt.close()

    # x = np.linspace(min(hcn_delivered / 1e13), max(hcn_delivered / 1e13), 1000)
    # pdf_hcn = scipy.stats.norm.pdf(x, mu_hcn, std_hcn)

    _ = plt.figure(figsize=set_size('thesis'))

    plt.hist(hcn_delivered / 1e13, bins=50, histtype='step', density=True, color=cm.bamako(0.2))
    # plt.plot(x, pdf_hcn, '--', c=cm.bamako(0.8))

    plt.xlabel(r'$M_{\rm HCN}$ [$10^{13}\,$kg]', fontsize=13)
    plt.ylabel(r'Frequency (normalised)', fontsize=13)

    plt.minorticks_on()

    plt.savefig('./src/figures/HCN_mass_dist.pdf', bbox_inches='tight', format='pdf')

    plt.close()

    _ = plt.figure(figsize=set_size('thesis'))

    plt.hist(hcn_frac, bins=50, histtype='step', density=True, color=cm.bamako(0.2))

    plt.xlabel(r'Average HCN survival [%]', fontsize=13)
    plt.ylabel(r'Frequency (normalised)', fontsize=13)

    plt.minorticks_on()

    plt.savefig('./src/figures/HCN_surv_dist.pdf', bbox_inches='tight', format='pdf')

    plt.close()

    ks_stat_mass, p_mass = scipy.stats.kstest(tot_masses / 1e18, 'norm',
                                              args=(mu_mass, std_mass))
    ks_stat_hcn, p_hcn = scipy.stats.kstest(hcn_delivered / 1e13, 'norm',
                                            args=(mu_hcn, std_hcn))
    ks_stat_hcn_frac, p_hcn_frac = scipy.stats.kstest(hcn_frac, 'norm',
                                                      args=(mu_hcn_frac, std_hcn_frac))

    print(f"Total Mass KS test: D={ks_stat_mass:.4f}, p={p_mass:.4f}")
    print(f"HCN Delivered KS test: D={ks_stat_hcn:.4f}, p={p_hcn:.4f}")
    print(f"HCN Delivered KS test: D={ks_stat_hcn_frac:.4f}, p={p_hcn_frac:.4f}")

    #np.savez('./data/Earth_pop/tot_masses.npz', tot_masses)
    #np.savez('./data/Earth_pop/hcn_delivered.npz', hcn_delivered)
    #np.savez('./data/Earth_pop/hcn_frac.npz', hcn_frac)


def run_simulation(num_runs):
    """
    Docstring for run_simulation
    
    :param num_runs: Description
    """

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

    for run_id in range(1, num_runs + 1):

        print(f"{round(100 * run_id / num_runs, 3)}%\r", end='')

        N_vals = popoulation_survival(
            T, F, HCN_INTERP, MFRAG_INTERP, REQ_EFF
        )
        # N_vals = popoulation_survival_21kms(
        #     T, F, HCN_INTERP, MFRAG_INTERP, REQ_EFF
        # )
        # N_vals = popoulation_survival_DS09(
        #     T, F, HCN_INTERP, MFRAG_INTERP, REQ_EFF
        # )
        # N_vals = popoulation_survival_brasser(
        #     HCN_INTERP, MFRAG_INTERP, REQ_EFF
        # )

        for idx, conc in enumerate(CONCENTRATIONS):
            dset = files[conc]["N"]
            new_row_index = dset.shape[0]
            dset.resize((new_row_index + 1, dset.shape[1]))
            dset[new_row_index] = N_vals[idx]


    for f in files.values():
        f.close()


def MB_GK_comparison():
    """
    Docstring for MB_GK_comparison
    """

    t = np.linspace(4.57, 0, 4570)[1:]
    Mc_50 = read_guokorenaga2020(t)

    f_crust_lo = 0.16 * 0.41 * t
    f_crust_hi = np.maximum(Mc_50 * 0.41, 0.0016)
    # this assumes an oceanic island landmass at early times following Bada & Korenaga 2021 (Life)

    mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.55)

    concentrations = ["01mM", "1mM", "10mM", "30mM"]
    colors = [cm.bamako(0.2), cm.bamako(0.5), cm.bamako(0.8), cm.bamako(0.9)]
    labels = [r'$c_{\rm HCN} > 0.1\,{\rm mM}$',
              r'$c_{\rm HCN} > 1\,{\rm mM}$',
              r'$c_{\rm HCN} > 10\,{\rm mM}$',
              r'$c_{\rm HCN} > 30\,{\rm mM}$']

    all_runs = {}
    for _, conc in enumerate(concentrations):

        sim = load_simulation_data(conc)

        all_runs[conc] = sim[:, 1:]
        print(f"conc={conc}", f"num_runs={np.shape(all_runs[conc])[0]}")

    print('c_HCN |', 'Guo & Korenaga (2020) |', 'McCulloch & Bennett (1993)')
    for conc, _, _ in zip(concentrations, colors, labels):
        runs = all_runs[conc]

        num_events_hi = np.random.poisson(runs[:, mask] * f_crust_hi[mask])
        num_events_lo = np.random.poisson(runs[:, mask] * f_crust_lo[mask])

        totals_hi = num_events_hi.sum(axis=1)  # total impacts per run
        totals_lo = num_events_lo.sum(axis=1)  # total impacts per run

        print(conc, np.median(totals_hi), np.median(totals_lo))


def successful_impacts_hist():
    """
    Docstring for successful_impacts_hist
    """

    t = np.linspace(4.57, 0, 4570)[1:]

    f_crust_lo = 0.16 * 0.41 * t
    f_crust = f_crust_lo

    conc = "10mM"
    all_runs = {}
    sim = load_simulation_data(conc)
    all_runs[conc] = sim[:, 1:]
    mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.55)
    # Nesvorny chronology expression should not be used for t \lesssim 10Myr

    runs = all_runs[conc]

    num_events = np.random.poisson(runs[:, mask] * f_crust[mask])

    totals = num_events.sum(axis=1)  # total impacts per run
    print(np.median(totals), np.mean(totals), len(totals[totals == 0]) / len(totals))

    minv = totals.min()
    maxv = totals.max()
    x = np.arange(minv, maxv + 1)

    counts_all = np.bincount(totals, minlength=maxv + 1)
    counts = counts_all[x]

    _ = plt.figure(figsize=set_size('thesis'))

    plt.bar(x, counts / totals.size,
       width=0.7, align='center',
       edgecolor='black', linewidth=0.6,
       color=cm.bamako(0.3), alpha=0.75)

    plt.xlim(-0.5, 6.5)

    plt.ylabel(r'Fraction of Monte Carlo realisations', fontsize=13)
    plt.xlabel(r'Number of cometary impacts delivering $>10~$mM HCN', fontsize=13)

    plt.minorticks_on()

    plt.gca().tick_params(axis='both', which='both', top=True, bottom=True,
                          left=True, right=True, direction='in')

    # plt.savefig('./src/figures/successful_impacts_bar.pdf', bbox_inches='tight', format='pdf')

    plt.show()

    np.savetxt('./data/x.txt', x)
    np.savetxt('./data/counts.txt', counts / totals.size)


def total_comet_mass_acc():
    """
    Docstring for total_comet_mass_acc
    """

    nbodies = int(F[-1])

    n_trials = 10000
    tot_masses = np.empty(n_trials)

    for i in range(n_trials):

        print(round(100 * i / n_trials, 2), end='%\r')

        _, radii0, _ = get_population(nbodies)
        masses0 = np.pi * 0.6e3 * ((radii0 * 1e3) ** 3) / 6

        tot_masses[i] = np.sum(masses0)

    mu_mass, std_mass = scipy.stats.norm.fit(tot_masses / 1e18)
    print(f"Total cometary mass: mu={mu_mass:.4f}, std={std_mass:.4f}")

    # x = np.linspace(min(tot_masses / 1e18), max(tot_masses / 1e18), 1000)
    # pdf_mass = scipy.stats.norm.pdf(x, mu_mass, std_mass)

    _ = plt.figure(figsize=set_size('thesis'))

    plt.hist(tot_masses / 1e18, bins=50, density=True, histtype='step', color=cm.bamako(0.2))

    plt.xlabel(r'$M_{\rm tot}$ [$10^{18}\,$kg]', fontsize=13)
    plt.ylabel(r'Frequency (normalised)', fontsize=13)

    plt.minorticks_on()

    plt.savefig('./src/figures/comet_mass_dist.pdf', bbox_inches='tight', format='pdf')

    plt.show()

    ks_stat_mass, p_mass = scipy.stats.kstest(tot_masses / 1e18, 'norm', args=(mu_mass, std_mass))

    print(f"Total Mass KS test: D={ks_stat_mass:.4f}, p={p_mass:.4f}")

    np.savez('./data/Earth_pop/tot_masses.npz', tot_masses)


if __name__ == "__main__":

    num_runs = 100000
    run_simulation(num_runs)

    # successful_impacts_hist()

    # MB_GK_comparison()

    # read_guokorenaga2020(np.linspace(4.57, 0, 4500)[1:], plot=True)
    # population_statistics()
    # total_comet_mass_acc()

    print("\n====  num_preb_envs.py  ====\n")
