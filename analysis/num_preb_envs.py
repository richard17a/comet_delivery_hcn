# pylint: disable=C0103,E1101,R0913,R0917,R0914,W0621,R0915,R0801

"""
Docstring for num_preb_envs
"""

import numpy as np
import atmos_entry_interp
import hcn_surv_interp
from impactor_distributions import get_population, get_population_21kms, get_population_DS09
from nesvorny2023 import convert_sfd
import h5py


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


def read_guokorenaga2020(t):
    """
    Docstring for read_guokorenaga2020

    The [Mc_25,Mc_50,Mc_75].dat files were created using the matlab script
    provided in the Supplementary Information of Guo and Korenega (2020) 
    (https://doi.org/10.1126/sciadv.aaz6234).

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


if __name__ == "__main__":

    num_runs = 100000
    run_simulation(num_runs)

    print("\n====  num_preb_envs.py  ====\n")
