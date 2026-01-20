"""
Docstring for init_params
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
import atmos_entry_interp
import hcn_surv_interp
from impactor_distributions import get_population, get_population_Mars, get_population_Venus
from nesvorny2023 import convert_sfd
from scipy.stats import kde
from tqdm import tqdm
from num_preb_envs import req_delivery_efficiency

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


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

HCN_INTERP_MARS = hcn_surv_interp.generate_hcn_interpolator(
            "./data/atmospheric_entry/simulation_results_Mars.pkl"
            )
MFRAG_INTERP_MARS, _, _ = atmos_entry_interp.generate_interpolator(
            "./data/atmospheric_entry/simulation_results_Mars.pkl"
            )

HCN_INTERP_VENUS = hcn_surv_interp.generate_hcn_interpolator(
            "./data/atmospheric_entry/simulation_results_Venus.pkl"
            )
MFRAG_INTERP_VENUS, _, _ = atmos_entry_interp.generate_interpolator(
            "./data/atmospheric_entry/simulation_results_Venus.pkl"
            )


def initial_params_success(t, F, hcn_interp, m_frag_interp, req_eff, planet):
    """
    Docstring for initial_params_success
    
    :param t: Description
    :param F: Description
    :param hcn_interp: Description
    :param m_frag_interp: Description
    :param req_eff: Description
    :param planet: Description
    """

    results = []
    for req in req_eff:
        results.append({'req': req, 'vels': [], 'angles': [], 'radii': [], 'times': []})

    m = np.pi * 0.6e3 * (100 ** 3) / 6

    dF = np.diff(F, prepend=0.0)

    for i, time in enumerate(t):
        nbodies = np.random.poisson(lam=dF[i])
        if nbodies <= 0:
            continue

        if planet == 'Earth':
            vels0, radii0, angles0 = get_population(nbodies)

            valid = ((15 <= angles0) & (angles0 <= 90) &
                    (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
                    (11.19e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
        elif planet == 'Mars':
            vels0, radii0, angles0 = get_population_Mars(nbodies)

            valid = ((15 <= angles0) & (angles0 <= 90) &
                     (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
                     (5.03e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
            nbodies = nbodies / 5
        elif planet == 'Venus':
            vels0, radii0, angles0 = get_population_Venus(nbodies)
            valid = ((15 <= angles0) & (angles0 <= 90) &
                     (100 <= radii0 * 1e3) & (radii0 * 1e3 <= 1e4) &
                     (10.36e3 <= vels0 * 1e3) & (vels0 * 1e3 <= 70e3))
            nbodies = nbodies * 0.94

        if not np.any(valid):
            continue

        angles_valid = angles0[valid]
        radii_valid = radii0[valid] * 1e3
        vels_valid = vels0[valid] * 1e3

        queries = np.vstack([angles_valid, radii_valid, vels_valid]).T

        HCN_surv = hcn_interp(queries)
        masses = m_frag_interp(queries)

        mass_mask = masses >= m

        for j, req in enumerate(req_eff):
            success_mask = mass_mask & (HCN_surv > req)
            if not np.any(success_mask):
                continue

            results[j]['vels'].append(vels_valid[success_mask])
            results[j]['angles'].append(angles_valid[success_mask])
            results[j]['radii'].append(radii_valid[success_mask])
            results[j]['times'].append(np.full(np.sum(success_mask), time))

    for res in results:
        if res['vels']:
            res['vels'] = np.concatenate(res['vels'])
            res['angles'] = np.concatenate(res['angles'])
            res['radii'] = np.concatenate(res['radii'])
            res['times'] = np.concatenate(res['times'])
        else:
            res['vels'] = np.array([], dtype=float)
            res['angles'] = np.array([], dtype=float)
            res['radii'] = np.array([], dtype=float)
            res['times'] = np.array([], dtype=float)

    return results


def repeat_sequential(n_runs, hcn_interp, mfrag_interp, base_seed=0, planet='Earth'):
    """
    Docstring for repeat_sequential
    
    :param n_runs: Description
    :param hcn_interp: Description
    :param mfrag_interp: Description
    :param base_seed: Description
    :param planet: Description
    """

    all_runs = []

    for run_idx in tqdm(range(n_runs), desc="runs"):
        np.random.seed(base_seed + run_idx)
        res = initial_params_success(T, F, hcn_interp, mfrag_interp, REQ_EFF, planet)
        all_runs.append(res)

    return all_runs


def hdr_contour(x, y, frac=0.5, nbins=100, subsample=1000):
    """
    Docstring for hdr_contour
    
    :param x: Description
    :param y: Description
    :param frac: Description
    :param nbins: Description
    :param subsample: Description
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # subsample if requested
    if subsample is not None and x.size > subsample:
        idx = np.random.choice(x.size, subsample, replace=False)
        x, y = x[idx], y[idx]

    # KDE fit
    k = kde.gaussian_kde(np.vstack([x, y]))

    # grid
    xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
    zi = k(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

    # compute HDR threshold
    flat = zi.ravel()
    order = np.argsort(flat)[::-1]
    flat_sorted = flat[order]
    dx = (x.max() - x.min()) / nbins
    dy = (y.max() - y.min()) / nbins
    cell_area = dx * dy
    cumsum = np.cumsum(flat_sorted) * cell_area
    pos = np.searchsorted(cumsum, frac)
    z_thresh = flat_sorted[pos] if pos < len(flat_sorted) else flat_sorted[-1]

    return xi, yi, zi, z_thresh


def subsample(x, y, n_points=2000):
    """
    Randomly sample n_points from arrays x, y (without replacement).
    """

    n = min(len(x), n_points)  # don’t oversample if dataset is small
    idx = np.random.choice(len(x), n, replace=False)

    return x[idx], y[idx]


def main_Earth():
    """
    Docstring for main_Earth
    """

    vels0, radii0, angles0 = get_population(1000000)
    results = repeat_sequential(1000, HCN_INTERP, MFRAG_INTERP)

    req_idx = 2
    vels2 = np.concatenate([run[req_idx]['vels'] for run in results\
                            if run[req_idx]['vels'].size > 0])
    radii2 = np.concatenate([run[req_idx]['radii'] for run in results\
                            if run[req_idx]['radii'].size > 0])
    angles2 = np.concatenate([run[req_idx]['angles'] for run in results\
                            if run[req_idx]['angles'].size > 0])

    req_idx = 0
    vels1 = np.concatenate([run[req_idx]['vels'] for run in results\
                            if run[req_idx]['vels'].size > 0])
    radii1 = np.concatenate([run[req_idx]['radii'] for run in results\
                            if run[req_idx]['radii'].size > 0])
    angles1 = np.concatenate([run[req_idx]['angles'] for run in results\
                            if run[req_idx]['angles'].size > 0])

    vels2 = vels2 / 1e3
    radii2 = np.log10(radii2)
    vels1 = vels1 / 1e3
    radii1 = np.log10(radii1)

    radii0 = np.log10(radii0 * 1e3)
    variables0 = [angles0, vels0, radii0]
    variables2  = [angles2, vels2, radii2]
    variables1  = [angles1, vels1, radii1]
    labels = [r"Angle [deg]", r"Velocity [km/s]", r"$\log_{10}$(Radius [m])"]

    nvar = len(variables2)

    color_0  = cm.bamako(0.2)
    color_1 = cm.bamako(0.5)
    color_2 = cm.bamako(0.8)

    color_0=matplotlib.colors.to_rgba(color_0)
    color_1=matplotlib.colors.to_rgba(color_1)
    color_2=matplotlib.colors.to_rgba(color_2)

    fig, axes = plt.subplots(nvar, nvar, figsize=(10,10))

    for i in range(nvar):
        for j in range(nvar):
            ax = axes[i, j]

            if i == j:
                ax.ecdf(variables0[i], label=r"all comets", color=color_0)
                ax.ecdf(variables1[i], label=r"$c_{\rm HCN}>0.1~{\rm mM}$", color=color_1)
                ax.ecdf(variables2[i], label=r"$c_{\rm HCN}>10~{\rm mM}$", color=color_2)
                if i == 0:
                    ax.legend(fontsize=10, frameon=False)

            elif i > j:

                x, y = subsample(variables0[j], variables0[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.1,
                           color=color_0, lw=0, rasterized=True)
                x, y = subsample(variables1[j], variables1[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.05,
                           color=color_1, lw=0, rasterized=True)
                x, y = subsample(variables2[j], variables2[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.025,
                           color=color_2, lw=0, rasterized=True)

                xi, yi, zi, z_thresh = hdr_contour(variables0[j], variables0[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_0])
                xi, yi, zi, z_thresh = hdr_contour(variables1[j], variables1[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_1])
                xi, yi, zi, z_thresh = hdr_contour(variables2[j], variables2[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_2])

            else:
                ax.axis("off")

            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=13)
            if i == nvar-1:
                ax.set_xlabel(labels[j], fontsize=13)

            ax.minorticks_on()

    axes[0, 0].set_xlim(15, 90)
    axes[1, 0].set_xlim(15, 90)
    axes[2, 0].set_xlim(15, 90)

    axes[1, 0].set_ylim(15, 50)
    axes[1, 1].set_xlim(15, 50)
    axes[2, 1].set_xlim(15, 50)

    axes[2, 0].set_ylim(2, 4)
    axes[2, 1].set_ylim(2, 4)
    axes[2, 2].set_xlim(2, 4)

    fig.tight_layout()

    # plt.savefig('./figures/10mM_initial_params_Earth.pdf', bbox_inches='tight', format='pdf')

    plt.show()


def main_Mars():
    """
    Docstring for main_Mars
    """

    vels0, radii0, angles0 = get_population_Mars(1000000)
    results = repeat_sequential(1000, HCN_INTERP_MARS, MFRAG_INTERP_MARS, planet="Mars")

    req_idx = 2
    vels2 = np.concatenate([run[req_idx]['vels'] for run in results\
                            if run[req_idx]['vels'].size > 0])
    radii2 = np.concatenate([run[req_idx]['radii'] for run in results\
                            if run[req_idx]['radii'].size > 0])
    angles2 = np.concatenate([run[req_idx]['angles'] for run in results\
                            if run[req_idx]['angles'].size > 0])

    req_idx = 0
    vels1 = np.concatenate([run[req_idx]['vels'] for run in results\
                            if run[req_idx]['vels'].size > 0])
    radii1 = np.concatenate([run[req_idx]['radii'] for run in results\
                            if run[req_idx]['radii'].size > 0])
    angles1 = np.concatenate([run[req_idx]['angles'] for run in results\
                            if run[req_idx]['angles'].size > 0])

    vels2 = vels2 / 1e3
    radii2 = np.log10(radii2)
    vels1 = vels1 / 1e3
    radii1 = np.log10(radii1)

    radii0 = np.log10(radii0 * 1e3)
    variables0 = [angles0, vels0, radii0]
    variables2  = [angles2, vels2, radii2]
    variables1  = [angles1, vels1, radii1]
    labels = [r"Angle [deg]", r"Velocity [km/s]", r"$\log_{10}$(Radius [m])"]

    nvar = len(variables2)

    color_0  = cm.bamako(0.2)
    color_1 = cm.bamako(0.5)
    color_2 = cm.bamako(0.8)

    color_0=matplotlib.colors.to_rgba(color_0)
    color_1=matplotlib.colors.to_rgba(color_1)
    color_2=matplotlib.colors.to_rgba(color_2)

    fig, axes = plt.subplots(nvar, nvar, figsize=(10,10))

    for i in range(nvar):
        for j in range(nvar):
            ax = axes[i, j]

            if i == j:
                ax.ecdf(variables0[i], label=r"all comets", color=color_0)
                ax.ecdf(variables1[i], label=r"$c_{\rm HCN}>0.1~{\rm mM}$", color=color_1)
                ax.ecdf(variables2[i], label=r"$c_{\rm HCN}>10~{\rm mM}$", color=color_2)
                if i == 0:
                    ax.legend(fontsize=10, frameon=False)

            elif i > j:

                x, y = subsample(variables0[j], variables0[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.1, color=color_0, lw=0, rasterized=True)
                x, y = subsample(variables1[j], variables1[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.05, color=color_1, lw=0, rasterized=True)
                x, y = subsample(variables2[j], variables2[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.025, color=color_2, lw=0, rasterized=True)

                xi, yi, zi, z_thresh = hdr_contour(variables0[j], variables0[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_0])
                xi, yi, zi, z_thresh = hdr_contour(variables1[j], variables1[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_1])
                xi, yi, zi, z_thresh = hdr_contour(variables2[j], variables2[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_2])

            else:
                ax.axis("off")

            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=13)
            if i == nvar-1:
                ax.set_xlabel(labels[j], fontsize=13)

            ax.minorticks_on()

    axes[0, 0].set_xlim(15, 90)
    axes[1, 0].set_xlim(15, 90)
    axes[2, 0].set_xlim(15, 90)

    axes[1, 0].set_ylim(10, 40)
    axes[1, 1].set_xlim(10, 40)
    axes[2, 1].set_xlim(10, 40)

    axes[2, 0].set_ylim(2, 4)
    axes[2, 1].set_ylim(2, 4)
    axes[2, 2].set_xlim(2, 4)

    fig.tight_layout()

    # plt.savefig('./figures/10mM_initial_params_Mars.pdf', bbox_inches='tight', format='pdf')

    plt.show()


def main_Venus():
    """
    Docstring
    """

    vels0, radii0, angles0 = get_population_Venus(1000000)
    results = repeat_sequential(1000, HCN_INTERP_VENUS, MFRAG_INTERP_VENUS, planet="Venus")

    req_idx = 2
    vels2 = np.concatenate([run[req_idx]['vels'] for run in results\
                            if run[req_idx]['vels'].size > 0])
    radii2 = np.concatenate([run[req_idx]['radii'] for run in results\
                            if run[req_idx]['radii'].size > 0])
    angles2 = np.concatenate([run[req_idx]['angles'] for run in results\
                            if run[req_idx]['angles'].size > 0])

    req_idx = 0
    vels1 = np.concatenate([run[req_idx]['vels'] for run in results\
                            if run[req_idx]['vels'].size > 0])
    radii1 = np.concatenate([run[req_idx]['radii'] for run in results\
                            if run[req_idx]['radii'].size > 0])
    angles1 = np.concatenate([run[req_idx]['angles'] for run in results\
                            if run[req_idx]['angles'].size > 0])

    vels2 = vels2 / 1e3
    radii2 = np.log10(radii2)
    vels1 = vels1 / 1e3
    radii1 = np.log10(radii1)

    radii0 = np.log10(radii0 * 1e3)
    variables0 = [angles0, vels0, radii0]
    variables2  = [angles2, vels2, radii2]
    variables1  = [angles1, vels1, radii1]
    labels = [r"Angle [deg]", r"Velocity [km/s]", r"$\log_{10}$(Radius [m])"]

    nvar = len(variables2)

    color_0  = cm.bamako(0.2)
    color_1 = cm.bamako(0.5)
    color_2 = cm.bamako(0.8)

    color_0=matplotlib.colors.to_rgba(color_0)
    color_1=matplotlib.colors.to_rgba(color_1)
    color_2=matplotlib.colors.to_rgba(color_2)

    fig, axes = plt.subplots(nvar, nvar, figsize=(10,10))

    for i in range(nvar):
        for j in range(nvar):
            ax = axes[i, j]

            if i == j:
                ax.ecdf(variables0[i], label=r"all comets", color=color_0)
                ax.ecdf(variables1[i], label=r"$c_{\rm HCN}>0.1~{\rm mM}$", color=color_1)
                ax.ecdf(variables2[i], label=r"$c_{\rm HCN}>10~{\rm mM}$", color=color_2)
                if i == 0:
                    ax.legend(fontsize=10, frameon=False)

            elif i > j:

                x, y = subsample(variables0[j], variables0[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.1, color=color_0, lw=0, rasterized=True)
                x, y = subsample(variables1[j], variables1[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.05, color=color_1, lw=0, rasterized=True)
                x, y = subsample(variables2[j], variables2[i], n_points=5000)
                ax.scatter(x, y, marker='.', alpha=0.025, color=color_2, lw=0, rasterized=True)

                xi, yi, zi, z_thresh = hdr_contour(variables0[j], variables0[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_0])
                xi, yi, zi, z_thresh = hdr_contour(variables1[j], variables1[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_1])
                xi, yi, zi, z_thresh = hdr_contour(variables2[j], variables2[i])
                ax.contour(xi, yi, zi, levels=[z_thresh], colors=[color_2])

            else:
                ax.axis("off")

            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=13)
            if i == nvar-1:
                ax.set_xlabel(labels[j], fontsize=13)

            ax.minorticks_on()

    axes[0, 0].set_xlim(15, 90)
    axes[1, 0].set_xlim(15, 90)
    axes[2, 0].set_xlim(15, 90)

    axes[1, 0].set_ylim(25, 55)
    axes[1, 1].set_xlim(25, 55)
    axes[2, 1].set_xlim(25, 55)

    axes[2, 0].set_ylim(2, 4)
    axes[2, 1].set_ylim(2, 4)
    axes[2, 2].set_xlim(2, 4)

    fig.tight_layout()

    # plt.savefig('./figures/10mM_initial_params_Venus.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    main_Earth()
    main_Mars()
    main_Venus()
