import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cmcrameri import cm
import distributions, opik
import parametrised_chemistry
import atmos_entry_interp, hcn_surv_interp

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


def calc_HCN_survival(vels, diams, angles, chem='equil'):
    """
    Return % HCN survival (ignores size of comets at the moment)
    """

    HCN_surv = np.zeros(len(vels))

    for i in range(len(vels)):

        HCN_surv[i] = parametrised_chemistry.surv_param_steps(diams[i], vels[i], angles[i], chem)
        print(round(100 * i / len(vels), 2), end='%\r')

    return HCN_surv


def get_population(npoints):

    radii = distributions.generate_radii(npoints, seed=42)
    vels = distributions.generate_velocities(npoints, mean=28.2, std_dev=5., v_esc=11.19)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def get_population_DS09(npoints):

    radii = distributions.generate_radii(npoints, alpha1=1.78, alpha2=2.34, alpha3=5.74, seed=42)
    vels = distributions.generate_velocities(npoints, mean=28.2, std_dev=5., v_esc=11.19)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def get_population_21kms(npoints):

    radii = distributions.generate_radii(npoints, seed=42)
    vels = distributions.generate_velocities(npoints, mean=22, std_dev=5., v_esc=11.19)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def get_population_Mars(npoints):

    radii = distributions.generate_radii(npoints, seed=42, r_min=0.1)
    vels = distributions.generate_velocities(npoints, mean=19.7, std_dev=5., v_esc=5.03)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 5.03e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def get_population_Venus(npoints):

    radii = distributions.generate_radii(npoints, seed=42)
    vels = distributions.generate_velocities(npoints, mean=36.8, std_dev=5., v_esc=10.36)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 10.36e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def calc_surface_distributions(vels0, radii0, angles0, fpath="/data/rja92/HCN-Delivery-Paper/data/simulation_results.pkl"):

    m_frag_interp, v_frag_interp, theta_frag_interp = atmos_entry_interp.generate_interpolator(fpath)
    hcn_interp = hcn_surv_interp.generate_hcn_interpolator(fpath)

    queries = np.vstack([angles0, radii0 * 1e3, vels0 * 1e3]).T

    HCN_surv = hcn_interp(queries)

    masses = m_frag_interp(queries)
    vels = v_frag_interp(queries)
    angles = theta_frag_interp(queries) + np.pi

    radii = ( (3 * masses) / (4 * np.pi * 0.6e3)) ** (1./3.)

    angles = angles * 180 / np.pi
    vels = vels / 1e3
    HCN_surv = HCN_surv

    radii = radii / 1e3

    return vels, radii, angles, masses


def generate_axes(fig):

    gridspec = fig.add_gridspec(nrows=8, ncols=12)
    
    axes = {}
    axes[0] = fig.add_subplot(gridspec[0:4, 3:9])
    axes[1] = fig.add_subplot(gridspec[4:8, 6:11])
    axes[2] = fig.add_subplot(gridspec[4:8, 1:6])
    
    return axes


def plot_split(ax, parent, subsets, bins, xlabel, ylabel, flag_):

    colors = [cm.bamako(0.2), cm.bamako(0.5), cm.bamako(0.8)]

    weights_parent = np.ones_like(parent) / len(parent)
    weights_subs = [np.ones_like(sub) / len(parent) for sub in subsets]

    ax.hist(parent, bins=bins, weights=weights_parent,
            color='tab:gray', alpha=0.3)
    
    for sub, w, col in zip(subsets, weights_subs, colors):
        ax.hist(sub, bins=bins, weights=w,
                histtype='step', linewidth=1.5, color=col)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)

    ax.minorticks_on()

    if flag_ == 'size':
        ax.legend([
            r'$0.1\,{\rm km} < R_0 \leq 0.5\,{\rm km}$',
            r'$0.5\,{\rm km} < R_0 \leq 1\,{\rm km}$',
            r'$R_0 > 1\,{\rm km}$'
        ], frameon=False, fontsize=12)
    elif flag_ == 'vel':
        ax.legend([
            r'$v_0 \leq 20\,{\rm km\,s}^{-1}$',
            r'$20\,{\rm km\,s}^{-1} < v_0 \leq 30\,{\rm km\,s}^{-1}$',
            r'$v_0 > 30\,{\rm km\,s}^{-1}$'
        ], frameon=False, fontsize=12)
    

def plot_distributions(dists0, distsf01=None, distsf1=None, distsf10=None):

    m = np.pi * 0.6e3 * (100 ** 3) / 6
    
    labels = [r'top-of-atmosphere',
              r'Mars',
              r'Earth',
              r'Venus']

    # datasets = {
    #     labels[0]: dists0,
    #     labels[1]: distsf01,
    #     labels[2]: distsf1,
    #     labels[3]: distsf10
    # }
    datasets = {
        labels[0]: dists0,
        labels[2]: distsf1,
    }

    datasets = {k: v for k, v in datasets.items() if v is not None}
    colors = np.linspace(0, 0.85, len(datasets)-1)

    fig_width, fig_height = set_size('thesis')

    fig = plt.figure(figsize=(2 * fig_width, 2 * fig_height))
    axes = generate_axes(fig)

    bins_vel = 200
    vels_toa = dists0[0]

    v_toa, _ = np.histogram(vels_toa, bins=bins_vel, density=False)
    scale = 1.0 / v_toa.max()

    for i, (label, (vels, _, _, masses)) in enumerate(datasets.items()):
        w = np.ones_like(vels) * scale

        if label == labels[0]:
            axes[1].hist(
                vels,
                bins=bins_vel,
                density=False,
                weights=w,
                color='tab:gray',
                alpha=0.25,
                label=label
            )
        else:
            mask = masses > m
            axes[1].hist(
                vels[mask],
                bins=bins_vel,
                density=False,
                weights=w[mask],
                histtype='step',
                label=label,
                color=cm.bamako(colors[i-1])
            )

        if label == labels[2]:
            y_lims = axes[1].get_ylim()
    
    axes[1].set_xlim(0, 50)
    axes[1].set_ylim(y_lims)

    axes[1].set_xlabel(r'Velocity $[{\rm km\,s}^{-1}]$', fontsize=13)
    axes[1].set_ylabel(r'Frequency (normalised)', fontsize=13)

    axes[1].tick_params(which='both', direction='in', right=True, top=True)
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    
    axes[1].minorticks_on()

    bins = np.logspace(2, 4, 200)

    r_toa = datasets[labels[0]][1] * 1e3
    counts_toa, _ = np.histogram(r_toa, bins=bins)

    scale = 1.0 / counts_toa[0]

    for i, (label, (_, radii, _, masses)) in enumerate(datasets.items()):
        r = radii * 1e3

        if label != labels[0]:
            mask = masses > m
            r = r[mask]

        weights = np.ones_like(r) * scale

        if label == labels[0]:
            axes[0].hist(
                r, bins=bins,
                color='tab:gray', alpha=0.25,
                density=False, weights=weights,
                label=label
            )
        else:
            axes[0].hist(
                r, bins=bins,
                histtype='step',
                density=False, weights=weights,
                label=label,
                color=cm.bamako(colors[i-1])
            )

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')

    axes[0].set_xticks([1e2, 1e3, 1e4], labels=[0.1, 1.0, 10])
    axes[0].set_ylim(1e-3, axes[0].get_ylim()[1])

    axes[0].set_xlim(1e2, 1e4)

    axes[0].set_xlabel(r'Radius [km]', fontsize=13)
    axes[0].set_ylabel(r'Frequency (normalised)', fontsize=13)

    axes[0].legend(frameon=False, loc='upper right')

    axes[0].tick_params(which='both', direction='in', right=True, top=True)

    axes[0].minorticks_on()

    bins_ang = 100
    ang_toa = dists0[2]

    a_toa, _ = np.histogram(ang_toa, bins=bins_ang, density=False)
    scale = 1.0 / a_toa.max()
    
    for i, (label, (_, _, angles, masses)) in enumerate(datasets.items()):
        w = np.ones_like(angles) * scale

        if label == labels[0]:
            axes[2].hist(
                angles,
                bins=bins_ang,
                density=False,
                weights=w,
                color='tab:gray',
                alpha=0.25,
                label=label
            )
        else:
            mask = masses > m
            axes[2].hist(
                angles[mask],
                bins=bins_ang,
                density=False,
                weights=w[mask],
                histtype='step',
                label=label,
                color=cm.bamako(colors[i-1])
            )

            print(np.average(angles, weights=masses), np.average(angles[mask], weights=masses[mask]))
            print(np.average(angles), np.average(angles[mask]))

        if label == labels[2]:
            y_lims = axes[2].get_ylim()
    
    axes[2].set_xlim(15, 90)
    axes[2].set_ylim(y_lims)

    axes[2].set_xlabel(r'Angle [deg]', fontsize=13)
    axes[2].set_ylabel(r'Frequency (normalised)', fontsize=13)
    
    axes[2].minorticks_on()

    axes[2].tick_params(which='both', direction='in', right=True, top=True)

    axes[0].annotate('A', xy=(-0., 1.04), xycoords='axes fraction', weight='bold', fontsize=12)
    axes[1].annotate('C', xy=(0.96, 1.04), xycoords='axes fraction', weight='bold', fontsize=12)
    axes[2].annotate('B', xy=(-0., 1.04), xycoords='axes fraction', weight='bold', fontsize=12)

    plt.subplots_adjust(hspace=2)

    # plt.savefig('./src/figures/TOA_surf_dists_terrestrial.pdf', format='pdf', bbox_inches='tight')
    # plt.savefig('./src/figures/seminar_EARTH.pdf', format='pdf', bbox_inches='tight')

    plt.show()


def frac_survival_comparison():

    npoints = 2_000_000

    vels0, radii0, angles0 = get_population(npoints)

    velsf_Mars, radiif_Mars, anglesf_Mars, massesf_Mars = calc_surface_distributions(vels0, radii0, angles0, fpath="./data/simulation_results_Mars.pkl")
    velsf_Earth, radiif_Earth, anglesf_Earth, massesf_Earth = calc_surface_distributions(vels0, radii0, angles0, fpath="./data/simulation_results.pkl")
    velsf_Venus, radiif_Venus, anglesf_Venus, massesf_Venus = calc_surface_distributions(vels0, radii0, angles0, fpath="./data/simulation_results_Venus.pkl")

    m = np.pi * 0.6e3 * (100 ** 3) / 6

    bin_edges = np.logspace(2, 4, 50) / 1e3
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    f_survE, f_survM, f_survV = [], [], []
    for i in range(len(bin_edges) - 1):
        r_min, r_max = bin_edges[i], bin_edges[i+1]

        n_total = len(radii0[(radii0 >= r_min) & (radii0 < r_max)])
        n_surviveE = len(radii0[(radii0 >= r_min) & (radii0 < r_max) & (massesf_Earth > m)])
        n_surviveM = len(radii0[(radii0 >= r_min) & (radii0 < r_max) & (massesf_Mars > m)])
        n_surviveV = len(radii0[(radii0 >= r_min) & (radii0 < r_max) & (massesf_Venus > m)])

        if n_surviveE > 0:
            f_survE.append(n_surviveE / n_total)
        else:
            f_survE.append(np.nan)
        if n_surviveM > 0:
            f_survM.append(n_surviveM / n_total)
        else:
            f_survE.append(np.nan)
        if n_surviveV > 0:
            f_survV.append(n_surviveV / n_total)
        else:
            f_survV.append(np.nan)
    
    _ = plt.figure(figsize=set_size('thesis'))

    plt.plot(bin_centers, f_survM)
    plt.plot(bin_centers, f_survE)
    plt.plot(bin_centers, f_survV)

    plt.axvline(150 / 1e3, color='tab:gray')

    plt.xlim(1e-1, 1e1)
    plt.ylim(0, 1)

    plt.xscale('log')

    plt.minorticks_on()
    
    plt.show()


def main():

    # npoints = 20_000_000
    npoints = 2_000_000
    
    vels0, radii0, angles0 = get_population(npoints)
    masses0 = np.pi * 0.6e3 * ((radii0 * 1e3) ** 3) / 6

    velsf01, radiif01, anglesf01, massesf01 = calc_surface_distributions(vels0, radii0, angles0, fpath="./data/simulation_results_Mars.pkl")
    velsf1, radiif1, anglesf1, massesf1 = calc_surface_distributions(vels0, radii0, angles0, fpath="./data/simulation_results.pkl")
    velsf10, radiif10, anglesf10, massesf10 = calc_surface_distributions(vels0, radii0, angles0, fpath="./data/simulation_results_Venus.pkl")

    plot_distributions((vels0, radii0, angles0, masses0),
                        distsf01=(velsf01, radiif01, anglesf01, massesf01),
                        distsf1=(velsf1, radiif1, anglesf1, massesf1),
                        distsf10=(velsf10, radiif10, anglesf10, massesf10))


if __name__ == "__main__":

    main()
    # frac_survival_comparison()
