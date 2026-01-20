"""
Docstring for figure2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from utils import set_size
from impactor_distributions import get_population, calc_surface_distributions

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def generate_axes(fig):
    """
    Docstring for generate_axes
    
    :param fig: Description
    """

    gridspec = fig.add_gridspec(nrows=8, ncols=12)

    axes = {}
    axes[0] = fig.add_subplot(gridspec[0:4, 3:9])
    axes[1] = fig.add_subplot(gridspec[4:8, 6:11])
    axes[2] = fig.add_subplot(gridspec[4:8, 1:6])

    return axes


def plot_distributions(dists0, distsf01=None, distsf1=None, distsf10=None):
    """
    Docstring for plot_distributions
    
    :param dists0: Description
    :param distsf01: Description
    :param distsf1: Description
    :param distsf10: Description
    """

    m = np.pi * 0.6e3 * (100 ** 3) / 6

    labels = [r'top-of-atmosphere',
              r'Mars',
              r'Earth',
              r'Venus']

    datasets = {
        labels[0]: dists0,
        labels[1]: distsf01,
        labels[2]: distsf1,
        labels[3]: distsf10
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

    plt.show()


def main():
    """
    Docstring for main
    """

    npoints = 20_000_000

    vels0, radii0, angles0 = get_population(npoints)
    masses0 = np.pi * 0.6e3 * ((radii0 * 1e3) ** 3) / 6

    velsf01, radiif01, anglesf01, massesf01 =\
        calc_surface_distributions(vels0,
                                    radii0,
                                    angles0,
                                    fpath="./data/atmospheric_entry/simulation_results_Mars.pkl")
    velsf1, radiif1, anglesf1, massesf1 =\
        calc_surface_distributions(vels0,
                                    radii0,
                                    angles0,
                                    fpath="./data/atmospheric_entry/simulation_results.pkl")
    velsf10, radiif10, anglesf10, massesf10 =\
        calc_surface_distributions(vels0,
                                    radii0,
                                    angles0,
                                    fpath="./data/atmospheric_entry/simulation_results_Venus.pkl")

    plot_distributions((vels0, radii0, angles0, masses0),
                        distsf01=(velsf01, radiif01, anglesf01, massesf01),
                        distsf1=(velsf1, radiif1, anglesf1, massesf1),
                        distsf10=(velsf10, radiif10, anglesf10, massesf10))


if __name__ == "__main__":

    main()
