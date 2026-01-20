"""
Docstring for figureS3_S4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
import h5py
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def load_simulation_data(concentration, output_dir):
    """
    Docstring for load_simulation_data
    
    :param concentration: Description
    :param output_dir: Description
    """

    filename = f"{output_dir}/N1_{concentration}.h5"

    with h5py.File(filename, 'r') as f:
        data = f['N'][:]

    return np.vstack(data) if data.ndim == 3 else data


def load_simulation_data_brasser(concentration, output_dir):
    """
    Docstring for load_simulation_data_brasser
    
    :param concentration: Description
    :param output_dir: Description
    """

    filename = f"{output_dir}/N1_{concentration}.h5"

    with h5py.File(filename, 'r') as f:
        data = f['N'][:]

    data = np.vstack(data) if data.ndim == 3 else data
    return data[:, ::-1]


def figureS3():
    """
    Docstring for figureS3
    """

    t = np.linspace(4.57, 0, 4570)[1:]

    f_crust = 0.16 * 0.41 * t

    output_dirs = ["./data/num_preb_envs_100m_Brasser", "./data/num_preb_envs_100m"]

    concentrations = ["10mM"]
    colors = [cm.bamako(0.2), cm.bamako(0.8)]
    alphas = [1, 2]
    labels = [r'Brasser et al., 2020', r'Nesvorný et al., 2023']

    all_runs = {}
    for output_dir, alpha in zip(output_dirs, alphas):

        if alpha == 1:
            sim = load_simulation_data_brasser(concentrations[0], output_dir)
        else:
            sim = load_simulation_data(concentrations[0], output_dir)
        all_runs[alpha] = sim[:, 1:]
        print(f"conc={alpha}", f"num_runs={np.shape(all_runs[alpha])[0]}")

    _ = plt.figure(figsize=set_size('thesis'))

    mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.55)
    for output_dir, color, label, alpha in zip(output_dirs, colors, labels, alphas):
        runs = all_runs[alpha]

        mean_curve = runs.mean(axis=0)
        plt.plot(4.57 - t, mean_curve * f_crust,
                label=label, color=color, zorder=10)

        num_events = np.random.poisson(runs[:, mask] * f_crust[mask])

        totals = num_events.sum(axis=1)  # total impacts per run
        median_total = np.median(totals)

        print(np.median(totals), len(totals[totals == 0]) / len(totals))
        print(output_dir, f"median_total_impacts = {median_total}")

    plt.axhline(1, c='tab:gray', alpha=0.2, zorder=0)

    plt.xlim(3.5, 4.5)
    plt.ylim(2e-5, 1e-1)

    plt.xticks([3.5, 3.7, 3.9, 4.1, 4.3, 4.5])

    plt.yscale('log')

    plt.xlabel('Time [Gya]', fontsize=13)
    plt.ylabel('Number of comets delivering\n $>10\\,$mM HCN per Myr', fontsize=13)

    plt.minorticks_on()

    plt.gca().invert_xaxis()

    plt.gca().tick_params(axis='both', which='both', top=True, bottom=True,
                          left=True, right=True, direction='in')

    plt.legend(frameon=False, fontsize=12, loc='upper right')

    plt.savefig('./figures/brasser_nesvorny_comparison.pdf', bbox_inches='tight', format='pdf')

    plt.show()


def figureS4():
    """
    Docstring for figureS4
    """

    t = np.linspace(4.57, 0, 4570)[1:]

    f_crust = 0.16 * 0.41 * t

    output_dirs = ["./data/num_preb_envs_100m_Brasser", "./data/num_preb_envs_100m"]
    widths = [0.7, 0.5]

    concentrations = ["10mM"]
    colors = [cm.bamako(0.2), cm.bamako(0.8)]
    alphas = [1, 2]
    labels = [r'Brasser et al., 2020', r'Nesvorný et al., 2023']

    all_runs = {}
    for output_dir, alpha in zip(output_dirs, alphas):

        if alpha == 1:
            sim = load_simulation_data_brasser(concentrations[0], output_dir)
        else:
            sim = load_simulation_data(concentrations[0], output_dir)
        all_runs[alpha] = sim[:, 1:]
        print(f"conc={alpha}", f"num_runs={np.shape(all_runs[alpha])[0]}")

    _ = plt.figure(figsize=set_size('thesis'))

    mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.55)
    for output_dir, color, label, alpha, width in zip(output_dirs, colors, labels, alphas, widths):
        runs = all_runs[alpha]

        num_events = np.random.poisson(runs[:, mask] * f_crust[mask])

        totals = num_events.sum(axis=1)  # total impacts per run
        print(np.median(totals), np.mean(totals), len(totals[totals == 0]) / len(totals))

        minv = totals.min()
        maxv = totals.max()
        x = np.arange(minv, maxv + 1)

        counts_all = np.bincount(totals, minlength=maxv + 1)
        counts = counts_all[x]

        plt.bar(x, counts / totals.size,
            width=width, align='center',
            edgecolor='black', linewidth=0.6,
            color=color, alpha=0.75, label=label
        )

    plt.xlim(-0.5, 6.5)

    plt.ylabel(r'Fraction of Monte Carlo realisations', fontsize=13)
    plt.xlabel(r'Number of cometary impacts delivering $>10~$mM HCN', fontsize=13)

    plt.minorticks_on()

    plt.gca().tick_params(axis='both', which='both', top=True, bottom=True,
                          left=True, right=True, direction='in')

    plt.legend(frameon=False, fontsize=11)

    plt.savefig('./figures/successful_impacts_bar.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    figureS3()
    figureS4()
