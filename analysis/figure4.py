# pylint: disable=E1101,R0914,R0801

"""
Docstring for figure4
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from nesvorny2023 import convert_sfd
import h5py
from utils import set_size
from num_preb_envs import req_delivery_efficiency

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


DENSITY = 0.6e3
MOLAR_MASS_HCN = 27.03
RADS = np.logspace(2, 4, 1000)
REQ_EFF = req_delivery_efficiency(DENSITY, MOLAR_MASS_HCN, RADS)
HCN_CONC = 1e-3

C_S = 1./4.9
F1, F2, F3 = 6.5e3, 6.5e3, 4e-3
ALPHA1, ALPHA2 = 1, 0.44
TAU1, TAU2 = 7, 13

T = np.linspace(4570, 10, 4500)
DELTA_T = (T[0] - T[-1]) / len(T)

F10 = C_S * (F1 * np.exp(-(T / TAU1) ** ALPHA1)
            + F2 * np.exp(-(T / TAU2) ** ALPHA2)
            + F3 * (4570 - T))
F = convert_sfd(F10, D_new=0.1)

F_Mars = F / 5

C_S = 1./6.
F1, F2, F3 = 6.5e3, 6.5e3, 4e-3
ALPHA1, ALPHA2 = 1, 0.44
TAU1, TAU2 = 7, 13

T = np.linspace(4570, 10, 4570)
DELTA_T = (T[0] - T[-1]) / len(T)

F10 = C_S * (F1 * np.exp(-(T / TAU1) ** ALPHA1)
            + F2 * np.exp(-(T / TAU2) ** ALPHA2)
            + F3 * (4570 - T))
F = convert_sfd(F10, D_new=0.1)

F_Earth = F


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


def main():
    """
    Docstring for main
    """

    t = np.linspace(4.57, 0, 4570)[1:]

    output_dirs = ["./data/num_preb_envs_Mars",
                   "./data/num_preb_envs_100m",
                   "./data/num_preb_envs_Venus"]
    # output_dirs = ["./data/num_preb_envs_Mars_1bar",
    #                "./data/num_preb_envs_100m",
    #                "./data/num_preb_envs_Venus"]

    concentrations = ["10mM"]
    colors = [cm.bamako(0.2), cm.bamako(0.5), cm.bamako(0.8)]
    alphas = [1]
    labels = ['Mars', 'Earth', 'Venus']

    _ = plt.figure(figsize=set_size('thesis'))

    mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.55)
    for output_dir, color, label in zip(output_dirs, colors, labels):
        counter = 0
        for conc, alpha in zip(concentrations, alphas):
            data = load_simulation_data(conc, output_dir)
            print(f"planet={label}, conc={conc}", f"num_runs={np.shape(data)[0]}")

            mean_curve = data[:, 1:].mean(axis=0)

            if counter == 0:
                plt.plot(4.57 - t, mean_curve, color=color, zorder=10, alpha=alpha, label=label)
            else:
                plt.plot(4.57 - t, mean_curve, color=color, zorder=10, alpha=alpha)

            totals = np.sum(data[:, 1:][:, mask], axis=1)
            median_total = np.median(totals)
            lo, hi = np.percentile(totals, [16, 84])
            print(f"num impacts={median_total}, lo={median_total - lo}, hi={hi - median_total}")

            x_vals = 4.57 - t[mask]
            cumperrun = np.cumsum(data[:, 1:][:, mask], axis=1)
            crossing_idxs = np.argmax(cumperrun >= 1, axis=1)
            never = cumperrun.max(axis=1) < 1
            crossing_idxs[never] = -1
            times = np.empty_like(crossing_idxs, dtype=float)
            times[crossing_idxs >= 0] = x_vals[crossing_idxs[crossing_idxs >= 0]]
            times[crossing_idxs < 0] = np.nan

            # 68% interval:
            p_lo, p_hi = np.nanpercentile(times, [16, 84])
            median_cross = np.nanmedian(times)
            lower_err = median_cross - p_lo
            upper_err = p_hi - median_cross
            print(median_cross, lower_err, upper_err)

            counter += 1

    plt.xlim(3.5, 4.55)
    plt.ylim(1e-3, 3e1)

    plt.xticks([3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.55], labels=[3.5, 3.7, 3.9, 4.1, 4.3, 4.5, ''])

    plt.gca().invert_xaxis()

    plt.yscale('log')

    plt.xlabel('Time [Gya]', fontsize=13)
    plt.ylabel('Number of comets delivering\n $>10\\,$mM HCN per Myr', fontsize=13)

    plt.minorticks_on()
    plt.gca().tick_params(axis='both', which='both', top=True, bottom=True,
                          left=True, right=True, direction='in')

    plt.legend(frameon=False, fontsize=12, loc='upper right')

    plt.savefig('./figures/num_impacts_terrestrial_comparison.pdf',
                bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    main()
