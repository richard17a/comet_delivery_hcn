"""
Docstring for figureS8
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from utils import set_size
from num_preb_envs import load_simulation_data

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def main():
    """
    Docstring for main
    """

    t = np.linspace(4.57, 0, 4570)[1:]

    f_crust = 0.16 * 0.41 * t

    concs = ["01mM", "1mM", "10mM"]
    labels = [r'$c_{\rm HCN} = 0.1~{\rm mM}$', r'$c_{\rm HCN} = 1.0~{\rm mM}$', r'$c_{\rm HCN} = 10.0~{\rm mM}$']
    colors = [cm.bamako(0.2), cm.bamako(0.5), cm.bamako(0.8)]

    _ = plt.figure(figsize=set_size('thesis'))
    ax = plt.gca()

    for conc, label, col in zip(concs, labels, colors):

        all_runs = {}
        sim = load_simulation_data(conc)
        all_runs[conc] = sim[:, 1:]
        mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.55)

        x_vals = 4.57 - t[mask]
        runs = all_runs[conc]

        mean_curve = runs.mean(axis=0)

        num_events = np.random.poisson(runs[:, mask] * f_crust[mask])

        totals = num_events.sum(axis=1)  # total impacts per run
        print(conc, np.median(totals), len(totals[totals == 0]) / len(totals))

        mask = num_events > 0
        n_runs = mask.shape[0]
        has_event = mask.any(axis=1)

        last_idx_rev = np.argmax(mask[:, ::-1], axis=1)
        last_idx = mask.shape[1] - 1 - last_idx_rev

        last_time = np.full(n_runs, np.nan)
        last_time[has_event] = x_vals[last_idx[has_event]]

        ax.plot(4.57 - t, mean_curve * f_crust, label=label, color=col)

    ax.set_xlabel(r'Time $t$ [Gya]', fontsize=13)
    ax.set_ylabel(r'Number of comets delivering' + '\n' +
                  r'$>c_{\rm HCN}~{\rm mM}$ HCN per Myr', fontsize=13)

    ax.set_xlim(3.5, 4.55)
    ax.set_xticks([4.55, 4.5, 4.3, 4.1, 3.9, 3.7, 3.5],
                  labels=['', 4.5, 4.3, 4.1, 3.9, 3.7, 3.5])
    ax.invert_xaxis()

    ax.set_yscale('log')
    ax.set_ylim(bottom=6e-5, top=1e-0)

    ax.set_xlim(ax.get_xlim())

    plt.legend(frameon=False, loc='upper right', fontsize=12)

    ax.minorticks_on()
    ax.tick_params(axis='both', which='both', top=True, bottom=True,
                    left=True, right=True, direction='in')

    plt.savefig('./figures/fig_S8.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    main()
