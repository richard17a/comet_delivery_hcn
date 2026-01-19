"""
Docstring for figure3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import set_size
from num_preb_envs import read_guokorenaga2020, load_simulation_data, vertical_gradient_span

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def main():

    t = np.linspace(4.57, 0, 4570)[1:]
    
    f_crust = 0.16 * 0.41 * t
    
    conc = "10mM"
    color = 'tab:blue'
    label = r'$c_{\rm HCN} > 10\,{\rm mM}$'
    all_runs = {}
    sim = load_simulation_data(conc)
    all_runs[conc] = sim[:, 1:]
    mask = (4.57 - t >= 3.5) & (4.57 - t <= 4.55)

    x_vals = 4.57 - t[mask]
    runs = all_runs[conc]
    
    mean_curve = runs.mean(axis=0)

    num_events = np.random.poisson(runs[:, mask] * f_crust[mask])

    totals = num_events.sum(axis=1)  # total impacts per run
    print(np.median(totals), len(totals[totals == 0]) / len(totals))

    mask = (num_events > 0)
    n_runs = mask.shape[0]
    has_event = mask.any(axis=1)

    last_idx_rev = np.argmax(mask[:, ::-1], axis=1)
    last_idx = mask.shape[1] - 1 - last_idx_rev

    last_time = np.full(n_runs, np.nan)
    last_time[has_event] = x_vals[last_idx[has_event]]

    _ = plt.figure(figsize=set_size('thesis'))
    ax = plt.gca()

    tmin, tmax = 3.5, 4.55
    t_grid = np.linspace(tmin, tmax, 400)

    frac_uncond = np.array([np.sum(last_time <= tt) / n_runs for tt in t_grid])
    level = 0.05
    t_10pct = np.interp(level, frac_uncond, t_grid)
    print(f'In ≤{100*level:.0f}% of simulations, the final successful cometary impacts occurs later than {t_10pct:.2f}Gya')

    n_bins = 100
    bin_edges = np.linspace(tmin, tmax, n_bins + 1)

    if n_runs > 0:
        frac_right = np.array([np.sum(last_time <= edge) / n_runs for edge in bin_edges[1:]])
    else:
        frac_right = np.zeros(n_bins)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_width = bin_edges[1] - bin_edges[0]

    ax.bar(bin_centers, frac_right, width=bar_width,
        align='center', color='tab:red', alpha=0.25, edgecolor='none', linewidth=0.,
        label='Frac. of sims with last impact ≤ t (CDF)')

    ax.set_xlabel(r'Time $t$ [Gya]', fontsize=13)
    ax.set_ylabel('Fraction of simulations with final\n successful impact later than $t$', fontsize=13, color='tab:red')
    ax.tick_params(axis='y', colors='tab:red')

    ax.set_ylim(0, 1)

    ax.set_xlim(3.5, 4.55)
    ax.set_xticks([4.55, 4.5, 4.3, 4.1, 3.9, 3.7, 3.5], labels=['', 4.5, 4.3, 4.1, 3.9, 3.7, 3.5])
    ax.invert_xaxis()

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(matplotlib.ticker.LinearLocator(numticks=22))
    ax.tick_params(axis='both', which='both', top=True, bottom=True, left=True, right=True, direction='in')

    median_last = np.nanmedian(last_time[has_event])
    p_lo, p_hi = np.nanpercentile(last_time[has_event], [16, 84])

    lower_err = median_last - p_lo
    upper_err = p_hi - median_last
    ax.axvline(median_last, color='tab:red', linestyle='-', lw=1)

    print(median_last, lower_err, upper_err)

    ax2 = ax.twinx()

    ax2.plot(4.57 - t, mean_curve * f_crust, color='tab:blue')
    ax2.set_ylabel('Number of comets delivering\n $>10\\,$mM HCN per Myr', fontsize=13, color='tab:blue', rotation=-90, labelpad=30)
    ax2.tick_params(axis='y', colors='tab:blue')

    ax2.set_yscale('log')
    ax2.set_ylim(bottom=6e-5, top=1e-2)

    ax2.set_xlim(ax.get_xlim())

    vertical_gradient_span(ax2, 3.5, 3.65, 'tab:brown', alpha_top=0.2, zorder=0, flip=False)
    ax2.text(3.581, 1e-3, "Stromatolites", rotation=90)

    vertical_gradient_span(ax2, 3.67, 3.73, 'tab:purple', alpha_top=0.2, zorder=0, flip=False)
    ax2.text(3.712, 6e-4, r"$^{13}\,$C depletion in graphite", rotation=90)

    vertical_gradient_span(ax2, 3.8, 3.85, 'tab:gray', alpha_top=0.2, zorder=0, flip=False)
    ax2.text(3.842, 8e-4, r"$^{13}\,$C depletion in apatite", rotation=90)

    vertical_gradient_span(ax2, 4.09, 4.33, 'tab:orange', alpha_top=0.2, zorder=0, flip=False)

    plt.text(4.20, 2.5e-3, "Phylogeny", rotation=90)
    plt.text(4.24, 3.1e-3, "LUCA", rotation=90)

    ax2.text(4.445, 8e-5, "most probable timing", rotation=90, color='tab:red')

    ax2.tick_params(axis='both', which='both', top=True, bottom=True, left=False, right=True, direction='in')

    # plt.savefig('./src/figures/main_fig.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    main()
