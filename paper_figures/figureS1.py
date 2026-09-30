"""
Docstring for figureS1
"""
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent / "../analysis"
sys.path.append(str(src_path))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
import seaborn as sns
import pandas as pd
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def kokotanekova_data():
    """
    Docstring for kokotanekova_data
    """

    P_rot = np.array([11.083, 6.67, 7.9, 41.335, 8.948, 9.02, 9.725, 26.0, 9.5, 12.30, 12.75, 5.58,
                      40, 6.00, 15.6, 29.00, 13.47, 4.9, 12.055, 3.2, 6.6, 13.5, 24, 32, 6.22, 20.7,
                      16.4, 10.153, 10, 17.21, 10.5, 4.8, 32.853, 8.4096, 10.93, 8.16, 2.8])
    P_rot_err = np.array([0.003, 0.03, 1.35, 0.005, 0.001, 0.01, 3.075, 1.0, 0.2, 0.8, 0.03, 0.03,
                          0, 0.3, 0.1, 0.04, 0.017, 0.2, 0.001, 0.2, 1.0, 0.1, 5, 9, 0.05, 0.07,
                          0.1, 0.001, 5, 0.1, 1, 0.2, 0.002, 0.0012, 0.020, 0.24, 0.3])

    r_n = np.array([3.95, 2.23, 2.64, 2.83, 5.98, 2.95, 1.62, 2.5, 1.0, 2.15, 10.7, 1.65, 2.55,
                    0.56, 3.11, 2.97, 4.24, 0.61, 1.649, 0.41, 0.31, 1.98, 0.59, 0.26, 2.08, 2.27,
                    0.58, 2.31, 3.87, 4.79, 0.21, 0.21, 7.03, 2.48, 1.53, 1.54, 0.235])
    r_n_err = np.array([0.06, 0.14, 0.17, 0.1, 0.04, 0.19, 0.01, 0.1, 0.0, 0.17, 0.7, 0.11, 0.01,
                        0.04, 0.2, 0.20, 0.2, 0.03, 0.007, 0.02, 0.01, 0.05, 0.04, 0.01, 0.01,
                        0.14, 0.0018, 0.03, 0.24, 0.33, 0.02, 0.02, 0.48, 0.14, 0., 0.09, 0.085])

    return (r_n, r_n_err), (P_rot, P_rot_err)


def main():
    """
    Docstring for main
    """

    # ********* CONSTANTS *********
    rho_com = 600
    G = 6.67e-11
    # *****************************

    R_frag = np.logspace(1, 5, 1000)

    sigma_T = 1
    kappa = np.pi * G * (rho_com ** 2) * (R_frag ** 2) / (3 * sigma_T)
    P_crit_1 = np.pi * R_frag * np.sqrt(rho_com / sigma_T) / np.sqrt(1 + kappa)

    sigma_T = 10
    kappa = np.pi * G * (rho_com ** 2) * (R_frag ** 2) / (3 * sigma_T)
    P_crit_10 = np.pi * R_frag * np.sqrt(rho_com / sigma_T) / np.sqrt(1 + kappa)

    sigma_T = 100
    kappa = np.pi * G * (rho_com ** 2) * (R_frag ** 2) / (3 * sigma_T)
    P_crit_100 = np.pi * R_frag * np.sqrt(rho_com / sigma_T) / np.sqrt(1 + kappa)

    (r_n, r_n_err), (P_rot, P_rot_err) = kokotanekova_data()

    df = pd.read_csv('./data/LCDB_20250416_085650.txt', sep='\t', lineterminator='\r')
    D = df['Diam']
    P = df['Period']

    fig_width, fig_height = set_size('thesis')

    fig = plt.figure(figsize=(1.2 * fig_width, fig_height))

    gs = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)

    ax_main = fig.add_subplot(gs[0])

    ax_main.errorbar(r_n, P_rot, xerr=r_n_err, yerr=P_rot_err, fmt='.',
                     capsize=3, zorder=5, rasterized=True)
    ax_main.plot(D / 2, P, '.', rasterized=True, alpha=0.05)

    ax_main.plot(R_frag / 1e3, P_crit_1 / 60 / 60, c=cm.bamako(0.2),
                    label=r'$\sigma_T = 1\,$Pa', zorder=10)
    ax_main.plot(R_frag / 1e3, P_crit_10 / 60 / 60, c=cm.bamako(0.5),
                    label=r'$\sigma_T = 10\,$Pa', zorder=10)
    ax_main.plot(R_frag / 1e3, P_crit_100 / 60 / 60, c=cm.bamako(0.8),
                    label=r'$\sigma_T = 100\,$Pa', zorder=10)

    ax_main.set_xscale('log')
    ax_main.set_yscale('log')

    ax_main.set_xlim(1e-2, 1e1)
    ax_main.set_ylim(1e-1, 40)

    ax_main.set_xlabel(r'Radius [km]', fontsize=13)
    ax_main.set_ylabel(r'$P_{\rm rot}$ [hr]', fontsize=13)

    ax_kde = fig.add_subplot(gs[1], sharey=ax_main)

    sns.histplot(y=P_rot, ax=ax_kde, cumulative=True, element='step', fill=True,
                 stat='density', bins=20, alpha=0.5, zorder=15)
    sns.histplot(y=P, ax=ax_kde, cumulative=True, element='step', fill=True,
                 stat='density', bins=40, alpha=0.5, zorder=5)

    plt.setp(ax_kde.get_yticklabels(), visible=False)

    ax_kde.set_xlabel(r'${\rm CDF}$', fontsize=13)
    ax_kde.set_ylabel('')

    ax_kde.set_xticks([0, 1], labels=[0, 1])
    ax_kde.set_yticks([], labels=[])

    ax_kde.set_xlim(0, 1)

    ax_main.minorticks_on()
    ax_kde.minorticks_on()

    ax_main.set_xticks([0.01, 0.1, 1, 10], labels=[0.01, 0.1, 1, 10])
    ax_main.set_yticks([0.1, 1, 10], labels=[0.1, 1, 10])

    ax_main.tick_params(axis='both', which='both', top=True, bottom=True, left=True,
                        right=True, direction='in')
    ax_kde.tick_params(axis='both', which='both', top=True, bottom=True, left=True,
                       right=True, direction='in')

    ax_main.legend(frameon=False, loc='lower right', fontsize=11)

    plt.savefig('./figures/comet_ast_Prot.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    main()
