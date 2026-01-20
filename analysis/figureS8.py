# pylint: disable=R0914,C0103,E1101,R0915

"""
Module Docstring
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from astropy.io.votable import parse_single_table
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def main():
    """
    Docstring for main
    """

    table = parse_single_table("./data/PLANETS.vot").to_table()

    R_pl = table['Planet Radius [Rjup]'] * 11.2
    R_pl_hi = table['Planet Radius - Upper Unc [Rjup]'] * 11.2
    R_pl_lo = table['Planet Radius - Lower Unc [Rjup]'] * 11.2

    M_pl = table['Planet Mass [Mjup]'] * 317.8
    M_pl_lo = table['Planet Mass - Lower Unc [Mjup]'] * 317.8
    M_pl_hi = table['Planet Mass - Upper Unc [Mjup]'] * 317.8

    A1, b1 = 1.02, 0.27
    A2, b2 = 0.56, 0.67
    A3, b3 = 18.6, -0.06
    M_b1 = 4.37
    M_b2 = 127.0

    G = 6.67e-11
    M_E = 5.97e24
    R_E = 6371e3

    Ms = np.logspace(-2, 2, 1000)

    R = np.empty_like(Ms)
    mask1 = Ms < M_b1
    mask2 = (Ms >= M_b1) & (Ms <= M_b2)
    mask3 = Ms > M_b2

    R[mask1] = A1 * Ms[mask1] ** b1
    R[mask2] = A2 * Ms[mask2] ** b2
    R[mask3] = A3 * Ms[mask3] ** b3

    mask = R_pl < 10

    v_esc = np.sqrt(2 * G * M_pl * M_E / R_pl / R_E) / 1e3
    frac_v_err = 0.5 * np.sqrt(((M_pl_lo + M_pl_hi) / (2 * M_pl))**2 +
                            ((R_pl_lo + R_pl_hi) / (2 * R_pl))**2)
    v_esc_err = v_esc * frac_v_err

    fig_width, fig_height = set_size('thesis')

    _, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                figsize=(fig_width, 1.5 * fig_height),
                                gridspec_kw={'height_ratios': [1, 1]})

    ax1.errorbar(
        R_pl[mask], M_pl[mask],
        xerr=[R_pl_lo[mask], R_pl_hi[mask]],
        yerr=[M_pl_lo[mask], M_pl_hi[mask]],
        fmt='o', capsize=3, color=cm.bamako(0.3), alpha=0.25
    )

    ax1.scatter(0.532, 0.107, marker='s', label='Mars', color='tab:red')
    ax1.scatter(1.0,   1.0,   marker='o', label='Earth', color='tab:red')
    ax1.scatter(0.950, 0.815, marker='*', label='Venus', color='tab:red')

    ax1.plot(R[R < 10], Ms[R < 10], zorder=10, c='k')

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.set_xlim(0.5, 10)
    ax1.set_ylim(bottom=0.05)

    ax1.set_ylabel(r'Mass $[M_E]$', fontsize=13)
    ax1.legend(frameon=False, fontsize=11, loc='upper left')

    ax1.tick_params(labelbottom=False)
    ax1.tick_params(axis='both', which='both', top=True, bottom=True,
                    left=True, right=True, direction='in')

    ax2.errorbar(
        R_pl[mask], v_esc[mask],
        xerr=[R_pl_lo[mask], R_pl_hi[mask]],
        yerr=[v_esc_err[mask], v_esc_err[mask]],
        fmt='o', capsize=3, color=cm.bamako(0.3), alpha=0.25
    )

    ax2.scatter(0.532, 5.03,  marker='s', label='Mars',  color='tab:red')
    ax2.scatter(1.0,   11.19, marker='o', label='Earth',  color='tab:red')
    ax2.scatter(0.950, 10.46, marker='*', label='Venus',  color='tab:red')

    ax2.plot(R[R < 10], np.sqrt(2 * G * Ms[R < 10] * M_E / R[R < 10] / R_E) / 1e3,
            zorder=10, c='k')

    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlim(0.5, 10)
    ax2.set_ylim(bottom=5)

    ax2.set_xticks([1, 10], labels=[r'1', r'10'])
    ax2.set_yticks([10, 100], labels=[r'10', r'100'])

    ax2.set_xlabel(r'Radius $[R_E]$', fontsize=13)
    ax2.set_ylabel(r'Escape velocity $[{\rm km\,s}^{-1}]$', fontsize=13)

    ax2.tick_params(axis='both', which='both', top=True, bottom=True,
                    left=True, right=True, direction='in')

    plt.tight_layout()

    plt.savefig('./figures/exoplanet_vesc.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    main()
