"""
Calculate the impact bombardment chronology used in the main analysis.

This module calculates the cumulative number of impacts as a function of time
and converts the impact chronology between different minimum impactor sizes
using an assumed size-frequency distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from cmcrameri import cm

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


def convert_sfd(F, D_new=0.2, D_old=10):

    alpha_1 = 1.78
    alpha_2 = 2.34
    alpha_3 = 5.78
    alpha_4 = 3.5

    D_12 = 2
    D_23 = 100
    D_34 = 300

    F_new = np.zeros_like(F)
    for i in range(len(F)):

        conv = ((100 ** (1 - alpha_2) - D_old ** (1 - alpha_2)) / (1 - alpha_2)) +\
                ((300 ** (1 - alpha_3) - 100 ** (1 - alpha_3)) / (1 - alpha_3)) * (D_23 ** (alpha_3 - alpha_2)) +\
                ((2000 ** (1 - alpha_4) - 300 ** (1 - alpha_4)) / (1 - alpha_4)) * (D_34 ** (alpha_4 - alpha_3))
        
        K2 = F[i] / conv
        K3 = K2 * (D_23 ** (alpha_3 - alpha_2))
        K4 = K2 * (D_34 ** (alpha_4 - alpha_3))
        K1 = K2 * (D_12 ** (alpha_1 - alpha_2))

        F_new[i] = (K1 / (1 - alpha_1)) * (2 ** (1 - alpha_1) - D_new ** (1 - alpha_1)) +\
                    (K2 / (1 - alpha_2)) * (100 ** (1 - alpha_2) - 2 ** (1 - alpha_2)) +\
                    (K3 / (1 - alpha_3)) * (300 ** (1 - alpha_3) - 100 ** (1 - alpha_3)) +\
                    (K4 / (1 - alpha_4)) * (2000 ** (1 - alpha_4) - 300 ** (1 - alpha_4))

    return F_new


def impact_threshold(t, F, threshold):

    residual = 1 - F / F[-1]

    mask = residual <= threshold
    if not np.any(mask):
        raise ValueError(f"Residual never reaches {threshold:.2f}")

    idx = np.argmax(mask)

    return idx


def main():

    Cs = 1./6.
    F1 = 6.5e3
    F2 = F1
    F3 = 4e-3
    alpha1 = 1
    alpha2 = 0.44
    tau1 = 7
    tau2 = 13

    t = np.linspace(4570, 0, 1000)
    F_10 = Cs * (F1 * np.exp(-(t/tau1) ** alpha1) + F2 * np.exp(-(t/tau2) ** alpha2) + F3 * (4570 - t))

    _ = plt.figure(figsize=set_size('thesis'))

    plt.plot(4.57 - t / 1e3, F_10, c=cm.bamako(0.2), label=r'$D>10\,$km')
    plt.plot(4.57 - t / 1e3, convert_sfd(F_10, 1), c=cm.bamako(0.5), label=r'$D>1\,$km')
    plt.plot(4.57 - t / 1e3, convert_sfd(F_10), c=cm.bamako(0.8), label=r'$D>0.1\,$km')
    
    plt.xlim(3.5, 4.57)
    plt.ylim(1e1, 1e6)

    plt.xlabel(r'Time [Myr]', fontsize=13)
    plt.ylabel(r'Cumulative impacts', fontsize=13)

    plt.yscale('log')

    plt.minorticks_on()

    plt.legend(frameon=False, fontsize=11)

    # plt.savefig('./src/figures/nesvorny_impacts.pdf', bbox_inches='tight', format='pdf')

    plt.show()

    F = convert_sfd(F_10)

    idx90 = impact_threshold(t, F, 0.9)
    idx99 = impact_threshold(t, F, 0.99)
    print(idx90, idx99)
    print(1 - F[idx90] / F[-1], 4.57 - t[idx90] / 1e3)
    print(1 - F[idx99] / F[-1], 4.57 - t[idx99] / 1e3)


if __name__ == "__main__":

    main()
