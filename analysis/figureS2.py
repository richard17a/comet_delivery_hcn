# pylint: disable=C0103,E1101,R0801

"""
Docstring for figureS2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def main():
    """
    Docstring for main
    """

    t = np.linspace(4.57, 0, 4500)[1:]

    time = np.loadtxt('./data/t_values.dat')
    Mc_25 = np.loadtxt('./data/Mc_25.dat')
    Mc_50 = np.loadtxt('./data/Mc_50.dat')
    Mc_75 = np.loadtxt('./data/Mc_75.dat')

    Mc_50 = Mc_50 / Mc_50[-1]
    Mc_25 = Mc_25 / Mc_25[-1]
    Mc_75 = Mc_75 / Mc_75[-1]

    Mc_25 = np.interp(
                t,
                time,
                Mc_25
            )

    Mc_50 = np.interp(
                t,
                time,
                Mc_50
            )

    Mc_75 = np.interp(
                t,
                time,
                Mc_75
            )

    t = t[0] - t

    _ = plt.figure(figsize=set_size('thesis'))

    plt.plot(t, Mc_50, label='Guo & Korenaga (2020)')
    plt.fill_between(t, Mc_25, Mc_75, alpha=0.25)

    plt.plot(t, 0.16 * (t[-1] - t), label='McCulloch & Bennett (1993)')

    plt.xlim(3.7, 4.5)
    plt.ylim(0, 1)

    plt.xlabel('Time [Gya]', fontsize=13)
    plt.ylabel('Normalised crustal mass', fontsize=13)

    plt.minorticks_on()

    plt.legend(frameon=False, loc='upper left')

    plt.gca().invert_xaxis()

    plt.gca().tick_params(axis='both', which='both', top=True, bottom=True,
                            left=True, right=True, direction='in')

    plt.savefig('./figures/GK_MB_comparison.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    main()
