"""
This script plots the cumulative size-frequency distribution (SFD) of comets entering the inner
Solar System subject to disruption via rotational instability.
"""
import sys
from pathlib import Path

src_path = Path(__file__).resolve().parent / "../analysis"
sys.path.append(str(src_path))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def compute_slope_in_range(R, N, r_min, r_max):
    """
    Compute the best-fit slope and intercept of log10(N) vs log10(R) in a 
    specified radius range.
    
    Parameters
    ----------
    R : array
        Comet radii [km]
    N : array
        N(>R) for the cumulative SFD
    r_min : float
        Minimum radius (inclusive) for specified radius range.
    r_max : float
        Maximum radius (inclusive) for specified radius range.

    Returns
    -------
    slope : float
        Best-fit slope of log10(N) vs log10(R).
    intercept : float
        Best-fit intercept of log10(N) vs log10(R).
    """

    mask = (R >= r_min) & (R <= r_max)

    log_R = np.log10(R[mask])
    log_N = np.log10(N[mask])

    slope, intercept = np.polyfit(log_R, log_N, 1)

    return slope, intercept


def main():
    """
    Plots the cumulative size-frequency distribution (SFD) of comets entering the inner Solar
    System subject to disruption via rotational instability.
    """

    nfiles = 100000
    all_N_R = []

    for i in range(1, nfiles + 1):
        print(round(100 * i / nfiles, 2), end='%\r')
        fname = "./data/sfd_evolution/nrt_jewitt_12e3/nrt_" + str(i) + ".dat"
        fname = "../HCN-Delivery-Paper/data/nrt_jewitt_12e3/nrt_" + str(i) + ".dat"
        dout = np.loadtxt(fname, comments="#")
        N_R = dout[:, 2]
        all_N_R.append(N_R)

    all_N_R = np.array(all_N_R)
    avg_N_R_J = np.mean(all_N_R, axis=0)

    ranges = [(0.1, 1), (1, 50), (50, 150)]

    _ = plt.figure(figsize=set_size('thesis'))

    data = np.loadtxt("./data/sfd_evolution/size_distribution.dat", comments="#")

    R = data[:, 1]
    N_R = data[:, 2]

    plt.plot(R, np.cumsum(N_R[::-1])[::-1] / np.cumsum(N_R[::-1])[::-1][0],
             c='tab:gray', label='Input (Modern JFCs)', alpha=0.5)

    counter = 0
    xvals = [0.5, 5, 80]
    heights = [0.02, 0.0002, 3e-8]
    angles = [-10, -20, 0]
    for r_min, r_max in ranges:
        slope, _ = compute_slope_in_range(R, N_R, r_min, r_max)
        plt.text(xvals[counter], heights[counter], round(slope, 2),
                 color='tab:gray', rotation=angles[counter])
        print(222, np.abs(slope))
        counter += 1

    avg_cum = np.cumsum(avg_N_R_J[::-1])[::-1]
    plt.plot(R, avg_cum / avg_cum[0], c=cm.bamako(0.5), label='Output (inner Solar System)')

    counter = 0
    xvals = [0.5, 5, 80]
    heights = [0.35, 0.04, 0.0001]
    angles = [-5, -10, 0]
    for r_min, r_max in ranges:
        slope, _ = compute_slope_in_range(R, avg_cum, r_min, r_max)
        plt.text(xvals[counter], heights[counter], round(slope, 2),
                 color=cm.bamako(0.5), rotation=angles[counter])
        print(222, np.abs(slope))
        counter += 1

    plt.xlabel(r'Radius $R$ [km]', fontsize=13)
    plt.ylabel(r'$N(>R)$ (normalised)', fontsize=13)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(1e-1, 150)
    plt.ylim(1e-8, 1e0)

    plt.gca().yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs="auto"))

    plt.legend(frameon=False, fontsize=11, loc='lower left')

    plt.gca().tick_params(axis='both', which='both', top=True, bottom=True,
                          left=True, right=True, direction='in')

    plt.savefig('./figures/SFD_evol.pdf', format='pdf', bbox_inches='tight')

    plt.show()


def disisto_comparison():
    """
    Compares the cumulative size-frequency distribution (SFD) of comets entering the inner Solar
    System subject to disruption via rotational instability (Jewitt+ 2021) vs the Di Sisto+2009 
    disruption model.
    """

    nfiles = 100000
    all_N_R = []

    for i in range(1, nfiles + 1):
        print(round(100 * i / nfiles, 2), end='%\r')
        fname = "./data/sfd_evolution/nrt_jewitt_12e3/nrt_" + str(i) + ".dat"
        dout = np.loadtxt(fname, comments="#")
        N_R = dout[:, 2]
        all_N_R.append(N_R)

    all_N_R = np.array(all_N_R)
    avg_N_R_J = np.mean(all_N_R, axis=0)

    ranges = [(0.1, 1), (1, 50), (50, 150)]

    _ = plt.figure(figsize=set_size('thesis'))

    data = np.loadtxt("./data/sfd_evolution/size_distribution.dat", comments="#")

    R = data[:, 1]
    N_R = data[:, 2]

    plt.plot(R, np.cumsum(N_R[::-1])[::-1] / np.cumsum(N_R[::-1])[::-1][0],
             c='tab:gray', label='Input (Modern JFCs)', alpha=0.5)

    counter = 0
    xvals = [0.5, 5, 80]
    heights = [0.02, 0.0002, 3e-8]
    angles = [-10, -20, 0]
    for r_min, r_max in ranges:
        slope, _ = compute_slope_in_range(R, N_R, r_min, r_max)
        plt.text(xvals[counter], heights[counter], round(slope, 2),
                 color='tab:gray', rotation=angles[counter])
        print(222, np.abs(slope))
        counter += 1

    avg_cum = np.cumsum(avg_N_R_J[::-1])[::-1]
    plt.plot(R, avg_cum / avg_cum[0], c=cm.bamako(0.5), label='Output (Jewitt 2021)')

    counter = 0
    xvals = [0.5, 5, 80]
    heights = [0.35, 0.04, 0.0001]
    angles = [-5, -10, 0]
    for r_min, r_max in ranges:
        slope, _ = compute_slope_in_range(R, avg_cum, r_min, r_max)
        plt.text(xvals[counter], heights[counter], round(slope, 2),
                 color=cm.bamako(0.5), rotation=angles[counter])
        print(222, np.abs(slope))
        counter += 1

    all_N_R = []

    for i in range(1, nfiles + 1):
        print(round(100 * i / nfiles, 2), end='%\r')
        fname = "./data/sfd_evolution/nrt_disisto_12e3/nrt_" + str(i) + ".dat"
        dout = np.loadtxt(fname, comments="#")
        N_R = dout[:, 2]
        all_N_R.append(N_R)

    all_N_R = np.array(all_N_R)
    avg_N_R_J = np.mean(all_N_R, axis=0)

    avg_cum = np.cumsum(avg_N_R_J[::-1])[::-1]
    plt.plot(R, avg_cum / avg_cum[0], c=cm.bamako(0.8), label='Output (Di Sisto et al. 2009)')

    counter = 0
    xvals = [0.5, 5, 80]
    heights = [0.35, 0.04, 0.0001]
    angles = [-5, -10, 0]
    for r_min, r_max in ranges:
        slope, _ = compute_slope_in_range(R, avg_cum, r_min, r_max)
        plt.text(xvals[counter], heights[counter], round(slope, 2),
                 color=cm.bamako(0.8), rotation=angles[counter])
        print(222, np.abs(slope))
        counter += 1

    plt.xlabel(r'Radius $R$ [km]', fontsize=13)
    plt.ylabel(r'$N(>R)$ (normalised)', fontsize=13)

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(1e-1, 150)
    plt.ylim(1e-8, 1e0)

    plt.gca().yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs="auto"))

    plt.legend(frameon=False, fontsize=11, loc='lower left')

    # plt.savefig('./figures/SFD_evol.pdf', format='pdf', bbox_inches='tight')

    plt.show()


if __name__ == "__main__":

    main()
    # disisto_comparison()
