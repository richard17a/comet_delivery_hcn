# pylint: disable=R0914,C0103,R0915,R0801

"""
Docstring for hcn_surv_interp
"""

import pickle
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import RegularGridInterpolator
import parametrised_chemistry
import distributions
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def calc_HCN_survival(vels, diams, angles, chem='equil'):
    """
    Return % HCN survival from McDonald+ 2025 (Icarus)
    """

    HCN_surv = np.zeros(len(vels))

    for i, vel in enumerate(vels):

        HCN_surv[i] = parametrised_chemistry.surv_param_steps(diams[i], vel, angles[i], chem)

    return HCN_surv


def generate_hcn_interpolator(fpath="./data/atmospheric_entry/simulation_results.pkl",
                              chem='equil'):
    """
    Docstring for generate_hcn_interpolator (using equilibrium chemistry from McDonald+ 2025)
    
    :param fpath: Description
    :param chem: Description
    """

    with open(fpath, "rb") as f:
        results = pickle.load(f)

    theta_vals = sorted(set(key[0] for key in results.keys()))
    r_vals = sorted(set(key[1] for key in results.keys()))
    v_vals = sorted(set(key[2] for key in results.keys()))

    hcn_values = np.zeros((len(theta_vals), len(r_vals), len(v_vals)))

    for (theta, r, v), data in results.items():
        i = theta_vals.index(theta)
        j = r_vals.index(r)
        k = v_vals.index(v)

        m0 = 4 * np.pi * 0.6e3 * (r ** 3) / 3

        if len(data['masses']) == 0:
            hcn_values[i, j, k] = 0

        else:
            hcn_surv = 0
            for counter in range(len(data['masses'])):

                mass = data['masses'][counter]
                vel = data['vels'][counter]
                angle = (data["angles"][counter] + np.pi) * 180 / np.pi

                rad = ( (3 * mass) / (4 * np.pi * 0.6e3)) ** (1./3.)

                if (rad >= 100) and (vel >= 10e3):
                    hcn_surv += parametrised_chemistry.surv_param_steps(2 * rad / 1e3,\
                                                            vel / 1e3, angle, chem) * mass
                else:
                    hcn_surv += 0

            hcn_values[i, j, k] = hcn_surv / m0

    hcn_interp = RegularGridInterpolator((theta_vals, r_vals, v_vals), hcn_values)

    return hcn_interp


def calc_avg_HCN_ang(ang0, vels, npoints, hcn_interp):
    """
    Docstring for calc_avg_HCN_ang
    
    :param ang0: Description
    :param vels: Description
    :param npoints: Description
    :param hcn_interp: Description
    """

    radii = distributions.generate_radii(npoints, seed=42)
    angles = ang0 * np.ones(len(vels))

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid] * 1e3
    vels_valid = vels[valid] * 1e3

    queries = np.vstack([angles_valid, radii_valid, vels_valid]).T

    HCN_surv = hcn_interp(queries)

    return np.nanmean(HCN_surv), np.nanstd(HCN_surv)


def parallel_calc_avg_HCN_ang(args):
    """
    Docstring for parallel_calc_avg_HCN_ang
    
    :param args: Description
    """

    ang, vels, npoints, hcn_interp = args

    return calc_avg_HCN_ang(ang, vels, npoints, hcn_interp)


def calc_avg_HCN_vel(vel, npoints, hcn_interp):
    """
    Docstring for calc_avg_HCN_vel
    
    :param vel: Description
    :param npoints: Description
    :param hcn_interp: Description
    """

    # radii = distributions.generate_radii(3., 1e2, 1e4, npoints) / 1e3
    radii = distributions.generate_radii(npoints, seed=42)
    angles = distributions.generate_angles(npoints) * 180 / np.pi
    vels = vel * np.ones(len(radii))

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid] * 1e3
    vels_valid = vels[valid] * 1e3

    queries = np.vstack([angles_valid, radii_valid, vels_valid]).T

    HCN_surv = hcn_interp(queries)

    return np.nanmean(HCN_surv), np.nanstd(HCN_surv)


def parallel_calc_avg_HCN_vel(args):
    """
    Docstring for parallel_calc_avg_HCN_vel
    
    :param args: Description
    """

    vel, npoints, hcn_interp = args

    return calc_avg_HCN_vel(vel, npoints, hcn_interp)


def calc_avg_HCN_rad(R0, vels, npoints, hcn_interp):
    """
    Docstring for calc_avg_HCN_rad
    
    :param R0: Description
    :param vels: Description
    :param npoints: Description
    :param hcn_interp: Description
    """

    angles = distributions.generate_angles(npoints) * 180 / np.pi
    radii = R0 * np.ones(len(vels)) / 1e3

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid] * 1e3
    vels_valid = vels[valid] * 1e3

    queries = np.vstack([angles_valid, radii_valid, vels_valid]).T

    HCN_surv = hcn_interp(queries)

    return np.nanmean(HCN_surv), np.nanstd(HCN_surv)


def parallel_calc_avg_HCN_rad(args):
    """
    Docstring for parallel_calc_avg_HCN_rad
    
    :param args: Description
    """

    rad, vels, npoints, hcn_interp = args

    return calc_avg_HCN_rad(rad, vels, npoints, hcn_interp)


def main():
    """
    Docstring for main
    """

    npoints = 10000

    vels = distributions.generate_velocities(npoints, mean=19., std_dev=5., v_esc=11.19)

    radii = np.logspace(2, 4, 1000)

    fpaths = ["./data/atmospheric_entry/simulation_results_01.pkl",
              "./data/atmospheric_entry/simulation_results.pkl",
              "./data/atmospheric_entry/simulation_results_10.pkl"]
    rho0s = [0.1, 1, 10]

    fnames = ["./data/atmospheric_entry/avg_HCN_rad_01.npy",
              "./data/atmospheric_entry/avg_HCN_rad_1.npy",
              "./data/atmospheric_entry/avg_HCN_rad_10.npy"]

    _ = plt.figure(figsize=set_size('thesis'))

    for i, fpath in enumerate(fpaths):
        hcn_interp = generate_hcn_interpolator(fpath)
        args_list = [(rad, vels, npoints, hcn_interp) for rad in radii]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(parallel_calc_avg_HCN_rad, args_list)

        HCN_mean, _ = np.array(results).T

        np.save(fnames[i], results)

        plt.plot(radii, HCN_mean,
                 label=rf'$\rho_{{\rm atm, 0}} = {rho0s[i]}\,{{\rm kg\,m^{{-3}}}}$')

    plt.xscale('log')

    plt.xlabel(r'Initial radius [m]', fontsize=13)
    plt.ylabel(r'HCN Survival [%]', fontsize=13)

    plt.xlim(1e2, 1e4)
    plt.ylim(0, 40)

    plt.minorticks_on()

    plt.legend(frameon=False)

    # plt.savefig('./figures/avg_HCN_survival_size.pdf', bbox_inches='tight', format='pdf')

    plt.show()

    print("\n====  radii plotted  ====\n")

    angles = np.linspace(15, 90, 1000)

    fnames = ["./data/atmospheric_entry/avg_HCN_ang_01.npy",
              "./data/atmospheric_entry/avg_HCN_ang_1.npy",
              "./data/atmospheric_entry/avg_HCN_ang_10.npy"]

    _ = plt.figure(figsize=set_size('thesis'))

    for i, fpath in enumerate(fpaths):
        hcn_interp = generate_hcn_interpolator(fpath)
        args_list = [(ang, vels, npoints, hcn_interp) for ang in angles]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(parallel_calc_avg_HCN_ang, args_list)

        HCN_mean, _ = np.array(results).T

        plt.plot(angles, HCN_mean,
                 label=rf'$\rho_{{\rm atm, 0}} = {rho0s[i]}\,{{\rm kg\,m^{{-3}}}}$')

        np.save(fnames[i], results)

    plt.xlabel(r'Initial angle [deg]', fontsize=13)
    plt.ylabel(r'HCN Survival [%]', fontsize=13)

    plt.xlim(15, 90)
    plt.ylim(0, 40)

    plt.minorticks_on()

    plt.legend(frameon=False)

    # plt.savefig('./figures/avg_HCN_survival_angle.pdf', bbox_inches='tight', format='pdf')

    plt.show()

    print("\n====  angles plotted  ====\n")

    vels = np.linspace(11.19, 70, 1000)

    fnames = ["./data/atmospheric_entry/avg_HCN_vel_01.npy",
              "./data/atmospheric_entry/avg_HCN_vel_1.npy",
              "./data/atmospheric_entry/avg_HCN_vel_10.npy"]

    _ = plt.figure(figsize=set_size('thesis'))

    for i, fpath in enumerate(fpaths):
        hcn_interp = generate_hcn_interpolator(fpath)
        args_list = [(vel, npoints, hcn_interp) for vel in vels]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(parallel_calc_avg_HCN_vel, args_list)

        HCN_mean, _ = np.array(results).T

        plt.plot(vels, HCN_mean,
                 label=rf'$\rho_{{\rm atm, 0}} = {rho0s[i]}\,{{\rm kg\,m^{{-3}}}}$')

        np.save(fnames[i], results)

    plt.axvline(11.19, c='tab:gray', ls='--')

    plt.xlabel(r'Initial velocity [${\rm km\,s}^{-1}$]', fontsize=13)
    plt.ylabel(r'HCN Survival [%]', fontsize=13)

    plt.xlim(10, 70)
    plt.ylim(0, 40)

    plt.minorticks_on()

    plt.legend(frameon=False)

    # plt.savefig('./figures/avg_HCN_survival_vel.pdf', bbox_inches='tight', format='pdf')

    plt.show()

    print("\n====  velocities plotted  ====\n")

    print("\n====  main.py  ====\n")


if __name__ == "__main__":

    main()
