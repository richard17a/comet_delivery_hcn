"""
Docstring for atmos_entry_interp
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import RegularGridInterpolator
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def generate_interpolator(fpath="./data/atmospheric_entry/simulation_results.pkl"):
    """
    Docstring for generate_interpolator
    
    :param fpath: Description
    """

    with open(fpath, "rb") as f:
        results = pickle.load(f)

    theta_vals = sorted(set(key[0] for key in results.keys()))
    r_vals = sorted(set(key[1] for key in results.keys()))
    v_vals = sorted(set(key[2] for key in results.keys()))

    m_frag_values = np.zeros((len(theta_vals), len(r_vals), len(v_vals)))
    v_frag_values = np.zeros((len(theta_vals), len(r_vals), len(v_vals)))
    theta_frag_values = np.zeros((len(theta_vals), len(r_vals), len(v_vals)))

    for (theta, r, v), data in results.items():

        i = theta_vals.index(theta)
        j = r_vals.index(r)
        k = v_vals.index(v)

        if len(data['masses']) == 0:

            m_frag_values[i, j, k] = 0
            v_frag_values[i, j, k] = 0
            theta_frag_values[i, j, k] = 0
        else:

            m_frag_values[i, j, k] = np.mean(data["masses"])
            v_frag_values[i, j, k] = np.sum(np.array(data["vels"]) *\
                                            np.array(data["masses"])) / np.sum(data["masses"])
            theta_frag_values[i, j, k] = np.sum(np.array(data["angles"]) *\
                                                np.array(data["masses"])) / np.sum(data["masses"])

    m_frag_interp = RegularGridInterpolator((theta_vals, r_vals, v_vals), m_frag_values)
    v_frag_interp = RegularGridInterpolator((theta_vals, r_vals, v_vals), v_frag_values)
    theta_frag_interp = RegularGridInterpolator((theta_vals, r_vals, v_vals), theta_frag_values)

    return m_frag_interp, v_frag_interp, theta_frag_interp


def interp_validation():
    """
    Docstring for interp_validation
    """

    m_frag_interp, _, _ = generate_interpolator()

    with open("./data/atmospheric_entry/simulation_results.pkl", "rb") as f:
        results = pickle.load(f)

    r_vals = []
    v_vals = []
    m_vals = []

    theta_vals = sorted(set(key[0] for key in results.keys()))

    for (theta, r, v), data in results.items():
        if theta == theta_vals[0]:

            m0 = 4 * np.pi * 0.6e3 * (r ** 3) / 3

            r_vals.append(r)
            v_vals.append(v)
            m_vals.append(np.mean(data["masses"]) / m0)

    _ = plt.figure(figsize=set_size('thesis'))

    sc = plt.scatter(r_vals, np.array(v_vals) / 1e3, c=m_vals, vmin=0, vmax=1)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"Final mass [$M_0$]", fontsize=13)

    query_points = [
        (6000, 50e3),
        (2000, 20e3),
        (470, 20e3),
        (150, 60e3),
        (350, 45e3),
        (610, 60e3),
        (110, 12e3)
    ]

    for r, v in query_points:
        query_point = np.array([theta_vals[0], r, v])
        m_frag_estimate = m_frag_interp(query_point)
        norm_mass = m_frag_estimate / (4 * np.pi * 0.6e3 * (r ** 3) / 3)

        plt.scatter(r, v / 1e3, c=norm_mass, vmin=0, vmax=1, cmap='viridis', edgecolor='k')

    plt.xlabel(r"Initial Radius [m]", fontsize=13)
    plt.ylabel(r"Initial velocity [${\rm km\,s}^{-1}$]", fontsize=13)

    plt.xscale("log")

    plt.minorticks_on()

    plt.title(fr'$\theta_0={theta_vals[0]}^\circ$')

    # plt.savefig('./figures/interpolation_validation.pdf', bbox_inches='tight', format='pdf')

    plt.show()


def interp_validation_angle():
    """
    Docstring for interp_validation_angle
    """

    _, _, theta_frag_interp = generate_interpolator()

    with open("./data/atmospheric_entry/simulation_results.pkl", "rb") as f:
        results = pickle.load(f)

    r_vals = []
    v_vals = []
    theta_vals = []

    initial_rads = sorted(set(key[1] for key in results.keys()))

    for (theta, r, v), data in results.items():
        if r == initial_rads[0]:
            r_vals.append(theta)
            v_vals.append(v)
            theta_vals.append(np.mean(data["angles"]) + np.pi)

    _ = plt.figure(figsize=set_size('thesis'))

    sc = plt.scatter(np.array(r_vals), np.array(v_vals) / 1e3,
                     c=np.array(theta_vals) * 180 / np.pi, cmap='viridis', vmin=15, vmax=90)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"Impact Angle [$\theta_f$]", fontsize=13)

    query_points = [
        (20, 50e3),
        (25, 20e3),
        (45, 20e3),
        (15, 60e3),
        (35, 45e3),
        (65, 60e3),
        (80, 12e3)
    ]

    for r, v in query_points:
        query_point = np.array([r, initial_rads[0], v])
        theta_frag_estimate = theta_frag_interp(query_point) + np.pi

        plt.scatter(r, v / 1e3, c=np.array(theta_frag_estimate) * 180 / np.pi,
                    cmap='viridis', edgecolor='k', vmin=15, vmax=90)

    plt.xlabel(r"Initial angle [deg]", fontsize=13)
    plt.ylabel(r"Initial velocity [${\rm km\,s}^{-1}$]", fontsize=13)

    plt.minorticks_on()

    plt.title(fr'$R_0={initial_rads[0]}\,{{\rm m}}$')

    # plt.savefig('./figures/interpolation_validation_angle.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    interp_validation()
    interp_validation_angle()
