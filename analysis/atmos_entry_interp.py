import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from scipy.interpolate import RegularGridInterpolator
import parametrised_chemistry, distributions

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


def generate_interpolator(fpath="./data/simulation_results.pkl"):
    
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
            v_frag_values[i, j, k] = np.sum(np.array(data["vels"]) *np.array(data["masses"])) / np.sum(data["masses"])
            theta_frag_values[i, j, k] = np.sum(np.array(data["angles"]) *np.array(data["masses"])) / np.sum(data["masses"])

    m_frag_interp = RegularGridInterpolator((theta_vals, r_vals, v_vals), m_frag_values)
    v_frag_interp = RegularGridInterpolator((theta_vals, r_vals, v_vals), v_frag_values)
    theta_frag_interp = RegularGridInterpolator((theta_vals, r_vals, v_vals), theta_frag_values)

    return m_frag_interp, v_frag_interp, theta_frag_interp


def interp_validation():
    
    m_frag_interp, v_frag_interp, theta_frag_interp = generate_interpolator()

    with open("./data/simulation_results.pkl", "rb") as f:
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

    plt.figure(figsize=(set_size('thesis')))

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
    FIX UP -- problem was that np.artan2 was giving results in negative quadrant - but interpolation for atmos needs positive
    """
    m_frag_interp, v_frag_interp, theta_frag_interp = generate_interpolator()

    with open("./data/simulation_results.pkl", "rb") as f:
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

    plt.figure(figsize=(set_size('thesis')))

    sc = plt.scatter(np.array(r_vals), np.array(v_vals) / 1e3, c=np.array(theta_vals) * 180 / np.pi, cmap='viridis', vmin=15, vmax=90)
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
        
        plt.scatter(r, v / 1e3, c=np.array(theta_frag_estimate) * 180 / np.pi, cmap='viridis', edgecolor='k', vmin=15, vmax=90)

    plt.xlabel(r"Initial angle [deg]", fontsize=13)
    plt.ylabel(r"Initial velocity [${\rm km\,s}^{-1}$]", fontsize=13)

    plt.minorticks_on()

    plt.title(fr'$R_0={initial_rads[0]}\,{{\rm m}}$')

    # plt.savefig('./figures/interpolation_validation_angle.pdf', bbox_inches='tight', format='pdf')

    plt.show()


def rho0_comparison():
    
    fpaths = ["./data/simulation_results_01.pkl",
              "./data/simulation_results.pkl",
              "./data/simulation_results_10.pkl"]

    fig_width, fig_height = set_size('thesis')

    fig, axes = plt.subplots(1, 3, figsize=(3 * fig_width, fig_height), sharey=True)

    for ax, fpath in zip(axes, fpaths):
       
        with open(fpath, "rb") as f:

            results = pickle.load(f)

        r_vals = []
        v_vals = []
        m_vals = []

        initial_angs = sorted(set(key[0] for key in results.keys()))

        idx = np.argmin(np.abs(np.array(initial_angs) - 45))

        for (theta, r, v), data in results.items():
            if theta == initial_angs[idx]:
                if len(data['masses']) == 0:
                    r_vals.append(r)
                    v_vals.append(v)
                    m_vals.append(0)
                else:
                    m0 = (4 * np.pi * 0.6e3 / 3) * (r ** 3)

                    r_vals.append(r)
                    v_vals.append(v)
                    m_vals.append(np.mean(data["masses"]) / m0)

        sc = ax.scatter(np.array(r_vals), np.array(v_vals) / 1e3, c=np.array(m_vals), cmap='viridis', vmin=0, vmax=1)

        # m_frag_interp, v_frag_interp, theta_frag_interp = generate_interpolator(fpath)

        # query_points = [
        #     (200, 50e3),
        #     (250, 20e3),
        #     (450, 20e3),
        #     (150, 60e3),
        #     (3500, 45e3),
        #     (6500, 60e3),
        #     (8000, 12e3)
        # ]

        # for r, v in query_points:
        #     m0 = (4 * np.pi * 0.6e3 / 3) * (r ** 3)

        #     query_point = np.array([initial_angs[idx], r, v])
        #     m_frag_estimate = m_frag_interp(query_point) / m0
            
        #     ax.scatter(r, v / 1e3, c=np.array(m_frag_estimate), cmap='viridis', edgecolor='k', vmin=0, vmax=1)

        ax.set_xscale('log')
        ax.minorticks_on()

    cbar = fig.colorbar(sc)
    cbar.set_label(r"Fragment mass [$M_0$]", fontsize=13)

    plt.tight_layout()

    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(3 * fig_width, fig_height), sharey=True)

    for ax, fpath in zip(axes, fpaths):
       
        with open(fpath, "rb") as f:

            results = pickle.load(f)

        r_vals = []
        v_vals = []
        m_vals = []

        initial_angs = sorted(set(key[0] for key in results.keys()))

        idx = np.argmin(np.abs(np.array(initial_angs) - 45))

        for (theta, r, v), data in results.items():
            if theta == initial_angs[idx]:
                if len(data['masses']) == 0:
                    r_vals.append(r)
                    v_vals.append(v)
                    m_vals.append(0)
                else:
                    rf = np.sum(np.array(data['masses']) * np.array(data['vels'])) / np.sum(data['masses'])

                    r_vals.append(r)
                    v_vals.append(v)
                    m_vals.append(rf)

        sc = ax.scatter(np.array(r_vals), np.array(v_vals) / 1e3, c=np.array(m_vals) / 1e3, cmap='viridis', vmin=11.19, vmax=70)

        # m_frag_interp, v_frag_interp, theta_frag_interp = generate_interpolator(fpath)

        # query_points = [
        #     (200, 50e3),
        #     (250, 20e3),
        #     (450, 20e3),
        #     (150, 60e3),
        #     (3500, 45e3),
        #     (6500, 60e3),
        #     (8000, 12e3)
        # ]

        # for r, v in query_points:
        #     m0 = (4 * np.pi * 0.6e3 / 3) * (r ** 3)

        #     query_point = np.array([initial_angs[idx], r, v])
        #     m_frag_estimate = m_frag_interp(query_point) / m0
            
        #     ax.scatter(r, v / 1e3, c=np.array(m_frag_estimate), cmap='viridis', edgecolor='k', vmin=0, vmax=1)

        ax.set_xscale('log')
        ax.minorticks_on()

    cbar = fig.colorbar(sc)
    cbar.set_label(r"log(Fragment radius) [m]", fontsize=13)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    
    # interp_validation()
    # interp_validation_angle()
    rho0_comparison()
    
    # want to plot at some point a (r0,v0) comparison for three atmos profiles...
