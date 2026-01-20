"""
Docstring

This script was used to produce the simulation_results_*.pkl files used throughout
the main analysis. The atmosentry package must be pip-installed from pypi.org
(pip install atmosentry).
"""

import pickle
import itertools
import multiprocessing
import numpy as np
from atmosentry.meteoroid import Meteoroid
from atmosentry import Simulation


def compute_impact(rad, theta, v0):
    """
    Docstring for compute_impact
    
    :param rad: Description
    :param theta: Description
    :param v0: Description
    """

    theta0 = theta * np.pi / 180

    rho_com = 0.6e3
    M0 = rho_com * (4 * np.pi / 3) * (rad ** 3)

    impactor = Meteoroid(
        x=0,
        y=0,
        z=100e3,
        vx=-v0 * np.cos(theta0),
        vy=0,
        vz=-v0 * np.sin(theta0),
        radius=rad,
        mass=M0,
        sigma=1e4,
        rho=rho_com,
        eta=2.5e6
    )

    sim = Simulation()

    sim.impactor = impactor
    sim.rho0 = 1.225
    sim.H = 7.2e3
    sim.dt = 1e-3

    sim.integrate()

    if len(sim.fragments):
        fragments_surface = [fragment for fragment in sim.fragments if fragment.state.z[-1] < 1]
        v_frags = np.array([
            np.sqrt(fragment.state.vx[-1] ** 2 +
                    fragment.state.vy[-1] ** 2 +
                    fragment.state.vz[-1] ** 2)
            for fragment in fragments_surface
        ])
        m_frags = np.array([fragment.state.mass[-1] for fragment in fragments_surface])
        theta_frags = np.array([
            np.arctan2(fragment.state.vz[-1], fragment.state.vx[-1])
            for fragment in fragments_surface
        ])

        return m_frags, v_frags, theta_frags

    # else:
    vel = np.sqrt(sim.impactor.state.vx ** 2 +
                    sim.impactor.state.vy ** 2 +
                    sim.impactor.state.vz ** 2)[-1]
    m = sim.impactor.state.mass[-1]
    theta = np.arctan2(sim.impactor.state.vz[-1], sim.impactor.state.vx[-1])

    return np.array([m]), np.array([vel]), np.array([theta])


def to_dict(masses, vels, angles):
    """
    Docstring for to_dict
    
    :param masses: Description
    :param vels: Description
    :param angles: Description
    """

    return {
        "vels": vels.tolist(),
        "masses": masses.tolist(),
        "angles": angles.tolist()
    }


def worker(params):
    """
    multiprocessing worker function
    """

    theta, r, v = params
    m_frags, v_frags, theta_frags = compute_impact(r, theta, v)

    return (theta, r, v), to_dict(m_frags, v_frags, theta_frags)


def main():
    """
    Docstring 

    The initialised velocity array, atmospheric surface density and 
    atmospheric scale height must all be changed when calculating 
    the effects of atmospheric entry for Earth, Mars (current and 1bar
    atmosphere), and Venus.
    """

    radii = np.logspace(2, 4, 50)
    # vels = np.linspace(11.19, 70, 50) * 1e3
    vels = np.linspace(5.03, 70, 50) * 1e3
    theta_vals = np.linspace(15, 90, 50)

    param_list = list(itertools.product(theta_vals, radii, vels))

    num_workers = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_list = pool.map(worker, param_list, chunksize=10)

    results = dict(results_list)

    with open("./data/simulation_results_Mars_1bar.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":

    main()

    print("\n====  comet_atmosentry_interpolation.py  ====\n")
