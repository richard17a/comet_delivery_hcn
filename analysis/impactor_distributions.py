# pylint: disable=E1101,R0914,R0915,E0606

"""
Docstring for figure2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
import distributions
import atmos_entry_interp

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def get_population(npoints):
    """
    Docstring for get_population
    
    :param npoints: Description
    """

    radii = distributions.generate_radii(npoints, seed=42)
    # NB mean = sqrt(28.2 **2 - 11.19 **2)
    vels = distributions.generate_velocities(npoints, mean=25.9, std_dev=5., v_esc=11.19)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def get_population_DS09(npoints):

    radii = distributions.generate_radii(npoints, alpha1=1.78, alpha2=2.34, alpha3=5.74, seed=42)
    # NB mean = sqrt(28.2 **2 - 11.19 **2)
    vels = distributions.generate_velocities(npoints, mean=25.9, std_dev=5., v_esc=11.19)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def get_population_21kms(npoints):

    radii = distributions.generate_radii(npoints, seed=42)
    # NB mean = sqrt(22 **2 - 11.19 **2)
    vels = distributions.generate_velocities(npoints, mean=18.94, std_dev=5., v_esc=11.19)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 11.19e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def get_population_Mars(npoints):

    radii = distributions.generate_radii(npoints, seed=42, r_min=0.1)
    vels = distributions.generate_velocities(npoints, mean=19.0, std_dev=5., v_esc=5.03)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 5.03e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def get_population_Venus(npoints):

    radii = distributions.generate_radii(npoints, seed=42)
    vels = distributions.generate_velocities(npoints, mean=35.2, std_dev=5., v_esc=10.36)
    angles = distributions.generate_angles(npoints) * 180 / np.pi

    valid = ((angles >= 15) & (angles <= 90) &
            (radii * 1e3 >= 100) & (radii * 1e3 <= 1e4) &
            ((vels * 1e3) >= 10.36e3) & ((vels * 1e3) <= 70e3))

    angles_valid = angles[valid]
    radii_valid = radii[valid]
    vels_valid = vels[valid]

    return vels_valid, radii_valid, angles_valid


def calc_surface_distributions(vels0, radii0, angles0,
                               fpath="/data/rja92/HCN-Delivery-Paper/data/simulation_results.pkl"):
    """
    Docstring for calc_surface_distributions
    
    :param vels0: Description
    :param radii0: Description
    :param angles0: Description
    :param fpath: Description
    """

    m_frag_interp, v_frag_interp, theta_frag_interp =\
                atmos_entry_interp.generate_interpolator(fpath)

    queries = np.vstack([angles0, radii0 * 1e3, vels0 * 1e3]).T

    masses = m_frag_interp(queries)
    vels = v_frag_interp(queries)
    angles = theta_frag_interp(queries) + np.pi

    radii = ( (3 * masses) / (4 * np.pi * 0.6e3)) ** (1./3.)

    angles = angles * 180 / np.pi
    vels = vels / 1e3

    radii = radii / 1e3

    return vels, radii, angles, masses
