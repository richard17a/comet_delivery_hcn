"""Generate angle, velocity and size distributions for comet imapcts"""

import numpy as np
import numba
from numba import njit, prange


@njit(inline='always')
def _inverse_cdf(u, r_low, r_high, alpha):

    if alpha == 1.0:
        return r_low * np.exp(u * (np.log(r_high) - np.log(r_low)))
    
    c_low = r_low ** (1 - alpha)
    c_high = r_high ** (1 - alpha)
    return (u * (c_high - c_low) + c_low) ** (1.0 / (1 - alpha))

@njit(parallel=True)
def generate_radii(n_samples,
                   r_min=0.1,
                   r_break1=1.0,
                   r_break2=50.0,
                   r_max=100.0,
                   alpha1=1.78,
                   alpha2=2.34,
                   alpha3=5.78,
                   seed=0):
    """
    Generate samples from a broken power-law size-frequency distribution:
      N(R) ~ R^{-alpha1} for r_min < R < r_break1,
            ~ R^{-alpha2} for r_break1 < R < r_break2,
            ~ R^{-alpha3} for r_break2 < R < r_max.

    Parameters:
    - n_samples (int): number of radii to generate
    - r_min (float): minimum radius
    - r_break1 (float): first break radius
    - r_break2 (float): second break radius
    - r_max (float): maximum radius
    - alpha1 (float): power-law exponent below r_break1
    - alpha2 (float): power-law exponent between r_break1 and r_break2
    - alpha3 (float): power-law exponent above r_break2
    - seed (int): random seed

    Returns:
    - numpy.ndarray: array of generated radii
    """
    np.random.seed(seed)

    if alpha1 == 1.0:
        area1 = np.log(r_break1) - np.log(r_min)
    else:
        area1 = (r_break1**(1 - alpha1) - r_min**(1 - alpha1)) / (1 - alpha1)

    if alpha2 == 1.0:
        area2 = np.log(r_break2) - np.log(r_break1)
    else:
        area2 = (r_break2**(1 - alpha2) - r_break1**(1 - alpha2)) / (1 - alpha2)

    if alpha3 == 1.0:
        area3 = np.log(r_max) - np.log(r_break2)
    else:
        area3 = (r_max**(1 - alpha3) - r_break2**(1 - alpha3)) / (1 - alpha3)

    total_area = area1 + area2 + area3

    p1 = area1 / total_area
    p2 = area2 / total_area

    samples = np.empty(n_samples, dtype=np.float64)

    for i in prange(n_samples):
        u0 = np.random.random()

        if u0 < p1:

            u = np.random.random()
            samples[i] = _inverse_cdf(u, r_min, r_break1, alpha1)
        elif u0 < p1 + p2:

            u = np.random.random()
            samples[i] = _inverse_cdf(u, r_break1, r_break2, alpha2)
        else:

            u = np.random.random()
            samples[i] = _inverse_cdf(u, r_break2, r_max, alpha3)

    return samples


def generate_velocities(npoints, mean=28.2, std_dev=5., v_esc=11.19):
    """
    Generate an array of impact velocities using a lognormal distribution.

    Parameters:
    - npoints (int): number of impactors
    - mean (float): mean impact velocity of short period comet (default: 19 km/s)
    - std_dev (float): standard deviation of lognormal distribution (default: 5 km/s)
    - v_esc (float): escape velocity of the target body (default: 11.19 km/s for Earth)

    Returns:
    - numpy.ndarray: array of impact velocities
    """

    mu = np.log(mean ** 2 / np.sqrt(std_dev ** 2 + mean ** 2))
    sigma = np.sqrt(np.log((std_dev / mean) ** 2 + 1))

    vels = np.random.lognormal(mu, sigma, npoints)

    return vels


def generate_angles(npoints):
    """
    Generate an array of impact angles assuming a a differential probability distribution
    dP = sin(2\theta)d\theta

    Parameters:
    - npoints (int): number of impactors

    Returns:
    - numpy.ndarray: array of impact angles (radians)
    """

    u = np.random.uniform(0, 1, npoints)

    theta = 0.5 * np.arccos(1 - 2 * u)

    return theta
