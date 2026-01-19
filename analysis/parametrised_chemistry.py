# pylint: disable=C0103,E0606

"""
Docstring for parametrised_chemistry

Author
------
Catriona McDonald (Institute of Astronomy, University of Cambridge)
"""

import numpy as np

def vimp_from_meanT(T):
    '''
    Calculates an expected impact velocity from the mean peak temperature achieved during the impact
    T - mean peak temperature (K)

    returns the impact velocity (km/s)
    '''

    b = 1/2.07
    vimp = (T/25.67)**b
    return vimp


def surv_from_vimp(v, chem):
    '''
    Calculates the total HCN survival expected for a given impact velocity
    v - impact velocity (km/s)
    chem - chemistry model ('equil' or 'kinetic')

    returns the scaled survival of HCN (%)
    '''

    # set the chemistry dependent parameters
    if chem == 'equil':
        A = 99.64
        q = 7.99
        v0 = 7.63
        vmin = 4.5
        vmax = 15

    if chem == 'kinetic':
        A = 100.22
        q = 7.53
        v0 = 11.10
        vmin = 5
        vmax = 20

    # calculate the survival rates
    if v < vmin:
        surv = 100

    if v > vmax:
        surv = 0

    denom = 1 + np.exp(q * (v**(1/2) - v0**(1/2)))
    surv = A/denom

    # safety check the values and return survival
    if surv > 100:
        return 100

    if surv < 0:
        return 0

    return surv


def surv_from_diam(d, v, chem):
    '''
    Calculates the total HCN survival scaled by the diameter of the impactor
    d - diameter (km)
    v - velocity (km/s)
    chem - chemistry model ('equil' or 'kinetic')
    
    returns the scaled survival of HCN (%)
    '''

    # find the survival for a 1km comet from the impact velocity
    surv_1km = surv_from_vimp(v, chem)

    if chem == 'equil':
        B = 0.97
        C = 0.085
        E = 5.61
    if chem == 'kinetic':
        B = 0.99
        C = 0.068
        E = 6.63

    # velocity fitting parameter
    vel_param = -(C * v)**E
    surv = surv_1km * ((B*d)**vel_param)

    if surv > 100:
        return 100

    if surv < 0:
        return 0

    return surv


def scaled_meanT_from_theta(a, v):
    '''
    Calculate the mean peak temperature achieved during an impact at a given angle.
    a - impact angle (degrees)
    v - impact velocity (km/s) 

    returns scaled mean peak temperature (K)
    '''

    # work out the mean T from the impact velocity
    meanT = meanT_from_vimp(v)

    theta_rad = np.deg2rad(a)
    meanT_theta = meanT * (np.sin(theta_rad)**(1.6))
    return meanT_theta


def meanT_from_vimp(v):
    """
    Docstring - using equation 8 from your paper
    """

    return 25.67 * (v ** 2.07)


def surv_param_steps(d, v, a, chem):
    '''
    Calculate the total survival expected for an arbitrary
    d - diameter (km)
    v - velocity (km/s)
    a - impact angle (degrees)

    returns the scaled survival of HCN (%)
    '''

    # scale the temperature to the angle
    scaled_meanT = scaled_meanT_from_theta(a, v)
    # work out the equivalent impact velocity from the angle
    equiv_vimp = vimp_from_meanT(scaled_meanT)
    # scale the survival to the impactor diameter
    surv_scaled = surv_from_diam(d, equiv_vimp, chem)

    return surv_scaled
