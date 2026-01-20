"""
Docstring
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from cmcrameri import cm
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def size_to_mass_conversion(t, a, b, c):
    """
    This converts from N(t, D) to N(t, >m) (the mass of impactor required 
    to produce a crater of diameter D)
    """

    alpha = c
    beta = a / alpha

    rho_imp = 2700
    rho_tar = 2550
    v_imp = 15e3
    theta = 45 * np.pi / 180
    D_c = 15e3
    g = 1.62

    conversion = 1.17 * ((1.161 * ((rho_imp / rho_tar) ** 0.333) *\
                          ((6 / np.pi / rho_imp) ** 0.26) * (v_imp ** 0.44) *\
                          (g ** -0.22) * (np.sin(theta) ** 0.3333)) ** 1.13) * (D_c ** -0.13)

    N1_M_prefactor = ((conversion) ** (- 1 / 0.294)) ** (1 / 3.36)

    R_m = 1737 # Radius of moon [km]
    A_m = 4 * np.pi * (R_m ** 2)

    # N(t, >m) = N1_M * (m ** -0.54)
    alpha_new = alpha * (N1_M_prefactor ** 1.8) * A_m

    N1_M = alpha_new * (beta * (np.exp(b * t) - 1) +  t)
    # print(alpha_new / 1e8)

    return t, N1_M, alpha_new, beta


def convert_sfd(F, D_new=0.1, D_old=10):
    """
    this needs to be careful that D_old really is in the range 1<D_old<50km
    """

    alpha_1 = 1.78
    alpha_2 = 2.34
    alpha_3 = 5.78
    alpha_4 = 3.5

    D_12 = 1
    D_23 = 50
    D_34 = 150

    F_new = np.zeros_like(F)
    for i, _ in enumerate(F):

        conv = ((50 ** (1 - alpha_2) - D_old ** (1 - alpha_2)) / (1 - alpha_2)) +\
                ((150 ** (1 - alpha_3) - 50 ** (1 - alpha_3)) / (1 - alpha_3)) *\
                (D_23 ** (alpha_3 - alpha_2)) + ((1000 ** (1 - alpha_4) - 150 ** (1 - alpha_4)) /\
                                                 (1 - alpha_4)) * (D_34 ** (alpha_4 - alpha_3))

        K2 = F[i] / conv
        K3 = K2 * (D_23 ** (alpha_3 - alpha_2))
        K4 = K2 * (D_34 ** (alpha_4 - alpha_3))
        K1 = K2 * (D_12 ** (alpha_1 - alpha_2))

        F_new[i] = (K1 / (1 - alpha_1)) * (1 ** (1 - alpha_1) - D_new ** (1 - alpha_1)) +\
                    (K2 / (1 - alpha_2)) * (50 ** (1 - alpha_2) - 1 ** (1 - alpha_2)) +\
                    (K3 / (1 - alpha_3)) * (150 ** (1 - alpha_3) - 50 ** (1 - alpha_3)) +\
                    (K4 / (1 - alpha_4)) * (1000 ** (1 - alpha_4) - 150 ** (1 - alpha_4))

    return F_new


def impact_threshold(F, threshold):
    """
    Docstring for impact_threshold
    
=    :param F: Description
    :param threshold: Description
    """

    residual = 1 - F / F[-1]

    mask = residual <= threshold
    if not np.any(mask):
        raise ValueError(f"Residual never reaches {threshold:.2f}")

    idx = np.argmax(mask)

    return idx


def main():
    """
    Docstring for main
    """

    Cs = 1./6.
    F1 = 6.5e3
    F2 = F1
    F3 = 4e-3
    alpha1 = 1
    alpha2 = 0.44
    tau1 = 7
    tau2 = 13

    t = np.linspace(4570, 0, 1000)
    F_10 = Cs * (F1 * np.exp(-(t/tau1) ** alpha1) + F2 * np.exp(-(t/tau2) ** alpha2) +\
                 F3 * (4570 - t))

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

    idx90 = impact_threshold(F, 0.9)
    idx99 = impact_threshold(F, 0.99)
    print(idx90, idx99)
    print(1 - F[idx90] / F[-1], 4.57 - t[idx90] / 1e3)
    print(1 - F[idx99] / F[-1], 4.57 - t[idx99] / 1e3)


def discretized_impact_rate():
    """
    Docstring for discretized_impact_rate
    """

    Cs = 1./6.
    F1 = 6.5e3
    F2 = F1
    F3 = 4e-3
    alpha1 = 1
    alpha2 = 0.44
    tau1 = 7
    tau2 = 13

    t = np.linspace(4570, 0, 4570)
    F_10 = Cs * (F1 * np.exp(-(t/tau1) ** alpha1) + F2 * np.exp(-(t/tau2) ** alpha2) +\
                 F3 * (4570 - t))

    F = convert_sfd(F_10)

    F2 = []
    for i, _ in enumerate(t):

        print(f"{round(100 * i / len(t), 2)}%\r", end='')

        # a very lazy way to not run into problems at end of for loop....
        try:
            F2 = np.append(F2, int(abs(F[i] - F[i+1])))
        except Exception:
            F2 = np.append(F2, int(abs(F[i])))

    _ = plt.figure()

    plt.plot(4570 - t[:-1], np.diff(F))
    plt.plot(4570 - t, F2)

    plt.xlim(3700, 4500)

    # plt.xscale('log')
    plt.yscale('log')

    plt.minorticks_on()

    plt.show()


def nesvorny_comparison():
    """
    Docstring for nesvorny_comparison
    """

    Cs = 1
    F1 = 6.5e3
    F2 = F1
    F3 = 4e-3
    alpha1 = 1
    alpha2 = 0.44
    tau1 = 7
    tau2 = 13

    t = np.linspace(4.57, 0, 4570)
    delta_t = (t[0] - t[-1]) / len(t)

    a = 1.23e-15 * (1e3 ** 1.8)
    b = 7.85
    c = 1.30e-3 * (1e3 ** 1.8)

    _, _, alpha, beta = size_to_mass_conversion(t, a, b, c)

    N_10km = np.zeros(len(t))
    for i, _ in enumerate(t):

        # print(round(100 * i / len(t), 2), end='%\r')

        m = np.pi * 3e3 * (1e4 ** 3) / 6

        N_now = alpha * (t[i] + beta * (np.exp(b * t[i]) - 1)) * (m ** -0.54)
        N_prev = alpha * ((t[i] - delta_t) +\
                          beta * (np.exp(b * (t[i] - delta_t)) - 1)) * (m ** -0.54)

        N_comets = 20 * (N_now - N_prev)

        N_10km[i] = N_comets

    _ = plt.figure(figsize=set_size('thesis'))

    plt.plot(4570 - t*1e3, N_10km)

    t = np.linspace(4570, 0, 1000)
    F_10 = Cs * (F1 * np.exp(-(t/tau1) ** alpha1) +\
                 F2 * np.exp(-(t/tau2) ** alpha2) + F3 * (4570 - t))
    plt.plot(t, F_10, ls='--')
    plt.plot(t, F_10 / 6, ls='-')

    plt.plot()

    plt.xlim(10, 4570)
    plt.ylim(1e-2, 1e4)

    plt.xscale('log')
    plt.yscale('log')

    plt.show()


if __name__ == "__main__":

    main()
    # nesvorny_comparison()
    # discretized_impact_rate()
