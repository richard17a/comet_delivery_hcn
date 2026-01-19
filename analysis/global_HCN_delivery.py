import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.integrate as spi
from cmcrameri import cm
from utils import set_size

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def dn_dM(mass, c):
    """
    Docstring for dn_dM
    
    :param mass: Description
    :param c: Description
    """

    return mass ** (-c)


def get_data():
    """
    Calibration points for crater ages from le Feuvre & Wieczorek (2011). Contentious refers to
    teh calibration point that is not agred upon by all chronologies used.

    Returns:
    - tuple: Four arrays containing age, contentious time, crater counts, and contentious
            crater count

    """

    age = np.array([0.025, 0.053, 0.109, 0.80, 3.15, 3.22, 3.30,\
                    3.41, 3.58, 3.75, 3.80, 3.85, 3.85, 3.85])
    N_crater = np.array([0.169, 0.390, 0.824, 5.77, 29.7, 30.2, 27.7,\
                         32.4, 60.1, 93.6, 84.5, 301, 298, 306]) * 1e-4

    t_contentious = np.array([4.35])
    N_contentious = np.array([7.851e-1])

    return age, t_contentious, N_crater, N_contentious


def chronology():
    """
    Calculates figure 1: the impact chronologies that will be used in our overlapping craters
    model
    """

    t = np.linspace(0, 4570, 10_000) / 1e3

    a = 1.893e-26 * (1e3 ** 1.8)
    b = 14.44
    c = 7.960e-4 * (1e3 ** 1.8)

    alpha = c
    beta = a / alpha

    N1_leFeuvreWieczorek = alpha * (beta * (np.exp(b * t) - 1) +  t) * (1e3 ** -1.8)

    N1_leFeuvreWieczorek = np.array(N1_leFeuvreWieczorek)

    a = 1.23e-15 * (1e3 ** 1.8)
    b = 7.85
    c = 1.30e-3 * (1e3 ** 1.8)

    alpha = c
    beta = a / alpha

    N1_marchi = alpha * (beta * (np.exp(b * t) - 1) + t) * (1e3 ** -1.8)

    N1_marchi = np.array(N1_marchi)

    age, t_contentious, N_crater, N_contentious = get_data()

    _ = plt.figure(figsize=set_size('thesis'))

    plt.plot(t, N1_leFeuvreWieczorek, label=r'Le Feuvre & Wieczorek 2011',\
            color=cm.bamako(0.2))
    
    plt.plot(t, N1_marchi, label=r'Marchi et al., 2009',\
            color=cm.bamako(0.8))

    plt.plot(age, N_crater, 'x', color='tab:brown')
    plt.plot(t_contentious, N_contentious, 'x', color='tab:red')

    plt.axvline(3.920, color='tab:gray')
    plt.text(3.650, 2e1, "unconstrained for\nages >3.92 Gyr", fontsize=11,\
            rotation=90, ha='center', va='center')

    plt.legend(fontsize=11, loc='upper left', frameon=False)

    plt.yscale('log')

    plt.xlim(0, 4.5)
    plt.ylim(1e-5, )

    ax = plt.gca()

    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs="auto"))

    plt.minorticks_on()

    plt.xticks([0, 1, 2, 3, 4])

    plt.xlabel(r'Age [Gya]', fontsize=12)
    plt.ylabel(r'$N(1\,{\rm km}, t)\,$ [km$^{-2}$]', fontsize=13)

    plt.savefig('/data/rja92/HCN-Delivery-Paper/src/figures/craterchronology.pdf', format='pdf', bbox_inches='tight')

    plt.show()


def size_to_mass_conversion(t, a, b, c):
    """
    This converts from N(t, D) to N(t, >m) (the mass of impactor required to produce a crater of diameter D)
    """

    alpha = c
    beta = a / alpha

    # add in references for all of these....
    rho_imp = 2700
    rho_tar = 2550
    v_imp = 15e3
    theta = 45 * np.pi / 180
    D_c = 15e3
    g = 1.62

    conversion = 1.17 * ((1.161 * ((rho_imp / rho_tar) ** 0.333) * ((6 / np.pi / rho_imp) ** 0.26) * (v_imp ** 0.44) * (g ** -0.22) * (np.sin(theta) ** 0.3333)) ** 1.13) * (D_c ** -0.13)
    # D = conversion * (m ** 0.294)
    
    N1_M_prefactor = ((conversion) ** (- 1 / 0.294)) ** (1 / 3.36)

    R_m = 1737 # Radius of moon [km]
    A_m = 4 * np.pi * (R_m ** 2)

    # N(t, >m) = N1_M * (m ** -0.54)
    alpha_new = alpha * (N1_M_prefactor ** 1.8) * A_m

    N1_M = alpha_new * (beta * (np.exp(b * t) - 1) +  t)
    # print(alpha_new / 1e8)

    return t, N1_M, alpha_new, beta


def LeFeuvreMarchi_comparison():

    t = np.linspace(0, 4570, 10_000) / 1e3

    a = 1.893e-26 * (1e3 ** 1.8)
    b = 14.44
    c = 7.960e-4 * (1e3 ** 1.8)

    t_LeFeuvre, N1_M_LeFeuvre, _, _ = size_to_mass_conversion(t, a, b, c)

    a = 1.23e-15 * (1e3 ** 1.8)
    b = 7.85
    c = 1.30e-3 * (1e3 ** 1.8)

    t_Marchi, N1_M_Marchi, _, _ = size_to_mass_conversion(t, a, b, c)

    _ = plt.figure(figsize=set_size('thesis'))

    plt.plot(t_LeFeuvre, N1_M_LeFeuvre * (1.4e19 ** 0.46), label=r'Le Feuvre & Wieczorek 2011',\
            color=cm.bamako(0.2))
    plt.plot(t_Marchi, N1_M_Marchi * (1.4e19 ** 0.46), label=r'Marchi et al., 2009',\
            color=cm.bamako(0.8))

    plt.axvline(3.920, color='tab:gray')
    plt.text(4.1, 8e16, "unconstrained for\nages >3.92 Gyr", fontsize=11,\
            rotation=270, ha='center', va='center')
    
    plt.axhline(7.35e22, c='tab:red')
    plt.text(3.1, 3e22, "Lunar mass", fontsize=11, color='tab:red')

    plt.legend(fontsize=11, loc='upper left', frameon=False)

    plt.yscale('log')

    plt.xlim(0, 4.5)
    plt.ylim(1e15, 1e23)

    ax = plt.gca()

    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=999))
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs="auto"))

    plt.minorticks_on()

    plt.xticks([0, 1, 2, 3, 4])

    plt.xlabel(r'Age [Gya]', fontsize=12)
    plt.ylabel(r'$M(t)$ [kg]', fontsize=13)

    plt.savefig('/data/rja92/HCN-Delivery-Paper/src/figures/Mtot_LeFeuvreMarchi_comparison.pdf', format='pdf', bbox_inches='tight')

    plt.show()

    print(np.log10(N1_M_Marchi[-1] * (1.4e19 ** 0.46) * 24))


def cometary_delivery(unit_conv):

    t = np.linspace(0, 4570, 10_000) / 1e3

    a = 1.23e-15 * (1e3 ** 1.8)
    b = 7.85
    c = 1.30e-3 * (1e3 ** 1.8)

    _, _, alpha, beta = size_to_mass_conversion(t, a, b, c)

    t = np.linspace(4.57, 0, 4500)
    n_t = alpha * (t + beta * (np.exp(b * t) - 1))

    results = np.load('./data/avg_HCN_rad_1.npy')
    hcn_mean, _ = results.T

    radii = np.logspace(2, 4, 1000)
    f_HCN = lambda M: np.interp(((3 * M) / (4 * np.pi * 0.6e3)) ** (1./3.), radii, hcn_mean / 100)

    masses = 4 * np.pi * 0.6e3 * (radii ** 3) / 3

    integrand = lambda M: 20 * 0.01 * 0.001 * f_HCN(M) * M * dn_dM(M, 1.54)
    I_lo, error = spi.quad(integrand, masses[0], masses[-1], limit=500)
    print(round(error / I_lo * 100, 3), end='%\n')
    integrand = lambda M: 20 * 0.01 * 0.01 * f_HCN(M) * M * dn_dM(M, 1.54)
    I_hi, error = spi.quad(integrand, masses[0], masses[-1], limit=500)
    print(round(error / I_hi * 100, 3), end='%\n')

    dMdt_lo = np.gradient(I_lo * n_t, t) / 1e9
    dMdt_hi = np.gradient(I_hi * n_t, t) / 1e9
    dMdt_lo = dMdt_lo * unit_conv
    dMdt_hi = dMdt_hi * unit_conv

    radii = np.logspace(2, 3, 1000)
    f_HCN = lambda M: np.interp(((3 * M) / (4 * np.pi * 0.6e3)) ** (1./3.), radii, hcn_mean / 100)

    masses = 4 * np.pi * 0.6e3 * (radii ** 3) / 3

    integrand = lambda M: 20 * 0.01 * 0.001 * f_HCN(M) * M * dn_dM(M, 1.54)
    I_lo, error = spi.quad(integrand, masses[0], masses[-1], limit=500)
    print(round(error / I_lo * 100, 3), end='%\n')
    integrand = lambda M: 20 * 0.01 * 0.01 * f_HCN(M) * M * dn_dM(M, 1.54)
    I_hi, error = spi.quad(integrand, masses[0], masses[-1], limit=500)
    print(round(error / I_hi * 100, 3), end='%\n')

    dMdt_lo_small = np.gradient(I_lo * n_t, t) / 1e9
    dMdt_hi_small = np.gradient(I_hi * n_t, t) / 1e9
    unit_conv = 1e3 * 6.022e23 / 27.03 / 3.154e7 / 4 / np.pi / (6.371e8 ** 2)
    dMdt_lo_small = dMdt_lo_small * unit_conv
    dMdt_hi_small = dMdt_hi_small * unit_conv

    return t, (dMdt_lo, dMdt_hi), (dMdt_lo_small, dMdt_hi_small)


def IDPs(unit_conv):

    a = 1.23e-15 * (1e3 ** 1.8)
    b = 7.85
    c = 1.30e-3 * (1e3 ** 1.8)

    alpha = c
    beta = a / alpha

    t = np.linspace(4.5, 0, 4500)
    n_t = alpha * (t + beta * (np.exp(b * t) - 1))
    n_t = np.gradient(n_t)

    HCN_conc_low = 1e-6
    HCN_conc_high = 10e-6

    IDP_low = (2e10 / n_t[0]) * n_t * HCN_conc_low
    IDP_high = (2e10 / n_t[0]) * n_t * HCN_conc_high

    return t, IDP_low * unit_conv, IDP_high * unit_conv


def compute_integral(m, N):

    dN_dm = np.abs(np.gradient(N, m))
    integrand = m * dN_dm

    return spi.simps(integrand, m)


def calc_Mmet(plot=False):

    m1 = np.logspace(-2, 0, 5000)
    N1 = 10**(-0.492 * np.log10(m1) + 3.646)

    m2 = np.logspace(0, 1, 5000)
    N2 = 10**(-0.820 * np.log10(m2) + 3.648)

    m3 = np.logspace(1, np.log10(2.4e3), 500)
    N3 = 10**(-1.250 * np.log10(m3) + 4.098)

    m4 = np.logspace(np.log10(2.4e3), 5, 5000)
    N4 = 10**(-0.996 * np.log10(m4) + 2.983)

    ms = [m1, m2, m3, m4]
    Ns = [N1, N2, N3, N4]

    M_met = 0
    for counter in range(len(ms)):

        M_met += compute_integral(ms[counter], Ns[counter])

    if plot:

        _ = plt.figure(figsize=set_size('thesis'))

        plt.plot(m1, N1, c='tab:blue')
        plt.plot(m2, N2, c='tab:blue')
        plt.plot(m3, N3, c='tab:blue')
        plt.plot(m4, N4, c='tab:blue')

        plt.xscale('log')
        plt.yscale('log')

        plt.xlabel(r"Mass [kg]", fontsize=13)
        plt.ylabel(r"N(>m) [yr$^{-1}$]", fontsize=13)

        formatted = f'{M_met:.2e}'  # Format in scientific notation
        mantissa, exponent = formatted.split('e')
        exponent = int(exponent)

        plt.title(rf'$\dot{{M}}_{{\rm today}}={mantissa} \times 10^{{{exponent}}}\,{{\rm kg\,yr}}^{{-1}}$')

        plt.show()

    return M_met


def meteorites(unit_conv):

    a = 1.23e-15 * (1e3 ** 1.8)
    b = 7.85
    c = 1.30e-3 * (1e3 ** 1.8)

    alpha = c
    beta = a / alpha

    t = np.linspace(4.5, 0, 4500)
    n_t = alpha * (t + beta * (np.exp(b * t) - 1))
    n_t = np.gradient(n_t)

    HCN_conc_low = 1e-6
    HCN_conc_high = 10e-6

    CC_frac = 0.05

    M_met = calc_Mmet()

    dMdt_low = (M_met / n_t[-2]) * n_t  * HCN_conc_low * CC_frac
    dMdt_high = (M_met / n_t[-2]) * n_t  * HCN_conc_high * CC_frac

    return t, dMdt_low * unit_conv, dMdt_high * unit_conv


def steam_atmospheres(plot=False):

    hcnames = ['/data/rja92/HCN-Delivery-Paper/data/wogan_out/HCN_production_1e21.txt',
               '/data/rja92/HCN-Delivery-Paper/data/wogan_out/HCN_production_1e22.txt',
               '/data/rja92/HCN-Delivery-Paper/data/wogan_out/HCN_production_1e23.txt']
    tnames = ['/data/rja92/HCN-Delivery-Paper/data/wogan_out/time_post_impact_1e21.txt',
              '/data/rja92/HCN-Delivery-Paper/data/wogan_out/time_post_impact_1e22.txt',
              '/data/rja92/HCN-Delivery-Paper/data/wogan_out/time_post_impact_1e23.txt']
    Mimps = [1e21, 1e22, 1e23]
    
    dMdt_out = []
    for counter, tname in enumerate(tnames):
        HCN_dt = np.loadtxt(hcnames[counter], unpack=True)
        t = np.loadtxt(tname, unpack=True)

        dMdt_out = np.append(dMdt_out, max(HCN_dt[t > 100]))

    if plot:
        _ = plt.figure(figsize=set_size('thesis'))

        for counter, tname in enumerate(tnames):
            t = np.loadtxt(tname, unpack=True)
            HCN_dt = np.loadtxt(hcnames[counter], unpack=True)

            plt.plot(t, HCN_dt, label=rf'$M_{{\rm imp}} = 10^{{{int(np.log10(Mimps[counter]))}}}\,$kg', zorder=10)

            plt.axhline(dMdt_out[counter], c='tab:gray')

        plt.xscale('log')
        plt.yscale('log')

        plt.xlim(1e2, 1e6)
        plt.ylim(1e1, 1e9)

        plt.xlabel('Time after impact [Myr]', fontsize=13)
        plt.ylabel('HCN production\n(molecules cm$\\mathrm{^{-2}}$ s$\\mathrm{^{-1}}$)', fontsize=13)

        plt.minorticks_on()

        plt.legend(frameon=False, loc='upper right')

        plt.show()

    return dMdt_out


def main():

    unit_conv = 1e3 * 6.022e23 / 27.03 / 3.154e7 / 4 / np.pi / (6.371e8 ** 2)

    tcom, (dMdt_com_lo, dMdt_com_hi), (dMdt_com_lo_small, dMdt_com_hi_small) = cometary_delivery(unit_conv)
    tIDP, IDP_low, IDP_high = IDPs(unit_conv)
    tmet, met_low, met_high = meteorites(unit_conv)
    dMdt_steam = steam_atmospheres()

    _ = plt.figure(figsize=set_size('thesis'))

    plt.fill_between(tcom, dMdt_com_lo, dMdt_com_hi, color=cm.oslo_r(0.8), alpha=0.75, lw=0)
    plt.fill_between(tcom, dMdt_com_lo_small, dMdt_com_hi_small, color=cm.oslo_r(0.6), alpha=0.75, lw=0)

    plt.fill_between(tIDP, IDP_low, IDP_high, alpha=0.5, color=cm.oslo_r(0.4), lw=0)
    plt.fill_between(tmet, met_low, met_high, alpha=0.5, color=cm.oslo_r(0.2), lw=0)

    plt.axhspan(min(dMdt_steam), max(dMdt_steam), color='tab:red', alpha=0.25, lw=0)
    plt.axhspan(2e9, 1e10, color='tab:brown', alpha=0.25, lw=0)
    plt.axhspan(2e15 * 2.95e11 / 4 / np.pi / (6.37e8 ** 2), 7e15 * 2.95e11 / 4 / np.pi / (6.37e8 ** 2), alpha=0.25, color='tab:purple')

    plt.text(3.4, 2e2, r'Intact cometary delivery', rotation=16)
    plt.text(3.42, 1.1e1, r'Intact cometary delivery ($D<1\,$km)', rotation=16)
    plt.text(4.4,2.5e3, r'IDPs', rotation=16)
    plt.text(4.3, 2e0, r'Meteorites', rotation=16)
    plt.text(3.05, 2e5, "Post-impact steam atmophere\n     (Wogan et al., 2023)")
    plt.text(3.05, 3e9, r"Photochemistry (reducing atmosphere) (Tian et al., 2011)")
    plt.text(3.05, 1.6e8, "Lightning (reducing atmosphere) (Barth et al., 2023)")

    plt.yscale('log')

    plt.xlim(3, 4.5)
    plt.ylim(1, 1e10)

    plt.minorticks_on()

    plt.gca().yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
    plt.gca().yaxis.set_minor_locator(matplotlib.ticker.LogLocator(numticks=999, subs="auto"))

    plt.xlabel(r'Age [Gya]', fontsize=13)
    plt.ylabel('HCN Accumulation\n(molecules cm$\\mathrm{^{-2}}$ s$\\mathrm{^{-1}}$)', fontsize=13)

    # plt.savefig('/data/rja92/HCN-Delivery-Paper/src/figures/endogenous_exogenous_comparison.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    main()

    # LeFeuvreMarchi_comparison()
    # chronology()
    # calc_Mmet(plot=True)
    # steam_atmospheres(plot=True)

