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


def calc_Kw(T):
    """
    Docstring for calc_Kw
    
    :param T: Description
    """

    Kw25 = 10 ** (-13.99)

    m_frac = np.exp(-(94.8e3 / 8.31) * (1 / T - 1 / 298.15))

    return Kw25 * m_frac


def horizontal_gradient_span(ax, ymin, ymax, color,
                             alpha_left=0.0, alpha_right=0.2,
                             xmin_frac=0.0, xmax_frac=0.75,
                             zorder=0, flip=False, nx=512,
                             clip_on=True, rasterized=True):
    """
    Draw a horizontal alpha gradient spanning vertically from ymin..ymax (data coords),
    and horizontally from xmin_frac..xmax_frac in axes fraction (0..1).
    alpha varies from alpha_left -> alpha_right (left->right). Set flip=True to invert.
    This function saves & restores axis limits to avoid autoscaling changes.
    """

    # sanity checks
    if not (0 <= xmin_frac <= 1 and 0 <= xmax_frac <= 1):
        raise ValueError("xmin_frac/xmax_frac must be in [0,1] (axes fraction).")

    # Save original limits so we can restore them after plotting the image
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # correct transform: x in axes fraction, y in data coords
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)

    rgb = matplotlib.colors.to_rgb(color)

    # Create an image with shape (1, nx, 4) so alpha varies horizontally (columns)
    gradient = np.ones((1, nx, 4), dtype=float)
    gradient[0, :, :3] = rgb

    if not flip:
        alpha_vals = np.linspace(alpha_left, alpha_right, nx)
    else:
        alpha_vals = np.linspace(alpha_right, alpha_left, nx)

    gradient[0, :, 3] = alpha_vals

    # Draw image: extent = (x0_axes_frac, x1_axes_frac, y0_data, y1_data)
    im = ax.imshow(gradient,
                    extent=(xmin_frac, xmax_frac, ymin, ymax),
                    origin='lower',
                    aspect='auto',
                    interpolation='nearest',   # preserves alpha at each column
                    transform=trans,
                    zorder=zorder,
                    clip_on=clip_on,
                    rasterized=rasterized)

    # Restore original axis limits to prevent autoscale from shifting text/axes
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return im


def main():
    """
    Docstring for main
    """

    T = 0.5  # Gyr
    N_imp = 1
    p_abio = np.logspace(-6, 0, 1000)  # per Myr (see Scherf and Cronin)
    tau_hcn = np.logspace(-6, 1, 1000)  # Myr
    tau_ool = T / (N_imp * (p_abio[None, :]) * (tau_hcn[:, None]))

    vmax = 13.787  # Gyr

    temp0 = 273.15
    temp25 = 298.15
    pH = 7

    k1_hcn_25 = (10**(-4950/temp25 + 8.43)) * (10**(-pH)) +\
                calc_Kw(temp25) * (10**(-4240/temp25 + 11.1)) / (10**(-pH))
    k1_hcn_0 = (10**(-4950/temp0 + 8.43)) * (10**(-pH)) +\
                calc_Kw(temp0) * (10**(-4240/temp0 + 11.1)) / (10**(-pH))

    fig, ax = plt.subplots(figsize=set_size('thesis'))

    p_edges = np.geomspace(p_abio[0], p_abio[-1], len(p_abio) + 1)
    t_edges = np.geomspace(tau_hcn[0], tau_hcn[-1], len(tau_hcn) + 1)
    pcm = ax.pcolormesh(p_edges * 1e3, t_edges * 1e3, tau_ool,
                        norm=matplotlib.colors.LogNorm(vmin = 0.5**2 / vmax, vmax=vmax),
                        shading='auto', cmap=cm.vik, rasterized=True)


    X, Y = np.meshgrid(p_abio * 1e3, tau_hcn * 1e3)

    masked_upper = np.ma.masked_where(tau_ool < vmax, tau_ool)

    _ = ax.contourf(X, Y, masked_upper,
                        levels=[vmax, tau_ool.max()],
                        colors=['lightgray'],
                        zorder=1,
                        extend='neither'
                        )


    ax.axhline(np.log(2) / k1_hcn_0 / 1e3 / 365 / 24 / 60 / 60, c='k', ls='--',
                xmax=0.875)
    ax.axhline(np.log(2) / k1_hcn_25 / 1e3 / 365 / 24 / 60 / 60, c='k', ls='--',
                xmax=1.00)

    ax.text(2e-3, 4e-1, r"$\tau_{\rm hydrolysis}$ ($25^\circ{\rm C},~{\rm pH}=7$)", fontsize=12)
    ax.text(2e-3, 3e+2, r"$\tau_{\rm hydrolysis}$ ($0^\circ{\rm C},~{\rm pH}=7$)", fontsize=12)

    horizontal_gradient_span(ax, ymin=2e-3, ymax=2e-2, color='tab:red',
                            alpha_left=0.5, alpha_right=0.0,
                            xmin_frac=0.0, xmax_frac=0.65, zorder=5, flip=False)
    ax.text(2e-3, 3.5e-3, r'$1000\times\tau_{\rm dilution}$', fontsize=12, zorder=10)

    horizontal_gradient_span(ax, ymin=4.4e-3, ymax=17.5e-3, color='tab:orange',
                            alpha_left=0.0, alpha_right=0.5,
                            xmin_frac=0.35, xmax_frac=1, zorder=5, flip=False)
    ax.text(1e2, 7.0e-3, r'$\tau_{\rm residence}$', fontsize=12, zorder=10)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'Abiogenesis probability [Gyr$^{-1}$]', fontsize=13)
    ax.set_ylabel(r'Cometary HCN lifetime [kyr]', fontsize=13)

    cbar = fig.colorbar(pcm, ax=ax, extend='max', pad=0.02, cmap=cm.vik)
    cbar.set_label(r'Origins of life timescale [Gyr]', fontsize=12)

    levels = [0.5, 4.57]  # Gyr
    cs = ax.contour(X, Y, tau_ool, levels=levels, linewidths=0.8, colors='k')
    ax.clabel(cs, fmt='%g Gyr', fontsize=8, inline=True,
            manual=[(5e2, 2e3), (1e2, 5e2)])

    plt.gca().tick_params(axis='both', which='both', top=True, bottom=True,
                        left=True, right=True, direction='in')

    plt.savefig('./figures/ool_timescale_contours.pdf', bbox_inches='tight', format='pdf')

    plt.show()


if __name__ == "__main__":

    main()
