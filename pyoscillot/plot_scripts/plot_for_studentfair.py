import numpy as np
import matplotlib.pyplot as plt
from cfg import parse_ticket, parse_global_ini
from three_dim_star import ThreeDimStar, TwoDimProjector
import matplotlib.animation as animation
from pathlib import Path
from pyoscillot.plot_scripts.animation_pulsation import (
    COLOR_DICT, DATADIR, SPECTRADIR, get_data_and_lims, create_layout)

global_dict = parse_global_ini()

instruments = ["CARMENES_VIS"]

mode = "temperature"
ticket = "/home/dane/Documents/PhD/pyoscillot/data/fake_spectra/TALK_0_m-1/talk_ticket3.ini"
conf = parse_ticket(ticket)
sim_star = conf["name"]

pulsations, rv_dict, crx_dict, dlw_dict, rvo_dict, lims = get_data_and_lims(
    sim_star, mode)

fig, ax = plt.subplots(5, 2, figsize=(8, 12.5))

VMIN = 4700
VMAX = 4900
instrument = "CARMENES_VIS"
COLOR_DICT = {"CARMENES_VIS":"green"}
for idx in range(0, 40, 8):
    a = ax[int(idx/8), 0]
    a2 = ax[int(idx/8), 1]

    print(idx)

    pos = a.imshow(pulsations[idx], origin="lower", cmap="seismic", vmin=VMIN, vmax=VMAX)
    a.set_xticks([])
    a.set_yticks([])
    t = rv_dict[instrument]["bjd"][idx] - rv_dict[instrument]["bjd"][0]
    a.set_title(f't={int(round(t,0))}d')
    # a.text(15,900,f't={int(round(t,0))}d' )
    cbar = fig.colorbar(pos, ax=a, location="left")

    cbar.ax.set_ylabel("Temperature [K]")

    orders = list(rvo_dict[instrument]["orders"].keys())
    rvs = np.array([rvo_dict[instrument]["orders"][o][idx]
                    for o in orders])

    rvs = rvs  - np.nanmedian(rv_dict[instrument]["rv_original"])

    rves = np.array([rvo_dict[instrument]["errors"][o][idx]
                     for o in orders])
    log_wave_A = np.array([crx_dict[instrument]["logwave"][o][idx]
                           for o in orders])
    beta = crx_dict[instrument]["crx"][idx]
    alpha = crx_dict[instrument]["crx_off"][idx]
    alpha = alpha - np.nanmedian(rv_dict[instrument]["rv_original"])

    l_v = crx_dict[instrument]["l_v"][idx]
    log_wave = np.linspace(log_wave_A.min(), log_wave_A.max())
    a2.plot(log_wave, alpha + beta * (log_wave - np.log(l_v)),
               color=COLOR_DICT[instrument], label=f"Chromatic Index={round(beta,1)} m/s/Np")
    a2.errorbar(log_wave_A, rvs, yerr=rves, linestyle="None", marker="o",
                   color=COLOR_DICT[instrument], label=f"RV per Spectral Order")
    # a2.legend()
    handles, labels = a2.get_legend_handles_labels()
    order = [1, 0]
    a2.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    mean = np.nanmean(rvs)
    a2.set_ylim(mean-15, mean+15)
    a2.set_ylabel(r"Radial Velocity $[\frac{m}{s}]$")
    a2.set_xlabel(r"$\ln(\lambda)[\AA]$")
fig.suptitle("Simulation of Non-Radial Pulsation (l=1, m=-1)")
# fig.subplots_adjust(left=0.03, bottom=0, right=1, top=0.97, wspace=0.03, hspace=0.03)
fig.tight_layout(rect=[0, 0, 1, 0.99])

plt.savefig("pulsation_dane_smaller.png", dpi=200)
# plt.show()
