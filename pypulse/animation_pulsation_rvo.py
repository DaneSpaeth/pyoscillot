import numpy as np
import matplotlib.pyplot as plt
from parse_ini import parse_ticket, parse_global_ini
from three_dim_star import ThreeDimStar, TwoDimProjector
import plapy.rv.dataloader as load
import matplotlib.animation as animation
from pathlib import Path
from animation_pulsation import (
    COLOR_DICT, DATADIR, SPECTRADIR, get_data_and_lims, create_layout)

global_dict = parse_global_ini()


def animate_rv_lambda(ticket, mode="pulsation"):
    instruments = ["CARMENES_VIS", "HARPS"]

    conf = parse_ticket(ticket)
    sim_star = conf["name"]

    pulsations, rv_dict, crx_dict, dlw_dict, rvo_dict, lims = get_data_and_lims(
        sim_star, mode)

    print(crx_dict["HARPS"].keys())
    # Initialize the plot part
    images = pulsations
    # fig, ax = plt.subplots(4, 1, figsize=(16, 9))
    fig, ax = create_layout()
    index = 0

    if mode == "pulsation":
        VMIN = -100
        VMAX = 100
    elif mode == "pulsation_rad":
        VMIN = -1
        VMAX = 1
    elif mode == "temperature":
        VMIN = 4600
        VMAX = 5000

    im = ax[0].imshow(images[index], animated=True,
                      origin="lower", cmap="seismic", vmin=VMIN, vmax=VMAX)
    ax = init_plots(ax, index + 1, rv_dict, crx_dict, rvo_dict, instruments)

    def updatefig(*args):
        nonlocal index
        nonlocal ax
        index += 1
        if index == len(images):
            index = 0
            return im,

        # update the image
        print(index)
        im.set_array(images[index])
        ax = update_plots(ax, index + 1, rv_dict,
                          crx_dict, rvo_dict, instruments, lims)
        return im,

    ani = animation.FuncAnimation(
        fig, updatefig, images, interval=125, blit=False, repeat=False)
    outfolder = Path("/home/dane/Documents/PhD/pypulse/animations")
    ani.save(outfolder / f"{sim_star}_{mode}_crx_rvo.gif")


def init_plots(ax, index, rv_dict, crx_dict, rvo_dict, instruments):
    """ Init plots."""
    for instrument in instruments:
        ax[1].errorbar(rv_dict[instrument]["bjd"][:index],
                       rv_dict[instrument]["rv"][:index],
                       yerr=rv_dict[instrument]["rve"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])

        ax[2].errorbar(crx_dict[instrument]["bjd"][:index],
                       crx_dict[instrument]["crx"][:index],
                       yerr=crx_dict[instrument]["crxe"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])

        ax[3].errorbar(rv_dict[instrument]["rv"][:index],
                       crx_dict[instrument]["crx"][:index],
                       yerr=crx_dict[instrument]["crxe"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])
        # if dlw is not None:
    #     ax[4].errorbar(time[:index], dlw[:index],
    #                    yerr=dlwe[:index], linestyle="None", marker="o")
    ax[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                 mode="expand", borderaxespad=0, ncol=len(ax[1].lines))
    return ax


def update_plots(ax, index, rv_dict, crx_dict, rvo_dict, instruments, lims):
    """ Init plots."""
    ax[1].clear()
    ax[2].clear()
    ax[3].clear()

    for instrument in instruments:
        # update the first plot
        last_bjd = rv_dict[instrument]["bjd"][:index][-1]
        ax[1].errorbar(rv_dict[instrument]["bjd"][:index],
                       rv_dict[instrument]["rv"][:index],
                       yerr=rv_dict[instrument]["rve"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])
        min_time = rv_dict["CARMENES_VIS"]["bjd"].min()
        max_time = rv_dict["CARMENES_VIS"]["bjd"].max()
        ax[1].set_xlim(min_time - 1, max_time + 1)
        ax[1].set_ylim(lims["MIN_RV"] - 5, lims["MAX_RV"] + 5,)
        ax[1].set_xlabel("Time [JD] - 2400000")
        ax[1].set_ylabel("RV [m/s]")
        ax[1].ticklabel_format(useOffset=False, style='plain')

        orders = list(rvo_dict[instrument]["orders"].keys())
        # Plot the RV over lambda
        rvs = np.array([rvo_dict[instrument]["orders"][o][index - 1]
                        for o in orders])

        rvs = rvs - np.nanmedian(rv_dict[instrument]["rv_original"])

        rves = np.array([rvo_dict[instrument]["errors"][o][index - 1]
                         for o in orders])
        log_wave_A = np.array([crx_dict[instrument]["logwave"][o][index - 1]
                               for o in orders])
        beta = crx_dict[instrument]["crx"][index - 1]
        alpha = crx_dict[instrument]["crx_off"][index - 1]
        alpha = alpha - np.nanmedian(rv_dict[instrument]["rv_original"])

        l_v = crx_dict[instrument]["l_v"][index - 1]
        ax[2].errorbar(log_wave_A, rvs, yerr=rves, linestyle="None", marker="o",
                       color=COLOR_DICT[instrument])
        log_wave = np.linspace(log_wave_A.min(), log_wave_A.max())
        ax[2].plot(log_wave, alpha + beta * (log_wave - np.log(l_v)),
                   color=COLOR_DICT[instrument])
        ax[2].set_xlabel("Ln Wavelength [A]")
        ax[2].set_ylabel("RV [m/s]")
        ax[2].set_ylim(lims["MIN_RV"] - 20, lims["MAX_RV"] + 20,)

        ax[3].errorbar(crx_dict[instrument]["bjd"][:index],
                       crx_dict[instrument]["crx"][:index],
                       yerr=crx_dict[instrument]["crxe"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])
        ax[3].set_xlim(min_time - 1, max_time + 1)
        ax[3].set_ylim(lims["MIN_CRX"] - 5, lims["MAX_CRX"] + 5,)

        ax[3].set_xlabel("Time [JD] - 2400000")
        ax[3].set_ylabel("CRX [m/s/Np]")
        ax[3].ticklabel_format(useOffset=False, style='plain')

    ax[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                 mode="expand", borderaxespad=0, ncol=len(ax[1].lines))

    return ax


if __name__ == "__main__":
    ticket = "/home/dane/Documents/PhD/pypulse/data/fake_spectra/TALK_0_m-1/talk_ticket3.ini"
    animate_rv_lambda(ticket)
