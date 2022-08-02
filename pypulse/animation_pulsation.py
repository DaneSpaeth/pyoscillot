import numpy as np
import matplotlib.pyplot as plt
from parse_ini import parse_ticket, parse_global_ini
from three_dim_star import ThreeDimStar, TwoDimProjector
import plapy.rv.dataloader as load
import matplotlib.animation as animation
from pathlib import Path

global_dict = parse_global_ini()
DATADIR = global_dict["datapath"]
DATADIR = Path("/home/dane/Documents/PhD/pypulse/mounted_data")
SPECTRADIR = DATADIR / "fake_spectra"


COLOR_DICT = {"Lick": "blue",
              "SONG": "orange",
              "CARMENES_VIS": "green",
              "CARMENES_NIR": "red",
              "EXPRESS": "black",
              "SIMULATION": "green",
              "HARPS_RVBANK": "purple",
              "HARPS": "purple",
              "HARPS-N": "magenta",
              "BOAO": "black"}


def get_arrays(projector):
    """ Get all arrays (e.g. pulsation, temp) of the simulation.

        Return all arrays as a dictionary.

        Useful for creating movies afterwards.
    """
    array_dict = {
        "pulsation_rad": projector.pulsation_rad(),
        "pulsation_phi": projector.pulsation_phi(),
        "pulsation_theta": projector.pulsation_theta(),
        "pulsation": projector.pulsation(),
        "temperature": projector.temperature(),
        "rotation": projector.rotation(),
        "intensity_stefan_boltzmann": projector.intensity_stefan_boltzmann()}

    return array_dict


def create_high_res_arrays(ticket):
    """ Create and save the high_res arrays for an existing simulation.

        At the moment only works with one mode.
    """
    conf = parse_ticket(ticket)
    name = conf["name"]

    spectra_dir = SPECTRADIR / name
    highres_array_dir = spectra_dir / "highres_arrays"
    if highres_array_dir.is_dir():
        return None
    else:
        highres_array_dir.mkdir()

    # Determine the bjd timestamps from the existing simulation
    flux_file = spectra_dir / "flux.txt"
    bjds = []
    with open(flux_file, "r") as f:
        for line in f:
            bjds.append(float(line.split()[0]))

    bjds = sorted(bjds)

    for bjd in bjds:

        # Determine the simulations
        N_star = 1000
        N_border = 3
        limb_darkening = bool(int(conf["limb_darkening"]))
        v_rot = conf["v_rot"]
        inclination = conf["inclination"]
        Teff = int(conf["teff"])
        star = ThreeDimStar(Teff=Teff, v_rot=v_rot)
        projector = TwoDimProjector(star,
                                    N=N_star,
                                    border=N_border,
                                    inclination=inclination,
                                    line_of_sight=True,
                                    limb_darkening=limb_darkening)
        simulation_keys = conf["simulations"]
        for sim in simulation_keys:
            P = conf[sim]["period"]
            l = int(conf[sim]["l"])
            k = int(conf[sim]["k"])
            v_p = conf[sim]["v_p"]
            dT = conf[sim]["dt"]
            T_phase = conf[sim]["t_phase"]

            if "m" in list(conf[sim].keys()):
                ms = [int(conf[sim]["m"])]
            else:
                ms = range(-l, l + 1)
            for m in ms:
                print(
                    f"Add Pulsation {sim}, with P={P}, l={l}, m={m}, v_p={v_p}, k={k}, dT={dT}, T_phase={T_phase} at bjd={bjd}")

                star.add_pulsation(t=bjd, l=l, m=m, nu=1 / P, v_p=v_p, k=k,
                                   T_var=dT, T_phase=T_phase)
        array_dict = get_arrays(projector)
        for directory, array in array_dict.items():
            out_dir = (highres_array_dir / directory)
            if not out_dir.is_dir():
                out_dir.mkdir()
            np.save(out_dir / f"{bjd}.npy", array)

    return None


def read_in_saved_arrays_pulsation(sim_name, mode="pulsation"):

    highres_array_dir = SPECTRADIR / sim_name / "highres_arrays"
    # Be clever, use the mode as the directoy
    array_dir = highres_array_dir / mode
    array_paths = array_dir.glob("*npy")
    pulse_maps = []
    bjds = []

    for array_path in sorted(array_paths):
        pulse_maps.append(np.load(array_path))
        bjd = float(array_path.name.split(".npy")[0])
        bjds.append(bjd)
    pulse_maps = [pulse_map for _,
                  pulse_map in sorted(zip(bjds, pulse_maps))]

    return pulse_maps


def create_layout(dlw=False):
    """ Create the grid layout."""
    fig = plt.figure(constrained_layout=True, figsize=(16, 9))
    if not dlw:
        gs = fig.add_gridspec(ncols=2, nrows=3)
    else:
        gs = fig.add_gridspec(ncols=2, nrows=4)
    ax_left = fig.add_subplot(gs[:, 0])
    ax_right_top = fig.add_subplot(gs[0, 1])
    ax_right_mid = fig.add_subplot(gs[1, 1])
    ax_right_bot = fig.add_subplot(gs[2, 1])
    ax = [ax_left, ax_right_top, ax_right_mid, ax_right_bot]

    if dlw:
        # Add dlw
        ax_right_dlw = fig.add_subplot(gs[3, 1])
        ax.append(ax_right_dlw)

    return fig, ax


def get_data_and_lims(sim_star, mode):
    """ Read in all data and the limits."""
    pulsations = read_in_saved_arrays_pulsation(sim_star, mode=mode)
    crx_dict = load.crx(sim_star, full="True")
    dlw_dict = load.dlw(sim_star)
    rvo_dict = load.rvo(sim_star)

    # Read in data
    rv_dict = load.rv(sim_star)
    for key in rv_dict.keys():
        rv_dict[key]["bjd"] = rv_dict[key]["bjd"] - 2400000
        try:
            crx_dict[key]["bjd"] = crx_dict[key]["bjd"] - 2400000
        except KeyError:
            continue
        dlw_dict[key]["bjd"] = dlw_dict[key]["bjd"] - 2400000
        rv_dict[key]["rv_original"] = rv_dict[key]["rv"]
        rv_dict[key]["rv"] = rv_dict[key]["rv"] - np.median(rv_dict[key]["rv"])

    all_crx = np.array([])
    for instrument in crx_dict.keys():
        all_crx = np.concatenate((all_crx, crx_dict[instrument]["crx"]))
    all_dlw = np.array([])
    for instrument in dlw_dict.keys():
        all_dlw = np.concatenate((all_dlw, dlw_dict[instrument]["dlw"]))
    all_rv = np.array([])
    for instrument in rv_dict.keys():
        all_rv = np.concatenate((all_rv, rv_dict[instrument]["rv"]))

    lims = {"MIN_RV": np.min(all_rv),
            "MAX_RV": np.max(all_rv),
            "MIN_CRX": np.min(all_crx),
            "MAX_CRX": np.max(all_crx),
            "MIN_DLW": np.min(all_dlw),
            "MAX_DLW": np.max(all_dlw)}

    return pulsations, rv_dict, crx_dict, dlw_dict, rvo_dict, lims


def animate_pulse(ticket, instrument=None, mode="pulsation"):
    """ Animate the pulsation.

        At the moment only allow one instruments at a time.
    """
    instruments = ["CARMENES_VIS", "HARPS"]

    conf = parse_ticket(ticket)
    sim_star = conf["name"]

    pulsations, rv_dict, crx_dict, dlw_dict, rvo_dict, lims = get_data_and_lims(
        sim_star, mode)

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
    ax = init_plots(ax, index + 1, rv_dict, crx_dict, instruments)

    def create_plot(im, fig, ax, images, index, rv_dict, crx_dict, instruments, lims, dlw=False):
        """ Convienence function to create the same plots once with crx and once with dlw.
        """
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
                              crx_dict, instruments, lims, dlw=dlw)
            return im,

        ani = animation.FuncAnimation(
            fig, updatefig, images, interval=125, blit=False, repeat=False)
        outfolder = Path("/home/dane/Documents/PhD/pypulse/animations")
        if not dlw:
            ani.save(outfolder / f"{sim_star}_{mode}_crx.gif")
        else:
            ani.save(outfolder / f"{sim_star}_{mode}_dlw.gif")

        plt.close()
        fig, ax = create_layout()
        index = len(images) - 1
        im = ax[0].imshow(images[index], animated=True,
                          origin="lower", cmap="seismic", vmin=VMIN, vmax=VMAX)
        ax = init_plots(ax, index, rv_dict, crx_dict, instruments, dlw=dlw)
        ax = update_plots(ax, index + 1, rv_dict,
                          crx_dict, instruments, lims, dlw=dlw)
        if not dlw:
            plt.savefig(outfolder / f"{sim_star}_{mode}_crx.pdf")
        else:
            plt.savefig(outfolder / f"{sim_star}_{mode}_dlw.pdf")

    create_plot(im, fig, ax, images, index, rv_dict,
                crx_dict, instruments, lims, dlw=False)
    create_plot(im, fig, ax, images, index, rv_dict,
                dlw_dict, instruments, lims, dlw=True)


def init_plots(ax, index, rv_dict, crx_dict, instruments, dlw=False):
    """ Init plots."""
    for instrument in instruments:
        ax[1].errorbar(rv_dict[instrument]["bjd"][:index],
                       rv_dict[instrument]["rv"][:index],
                       yerr=rv_dict[instrument]["rve"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])

        if dlw:
            key = "dlw"
        else:
            key = "crx"
        ax[2].errorbar(crx_dict[instrument]["bjd"][:index],
                       crx_dict[instrument][key][:index],
                       yerr=crx_dict[instrument][key + "e"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])

        ax[3].errorbar(rv_dict[instrument]["rv"][:index],
                       crx_dict[instrument][key][:index],
                       yerr=crx_dict[instrument][key + "e"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])
        # if dlw is not None:
    #     ax[4].errorbar(time[:index], dlw[:index],
    #                    yerr=dlwe[:index], linestyle="None", marker="o")
    ax[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                 mode="expand", borderaxespad=0, ncol=len(ax[1].lines))
    return ax


def update_plots(ax, index, rv_dict, crx_dict, instruments, lims, dlw=False):
    """ Init plots."""
    ax[1].clear()
    ax[2].clear()
    if len(ax) == 4:
        ax[3].clear()

    for instrument in instruments:
        # update the first plot
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

        if not dlw:
            key = "crx"
        else:
            key = "dlw"
        ax[2].errorbar(crx_dict[instrument]["bjd"][:index],
                       crx_dict[instrument][key][:index],
                       yerr=crx_dict[instrument][key + "e"][:index],
                       linestyle="None", marker="o",
                       label=instrument,
                       color=COLOR_DICT[instrument])
        ax[2].set_xlim(min_time - 1, max_time + 1)
        if not dlw:
            ax[2].set_ylim(lims["MIN_CRX"] - 5, lims["MAX_CRX"] + 5,)
        else:
            ax[2].set_ylim(lims["MIN_DLW"] - 5, lims["MAX_DLW"] + 5,)
        ax[2].set_xlabel("Time [JD] - 2400000")
        if not dlw:
            ax[2].set_ylabel("CRX [m/s/Np]")
        else:
            ax[2].set_ylabel("dLW [m^2/s^2]")
        ax[2].ticklabel_format(useOffset=False, style='plain')

        if len(ax) == 4:
            ax[3].errorbar(rv_dict[instrument]["rv"][:index],
                           crx_dict[instrument][key][:index],
                           yerr=crx_dict[instrument][key + "e"][:index],
                           linestyle="None", marker="o",
                           label=instrument,
                           color=COLOR_DICT[instrument])
            ax[3].set_xlim(lims["MIN_RV"] - 5, lims["MAX_RV"] + 5)
            if not dlw:
                ax[3].set_ylim(lims["MIN_CRX"] - 5, lims["MAX_CRX"] + 5,)
            else:
                ax[3].set_ylim(lims["MIN_DLW"] - 5, lims["MAX_DLW"] + 5,)
            ax[3].set_xlabel("RV [m/s]")
            if not dlw:
                ax[3].set_ylabel("CRX [m/s/Np]")
            else:
                ax[3].set_ylabel("dLW [m^2/s^2]")

    ax[1].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                 mode="expand", borderaxespad=0, ncol=len(ax[1].lines))

    return ax


if __name__ == "__main__":

    ticket = "/home/dane/Documents/PhD/pypulse/data/fake_spectra/TALK_0/talk_ticket.ini"
    animate_pulse(ticket, mode="pulsation")
