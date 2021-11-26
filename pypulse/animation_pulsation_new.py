import numpy as np
import matplotlib.pyplot as plt
from parse_ini import parse_ticket, parse_global_ini
from three_dim_star import ThreeDimStar, TwoDimProjector
import plapy.rv.dataloader as load
import matplotlib.animation as animation
from pathlib import Path

global_dict = parse_global_ini()
DATADIR = global_dict["datapath"]
SPECTRADIR = DATADIR / "fake_spectra"


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


def read_in_saved_arrays_pulsation(sim_name):

    highres_array_dir = SPECTRADIR / sim_name / "highres_arrays"
    array_dir = highres_array_dir / "pulsation"
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


def animate_pulse(ticket, instrument="CARMENES_VIS"):
    """ Animate the pulsation.

        At the moment only allow one instruments at a time.
    """
    conf = parse_ticket(ticket)
    sim_star = conf["name"]
    pulsations = read_in_saved_arrays_pulsation(sim_star)

    # Read in data
    rv_dict = load.rv(sim_star)
    time = np.array(rv_dict[instrument]["bjd"])
    time = time - 2400000
    rv = np.array(rv_dict[instrument]["rv"])
    rve = np.array(rv_dict[instrument]["rve"])
    crx_dict = load.crx(sim_star)
    crx = np.array(crx_dict[instrument]["crx"])
    crxe = np.array(crx_dict[instrument]["crxe"])
    dlw_dict = load.dlw(sim_star)
    dlw = np.array(dlw_dict[instrument]["dlw"])
    dlwe = np.array(dlw_dict[instrument]["dlwe"])

    rv = rv - np.median(rv)

    # Initialize the plot part
    images = pulsations
    # fig, ax = plt.subplots(4, 1, figsize=(16, 9))
    fig, ax = create_layout()
    index = 0
    im = ax[0].imshow(images[index], animated=True,
                      origin="lower", cmap="seismic", vmin=-50, vmax=50)
    ax = init_plots(ax, index + 1, time, rv, rve, crx, crxe)

    def create_plot(im, fig, ax, images, index, time, rv, rve, crx, crxe, dlw=False):
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
            ax = update_plots(ax, index + 1, time, rv, rve, crx, crxe, dlw=dlw)
            return im,

        ani = animation.FuncAnimation(
            fig, updatefig, images, interval=175, blit=False, repeat=False)
        outfolder = Path("/home/dane/Documents/PhD/pypulse/animations")
        if not dlw:
            ani.save(outfolder / f"{sim_star}_crx.gif")
        else:
            ani.save(outfolder / f"{sim_star}_dlw.gif")

        plt.close()
        fig, ax = create_layout()
        index = len(images) - 1
        im = ax[0].imshow(images[index], animated=True,
                          origin="lower", cmap="seismic", vmin=-100, vmax=100)
        ax = init_plots(ax, index, time, rv, rve, crx, crxe, )
        ax = update_plots(ax, index + 1, time, rv, rve, crx, crxe, dlw=dlw)
        if not dlw:
            plt.savefig(outfolder / f"{sim_star}_crx.pdf")
        else:
            plt.savefig(outfolder / f"{sim_star}_dlw.pdf")

    create_plot(im, fig, ax, images, index, time,
                rv, rve, crx, crxe, dlw=False)
    create_plot(im, fig, ax, images, index, time, rv, rve, dlw, dlwe, dlw=True)


def init_plots(ax, index, time, rv, rve, crx, crxe):
    """ Init plots."""
    ax[1].errorbar(time[:index], rv[:index],
                   yerr=rve[:index], linestyle="None", marker="o")

    ax[2].errorbar(time[:index], crx[:index],
                   yerr=crxe[:index], linestyle="None", marker="o")

    ax[3].errorbar(rv[:index], crx[:index],
                   yerr=crxe[:index], linestyle="None", marker="o")
    # if dlw is not None:
    #     ax[4].errorbar(time[:index], dlw[:index],
    #                    yerr=dlwe[:index], linestyle="None", marker="o")
    return ax


def update_plots(ax, index, time, rv, rve, crx, crxe, dlw=False):
    """ Init plots."""

    # update the first plot
    ax[1].clear()
    ax[1].errorbar(time[:index], rv[:index],
                   yerr=rve[:index], linestyle="None", marker="o")
    ax[1].set_xlim(time.min() - 1, time.max() + 1)
    ax[1].set_ylim(rv.min() - 5, rv.max() + 5,)
    ax[1].set_xlabel("Time [JD] - 2400000")
    ax[1].set_ylabel("RV [m/s]")
    ax[1].ticklabel_format(useOffset=False, style='plain')

    ax[2].clear()
    ax[2].errorbar(time[:index], crx[:index],
                   yerr=crxe[:index], linestyle="None", marker="o")
    ax[2].set_xlim(time.min() - 1, time.max() + 1)
    ax[2].set_ylim(crx.min() - 5, crx.max() + 5,)
    ax[2].set_xlabel("Time [JD] - 2400000")
    if not dlw:
        ax[2].set_ylabel("CRX [m/s/Np]")
    else:
        ax[2].set_ylabel("dLW [m^2/s^2]")
    ax[2].ticklabel_format(useOffset=False, style='plain')

    if len(ax) == 4:
        ax[3].clear()
        ax[3].errorbar(rv[:index], crx[:index],
                       yerr=crxe[:index], linestyle="None", marker="o")
        ax[3].set_xlim(rv.min() - 5, rv.max() + 5)
        ax[3].set_ylim(crx.min() - 5, crx.max() + 5)
        ax[3].set_xlabel("RV [m/s]")
        if not dlw:
            ax[3].set_ylabel("CRX [m/s/Np]")
        else:
            ax[3].set_ylabel("dLW [m^2/s^2]")

    # if dlw is not None:
    #     ax[4].clear()
    #     ax[4].errorbar(time[:index], dlw[:index],
    #                    yerr=dlwe[:index], linestyle="None", marker="o")
    #     ax[4].set_xlim(time.min() - 1, time.max() + 1)
    #     ax[4].set_ylim(dlw.min() - 1, dlw.max() + 1)
    #     ax[4].set_xlabel("Time [JD] - 2400000")
    #     ax[4].set_ylabel("dLW [m^2/s^2]")
    #     ax[4].ticklabel_format(useOffset=False, style='plain')

    return ax


if __name__ == "__main__":
    ticket = "/home/dane/Documents/PhD/pypulse/data/fake_spectra/talk_ngc2423_0_dt200_k100_vrot3000/talk_ticket.ini"
    create_high_res_arrays(ticket)
    animate_pulse(ticket)
