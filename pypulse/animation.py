import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from three_dim_star import ThreeDimStar, TwoDimProjector
from pathlib import Path
from check_time_series import read_in_rvs, _read_in_crx
import plapy.rv.dataloader as load


def _create_and_save_arrays():

    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60)
    grid = projector.grid()
    radius = 20
    phis = np.linspace(0, 360, 30)
    print(phis)
    star_temps = []
    spot_masks = []
    for phi in phis:
        print(f"Spot at Position {phi}")
        star.default_maps()
        star.add_spot(radius, phi_pos=phi)
        temp_map = projector.temperature()
        spotmask = np.where(np.logical_and(
            temp_map != star.Teff, ~np.isnan(temp_map)), True, False)
        np.save(f"arrays/spot_animation/temp_{round(phi,1)}phi", temp_map)
        np.save(f"arrays/spot_animation/spotmask_{round(phi,1)}phi", spotmask)
        star_temps.append(temp_map)
        spot_masks.append(spotmask)

def _create_and_save_arrays_pulsation():

    star = ThreeDimStar(k=100, V_p=1)
    projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
    grid = projector.grid()

    P = 600
    phases = np.linspace(0, 1, 30)
    print(phases)
    exit()
    folder = Path(f"arrays/{PULSATION_NAME}")
    if not folder.is_dir():
        folder.mkdir()
    for p in phases:
        star = ThreeDimStar(k=100, V_p=1)
        projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
        star.add_pulsation(l=1, m=1, t=p * P)
        np.save(f"arrays/{PULSATION_NAME}/{round(p,3)}",
                -1 * projector.pulsation())

        # fig, ax = plt.subplots(2, 2)
        # ax[0, 0].imshow(projector.pulsation_rad(), cmap="seismic",
        #                 origin="lower", vmin=-1, vmax=1)
        # ax[0, 1].imshow(projector.pulsation_phi(), cmap="seismic",
        #                 origin="lower", vmin=-50, vmax=50)
        # ax[1, 0].imshow(projector.pulsation_theta(),
        #                 cmap="seismic", origin="lower", vmin=-50, vmax=50)
        # ax[1, 1].imshow(projector.pulsation(), cmap="seismic",
        #                 origin="lower", vmin=-50, vmax=50)
        # plt.show()
        plt.close()


def read_in_saved_arrays_pulsation():
    path = Path(f"arrays/{PULSATION_NAME}")
    array_paths = path.glob("*npy")
    pulse_maps = []
    phis = []

    for array_path in sorted(array_paths):
        pulse_maps.append(np.load(array_path))
        phi = float(array_path.name.split(".npy")[0])
        phis.append(phi)
    pulse_maps = [-1 * pulse_map for _,
                  pulse_map in sorted(zip(phis, pulse_maps))]

    return pulse_maps


def read_in_saved_arrays():

    path = Path("arrays/spot_animation")
    temp_maps_paths = path.glob("temp_*phi.npy")
    spot_mask_paths = path.glob("spotmask_*phi.npy")
    temp_maps = []
    spot_masks = []
    phis = []
    for tmap_path in sorted(temp_maps_paths):
        temp_map = np.load(tmap_path)
        phi = float(str(tmap_path).split("temp_")[-1].split("phi")[0])
        temp_maps.append(temp_map)
        phis.append(phi)

    temp_maps = [temp_map for _, temp_map in sorted(zip(phis, temp_maps))]
    phis = []
    for spot_mask_path in sorted(spot_mask_paths):
        spot_mask = np.load(spot_mask_path)
        phi = float(str(spot_mask_path).split("spotmask_")[-1].split("phi")[0])
        spot_masks.append(spot_mask)
        phis.append(phi)
    spot_masks = [spot_mask for _, spot_mask in sorted(zip(phis, spot_masks))]

    return temp_maps, spot_masks


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


def animate_pulse():
    """ Animate the pulsation."""
    pulsations = read_in_saved_arrays_pulsation()

    # Read in data
    sim_star = PULSATION_NAME
    rv_dict = load.rv(sim_star)
    time = np.array(rv_dict["SIMULATION"][0])
    time = time - 2400000
    rv = np.array(rv_dict["SIMULATION"][1])
    rve = np.array(rv_dict["SIMULATION"][2])
    crx_dict = load.crx(sim_star)
    crx = np.array(crx_dict["SIMULATION"]["crx"])
    crxe = np.array(crx_dict["SIMULATION"]["crxe"])
    dlw_dict = load.dlw(sim_star)
    dlw = np.array(dlw_dict["SIMULATION"]["dlw"])
    dlwe = np.array(dlw_dict["SIMULATION"]["dlwe"])

    rv = rv - np.median(rv)

    # Initialize the plot part
    images = pulsations
    # fig, ax = plt.subplots(4, 1, figsize=(16, 9))
    fig, ax = create_layout()
    index = 0
    im = ax[0].imshow(images[index], animated=True,
                      origin="lower", cmap="seismic", vmin=-50, vmax=50)
    ax = init_plots(ax, index, time, rv, rve, crx, crxe)

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
            ax = update_plots(ax, index, time, rv, rve, crx, crxe, dlw=dlw)
            return im,

        ani = animation.FuncAnimation(
            fig, updatefig, images, interval=175, blit=False, repeat=False)
        if not dlw:
            ani.save("pulse_crx.gif")
        else:
            ani.save("pulse_dlw.gif")

        plt.close()
        fig, ax = create_layout()
        index = len(images) - 1
        im = ax[0].imshow(images[index], animated=True,
                          origin="lower", cmap="seismic", vmin=-100, vmax=100)
        ax = init_plots(ax, index, time, rv, rve, crx, crxe, )
        ax = update_plots(ax, index, time, rv, rve, crx, crxe, dlw=dlw)
        if not dlw:
            plt.savefig("pulse_crx.pdf")
        else:
            plt.savefig("pulse_dlw.pdf")

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


def animate_spot():
    """ Create a animation of a spot."""
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60)
    grid = projector.grid()

    temp_maps, spot_masks = read_in_saved_arrays()

    fig, ax = create_layout()
    index = 0

    images = []
    for tempmap, spotmask in zip(temp_maps, spot_masks):
        image = tempmap
        image = np.where(np.logical_and(grid.astype(bool), np.logical_not(
            spotmask.astype(bool))), 0, image)
        images.append(image)
    im = ax[0].imshow(images[index], animated=True,
                      origin="lower", cmap="hot", vmax=5200, vmin=3900)

    # Read in data
    sim_star = "spot_inclination"
    rv_dict = load.rv(sim_star)
    time = np.array(rv_dict["SIMULATION"][0])
    time = time - 2400000
    rv = np.array(rv_dict["SIMULATION"][1])
    rve = np.array(rv_dict["SIMULATION"][2])
    crx_dict = load.crx(sim_star)
    crx = np.array(crx_dict["SIMULATION"]["crx"])
    crxe = np.array(crx_dict["SIMULATION"]["crxe"])
    dlw_dict = load.dlw(sim_star)
    dlw = np.array(dlw_dict["SIMULATION"]["dlw"])
    dlwe = np.array(dlw_dict["SIMULATION"]["dlwe"])

    rv -= np.median(rv)
    crx -= np.median(crx)
    # dlw -= np.median(dlw)

    def create_plot(im, fig, ax, images, index, time, rv, rve, crx, crxe, dlw=False):
        """ Convienence function to create the same plots once with crx and once with dlw.
        """
        # Initialize the plot part
        ax = update_plots(ax, index + 1, time, rv, rve, crx, crxe, dlw=dlw)

        def updatefig(*args):
            nonlocal index
            nonlocal ax
            if index == len(images):
                index = 0
                return im,

            # update the image
            im.set_array(images[index])

            ax = update_plots(ax, index + 1, time, rv, rve, crx, crxe, dlw=dlw)
            index += 1

            return im,
        ani = animation.FuncAnimation(
            fig, updatefig, images, interval=175, blit=False, repeat=False)
        # plt.show()
        if not dlw:
            ani.save("spot_crx.gif")
        else:
            ani.save("spot_dlw.gif")
        plt.close()
        fig, ax = create_layout()
        index = len(images) - 1
        im = ax[0].imshow(images[index], animated=True,
                          origin="lower", cmap="hot", vmax=5200, vmin=3900)
        ax = init_plots(ax, index, time, rv, rve, crx, crxe)
        ax = update_plots(ax, index, time, rv, rve, crx, crxe, dlw=dlw)
        if not dlw:
            plt.savefig("spot_crx.pdf")
        else:
            plt.savefig("spot_dlw.pdf")

    # create_plot(im, fig, ax, images, index, time, rv, rve, crx, crxe, dlw=False)
    create_plot(im, fig, ax, images, index, time, rv, rve, dlw, dlwe, dlw=True)


def plot_simplanet():
    """ Plot the simulated planet."""
    # Read in data
    sim_star = "planet"
    rv_dict = load.rv(sim_star)
    time = np.array(rv_dict["SIMULATION"][0])
    time = time - 2400000
    rv = np.array(rv_dict["SIMULATION"][1])
    rve = np.array(rv_dict["SIMULATION"][2])
    crx_dict = load.crx(sim_star)
    crx = np.array(crx_dict["SIMULATION"]["crx"])
    crxe = np.array(crx_dict["SIMULATION"]["crxe"])
    dlw_dict = load.dlw(sim_star)
    dlw = np.array(dlw_dict["SIMULATION"]["dlw"])
    dlwe = np.array(dlw_dict["SIMULATION"]["dlwe"])

    fig, ax = plt.subplots(3, 1, figsize=(16, 9))
    index = len(rv) - 1
    new_ax = [None, *ax[:3]]
    ax = new_ax
    ax = update_plots(ax, index, time, rv, rve, crx, crxe)

    ax[3].clear()
    ax[3].errorbar(time[:index], dlw[:index],
                   yerr=dlwe[:index], linestyle="None", marker="o")
    ax[3].set_xlim(time.min() - 1, time.max() + 1)
    ax[3].set_ylim(dlw.min() - 1, dlw.max() + 1)
    ax[3].set_xlabel("Time [JD] - 2400000")
    ax[3].set_ylabel("dLW [m^2/s^2]")
    ax[3].ticklabel_format(useOffset=False, style='plain')
    plt.tight_layout()
    plt.savefig("planet.pdf")


if __name__ == "__main__":
    # _create_and_save_arrays()
    # animate_spot()
    PULSATION_NAME = "pulsation_l1m1_k100_vp1_incl60"
    _create_and_save_arrays_pulsation()
    # read_in_saved_arrays_pulsation()
    # animate_pulse()
    # plot_simplanet()
