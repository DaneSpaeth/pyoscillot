import matplotlib.pyplot as plt
import numpy as np
from three_dim_star import ThreeDimStar, TwoDimProjector
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation


def create_projections(t, params_dict, inclination, proj_func):
    """ Dynamically create the projections and return a dict of title and array
    """
    l = params_dict["l"]
    return_dict = {}
    for m in range(-l, l + 1):
        star = ThreeDimStar()
        star.add_pulsation(t=t, m=m, **params_dict)
        projector = TwoDimProjector(star,
                                    inclination=inclination,
                                    limb_darkening=False,
                                    line_of_sight=True)
        return_dict[f"l={l}, m={m}"] = proj_func(projector)

    # combined
    star = ThreeDimStar()
    for m in range(-l, l + 1):
        print(f"Add l={l}, m={m}")
        star.add_pulsation(t=t, m=m, **params_dict)
    projector = TwoDimProjector(star,
                                inclination=inclination,
                                limb_darkening=False,
                                line_of_sight=True)
    return_dict["combined"] = proj_func(projector)

    return return_dict


def animate_radial_pulsation():
    """ Animate only the radial displacements."""
    # define params
    P = 500
    params_dict = {
        "l": 1,
        "nu": 1 / P,
        "v_p": 5,
        "k": 0,
        "T_phase": 0,
        "T_var": 100}
    inclination = 90

    ts = np.linspace(0, P, 50)

    v_p = params_dict["v_p"]

    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    plot_params = {"origin": "lower",
                   "cmap": "seismic",
                   "vmin": -v_p / 2,
                   "vmax": v_p / 2}

    def updatefig(t):
        nonlocal ax
        nonlocal params_dict
        nonlocal inclination

        plot_dict = create_projections(
            t, params_dict, inclination, TwoDimProjector.pulsation_rad)

        titles = plot_dict.keys()
        arrays = plot_dict.values()
        for title, array, a in zip(titles, arrays, ax.flatten()):
            a.imshow(array, **plot_params)
            a.set_title(title)

    ani = animation.FuncAnimation(
        fig, updatefig, ts, interval=175, blit=False, repeat=False)

    fig.suptitle("Radial Pulsation")
    ani.save("radial_pulsation.gif")


def animate_temp_variation():
    """ Animate only the radial displacements."""
    # define params
    P = 500
    params_dict = {
        "l": 1,
        "nu": 1 / P,
        "v_p": 5,
        "k": 0,
        "T_phase": 0,
        "T_var": 100}
    inclination = 90

    ts = np.linspace(0, P, 50)

    v_p = params_dict["v_p"]

    fig, ax = plt.subplots(2, 2, figsize=(9, 9))
    plot_params = {"origin": "lower",
                   "cmap": "seismic",
                   "vmin": 4800 - params_dict["T_var"],
                   "vmax": 4800 + params_dict["T_var"]}

    def updatefig(t):
        nonlocal ax
        nonlocal params_dict
        nonlocal inclination

        plot_dict = create_projections(
            t, params_dict, inclination, TwoDimProjector.temperature)

        titles = plot_dict.keys()
        arrays = plot_dict.values()
        for title, array, a in zip(titles, arrays, ax.flatten()):
            a.imshow(array, **plot_params)
            a.set_title(title)

    ani = animation.FuncAnimation(
        fig, updatefig, ts, interval=175, blit=False, repeat=False)

    fig.suptitle("Temperature Variations")
    ani.save("temperature.gif")


if __name__ == "__main__":
    animate_temp_variation()
