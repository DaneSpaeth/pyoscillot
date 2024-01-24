import matplotlib.pyplot as plt
import numpy as np
from dataloader import granulation_map
from cfg import parse_global_ini
from physics import radiance_to_temperature, calc_granulation_velocity_phi_theta
import cv2
from pathlib import Path
from scipy.signal import find_peaks


def plot_mean_granulation():
    """ Plot the mean granulation map"""
    tmppath = parse_global_ini()["datapath"] / "tmp_plots"
    granulation = granulation_map()

    mean_granulation = np.mean(granulation, axis=0)
    random_intensity = granulation[1592]
    vmin = random_intensity.min()
    vmax = random_intensity.max()
    fig, ax = plt.subplots()
    img = ax.imshow(granulation[1592], vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=ax, label="Intensity [erg/cm^2/s/A/srad]")
    plt.tight_layout()
    plt.savefig(tmppath / "random_intensity.png")
    plt.close()

    fig, ax = plt.subplots()
    img = ax.imshow(mean_granulation,  vmin=vmin, vmax=vmax)
    fig.colorbar(img, ax=ax, label="Intensity [erg/cm^2/s/A/srad]")
    plt.savefig(tmppath / "mean_intensity_full.png")
    plt.tight_layout()
    plt.close()

    fig, ax = plt.subplots()
    img = ax.imshow(mean_granulation)
    fig.colorbar(img, ax=ax, label="Intensity [erg/cm^2/s/A/srad]")
    plt.savefig(tmppath / "mean_intensity_full_noscale.png")
    plt.tight_layout()
    plt.close()


def test_velocity_map():
    granulation_radiance = granulation_map()
    temperature = radiance_to_temperature(granulation_radiance)
    temp = temperature[992 + 1292]
    # Crude way to find the granular lanes and the granules
    vel_phi, vel_theta, vec_field, granule_mask2x2 = calc_granulation_velocity_phi_theta(temp)

    size = temp.shape[0]
    coords = np.linspace(0, size, size, dtype=int)
    cols, rows = np.meshgrid(coords, coords)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(granule_mask2x2, cmap="hot")
    # imshow "thinks" in row and col (that is also the way the numpy array is defined)
    # but quiver wants x and y, that is essentially the opposite
    # i.e. row corresponds to y and col to x

    # cols = 50
    # rows = 50
    # vel_phi = 100
    # vel_theta = 0
    # It makes sense to plot the image with the theta axis going up down, and phi going left, right
    # Since quiver thinks in x and y that mean that vel_theta should be plotted with a minus
    # but vel_phi not
    # But, to get the signs correctly we add an additional - in the velocities, so here you need to flip the signs
    qu = ax.quiver(cols, rows, -vel_phi, vel_theta, np.linalg.norm(vec_field, axis=2), scale=1e5)
    ax.set_ylabel("Theta (polar angle) (arb. units)")
    ax.set_xlabel("Phi (azimuthal angle) (arb. units)")
    fig.colorbar(qu, ax=ax, label="Horizontal Velocity [m/s]")
    out_dir = Path("/home/dspaeth/data/simulations/tmp_plots/")
    plt.savefig(out_dir / "vec_field_phi_theta.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    img = ax[0].imshow(vel_theta, cmap="jet", vmin=-1500, vmax=1500)
    fig.colorbar(img, ax=ax[0], label="Theta Velocity [m/s]")

    img = ax[1].imshow(vel_phi, cmap="jet", vmin=-1500, vmax=1500)
    fig.colorbar(img, ax=ax[1], label="Phi Velocity [m/s]", )
    plt.tight_layout()
    plt.savefig(out_dir / "vel_components.png", dpi=600)
    plt.close()


def plot_slice():
    granulation_radiance = granulation_map()
    temperature = radiance_to_temperature(granulation_radiance)
    temp = temperature[3000]

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(temp)
    slice = temp[70, :]
    slice = np.append(slice[-20:], slice)
    slice = np.append(slice, slice[0:20])


    peaks, _ = find_peaks(-slice, height=-5100)


    ax[1].plot(slice)
    ax[1].plot(peaks.astype(float), np.take(slice, peaks) , "ro")
    plt.show()

def plot_overlaps():
    intensity = granulation_map()

    plt.imshow(intensity[991+1292-20, :, : ])
    plt.show()

def plot_granulation_map():
    from scipy.ndimage import sobel
    granulation_radiance = granulation_map()
    temperature = radiance_to_temperature(granulation_radiance)
    temp = temperature[992 + 1292]
    # temp = temperature[0]

    # v_rad = calc_granulation_velocity_rad

    DIVIDING_TEMP = 5100
    granule_mask = temp >= DIVIDING_TEMP
    granular_lane_mask = temp < DIVIDING_TEMP


    fig, ax = plt.subplots(2, 2, figsize=(10, 10))

    ax[0, 0].set_title("Temperature")
    img = ax[0, 0].imshow(temp)
    # fig.colorbar(img, ax=ax[0])


    ax[0, 1].set_title(f"T < {DIVIDING_TEMP}K")
    ax[0, 1].imshow(granular_lane_mask)

    edges_hor = np.abs(sobel(temp, mode="wrap", axis=0))
    edges_ver = np.abs(sobel(temp, mode="wrap", axis=1))
    edge_hor_mask = edges_hor > 1000
    edge_ver_mask = edges_ver > 1000
    edge_mask = np.logical_or(edge_hor_mask, edge_ver_mask)
    img = ax[1, 0].imshow(edge_mask)
    ax[1, 0].set_title("Edge Mask (Sobel Operator)")

    new_mask = np.logical_or(edge_mask, granular_lane_mask)
    ax[1, 1].set_title("Granule mask")
    ax[1, 1].imshow(new_mask)

    fig.set_tight_layout(True)

    out_dir = Path("/home/dspaeth/data/simulations/tmp_plots/")
    plt.savefig(out_dir / f"granule_masp_T<{DIVIDING_TEMP}.png", dpi=300)

    plt.show()



    # fig, ax = plt.subplots()
    # img = ax.imshow(dist, cmap="jet")
    # ax.scatter(105, 70, marker="x", color="black")
    # fig.colorbar(img, ax=ax, label="Velocity [m/s]")
    # plt.show()

def plot_heightmap():
    from mpl_toolkits.mplot3d import Axes3D
    granulation_radiance = granulation_map()
    temperature = radiance_to_temperature(granulation_radiance)
    temp = temperature[992 + 1292]



    grad = np.gradient(temp)
    xx, yy, = np.meshgrid(range(temp.shape[0]), range(temp.shape[1]))

    threedheight = np.dstack((xx, yy, temp))
    threedgradient = np.gradient(threedheight)

    fig = plt.figure(2, figsize=(16,9))
    ax = fig.add_subplot(121, projection='3d')
    # ax.plot_surface(threedheight, cmap="hot")
    # ax.scatter(threedheight[:,:,0], threedheight[:,:,1], threedheight[:,:,2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()
    ax.quiver(threedheight[:,:,0], threedheight[:,:,1], threedheight[:,:,2], 1, 0, 0, length=10, width=0.001)


    ax.elev = 10
    ax.azim = 90


    # Plot the image
    ax2 = fig.add_subplot(122)
    ax2.imshow(temp, cmap="hot")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.quiver(xx, yy, -grad[1], grad[0], scale=10000)


    fig.set_tight_layout(True)

    out_dir = Path("/home/dspaeth/data/simulations/tmp_plots/")
    # plt.savefig(out_dir / f"temp_height_elev{ax.elev}_gradient.png", dpi=300)
    plt.savefig(out_dir / "GRADIENT.png", dpi=300)

    plt.show()

def test_radial_velocity():
    granulation_radiance = granulation_map()
    temperature = radiance_to_temperature(granulation_radiance)
    temp = temperature[992 + 1292]

    v_rad = -2000 + 3000*((temp - np.min(temp)) / (np.max(temp) - np.min(temp)))
    v_rad -= np.mean(v_rad)

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    img = ax[0].imshow(-v_rad, vmin=-2000, vmax=2000, cmap="jet")
    fig.colorbar(img, ax=ax, label="Vertical Velocity [m/s]", location="bottom")
    ax[0].set_title("Vertical Velocity")

    img = ax[1].imshow(v_rad<0, cmap="jet")
    ax[1].set_title("Red: Vertical Velocity > 0, Blue: Vertical Velocity < 0")

    # fig.set_tight_layout(True)
    out_dir = Path("/home/dspaeth/data/simulations/tmp_plots/")
    plt.savefig(out_dir / f"new_vertical_velocity.png", dpi=300)

    plt.show()


def test_gradient():
    granulation_radiance = granulation_map()
    temperature = radiance_to_temperature(granulation_radiance)
    temp = temperature[992 + 1292]

    # TODO add the images next to it to calculate the gradient at the border

    v_rad = -2000 + 3000 * ((temp - np.min(temp)) / (np.max(temp) - np.min(temp)))
    v_rad -= np.mean(v_rad)
    xx, yy, = np.meshgrid(range(temp.shape[0]), range(temp.shape[1]))

    grad_x, grad_y = np.gradient(temp)
    normalization = np.sqrt(np.square(grad_x) + np.square(grad_y))

    grad_x_norm = grad_x / normalization
    grad_y_norm = grad_y / normalization

    fig, ax = plt.subplots(1, 2, figsize=(16, 9))
    ax[0].imshow(temp, cmap="hot")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].quiver(xx, yy, -grad_y, grad_x, scale=5000)
    ax[0].set_title("Gradient")

    ax[1].imshow(temp, cmap="hot")
    ax[1].set_xlabel("X")
    ax[1].set_ylabel("Y")
    ax[1].quiver(xx, yy, grad_x_norm, grad_x_norm, scale=100)
    ax[1].set_title("Normalized Gradient")


    out_dir = Path("/home/dspaeth/data/simulations/tmp_plots/")
    fig.set_tight_layout(True)
    # plt.savefig(out_dir / "GRADIENT.png", dpi=300)
    plt.show()

    # Next calculate the cells that influence each other
    # Idea define a nex pixel with the center at the location at which the vector points to
    # xx_off = xx + grad_x
    # yy_off = yy + grad_y






if __name__ == "__main__":
    test_case()