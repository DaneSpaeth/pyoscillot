import matplotlib.pyplot as plt
import numpy as np
from dataloader import granulation_map
from parse_ini import parse_global_ini
from physics import radiance_to_temperature, calc_granulation_velocity_phi_theta
import cv2
from pathlib import Path


def plot_mean_granulation():
    """ Plot the mean granulation map"""
    tmppath = parse_global_ini()["datapath"] / "tmp_plots"
    granulation = granulation_map()

    print(granulation.shape)
    exit()
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
    temp = temperature[1000]
    # Crude way to find the granular lanes and the granules
    dividing_temp = 5100
    vel_phi, vel_theta, vec_field, granule_mask2x2 = calc_granulation_velocity_phi_theta(temp)

    size = temp.shape[0]
    coords = np.linspace(0, size, size, dtype=int)
    cols, rows = np.meshgrid(coords, coords)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(granule_mask2x2, cmap="hot")
    # imshow "thinks" in row and col (that is also the way the numpy array is defined)
    # but quiver wants x and y, that is essentially the opposite
    # i.e. row corresponds to y and col to x
    qu = ax.quiver(cols, rows, -vel_phi, vel_theta, np.linalg.norm(vec_field, axis=2), scale=1e5)
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
    temp = temperature[1000]

    slice = temp[70, :]

    plt.plot(slice)
    plt.show()

def plot_overlaps():
    intensity = granulation_map()

    plt.imshow(intensity[991+1292-20, :, : ])
    plt.show()



    # fig, ax = plt.subplots()
    # img = ax.imshow(dist, cmap="jet")
    # ax.scatter(105, 70, marker="x", color="black")
    # fig.colorbar(img, ax=ax, label="Velocity [m/s]")
    # plt.show()


if __name__ == "__main__":
    plot_mean_granulation()