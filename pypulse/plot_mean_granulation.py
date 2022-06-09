import matplotlib.pyplot as plt
import numpy as np
from dataloader import granulation_map
from parse_ini import parse_global_ini
from physics import radiance_to_temperature, calc_granulation_velocity_rad
import cv2
from pathlib import Path


def plot_mean_granulation():
    """ Plot the mean granulation map"""
    tmppath = parse_global_ini()["datapath"] / "tmp_plots"
    granulation = granulation_map()
    mean_granulation = np.mean(granulation, axis=0)
    fig, ax = plt.subplots()
    img = ax.imshow(mean_granulation)
    fig.colorbar(img, ax=ax, label="Intensity [erg/cm^2/s/A/srad]")
    plt.savefig(tmppath / "mean_intensity.png")
    plt.close()

    fig, ax = plt.subplots()
    img = ax.imshow(granulation[343])
    fig.colorbar(img, ax=ax, label="Intensity [erg/cm^2/s/A/srad]")
    plt.savefig(tmppath / "random_intensity.png")
    plt.close()
    
def test_velocity_map():
    granulation_radiance = granulation_map()
    temperature = radiance_to_temperature(granulation_radiance)
    temp = temperature[1000]
    v_gran_rad = np.zeros_like(temp)
    # Crude way to find the granular lanes and the granules
    dividing_temp = 5100
    granule_mask = temp >= dividing_temp
    granular_lane_mask = temp < dividing_temp

    vel_rad = calc_granulation_velocity_rad(temp)

    size = granule_mask.shape[0]

    vel_rad3x3 = np.zeros((size*3, size*3))
    granule_mask3x3 = np.zeros((size*3, size*3), dtype=bool)
    for i in range(3):
        for j in range(3):
            vel_rad3x3[i*size:(i+1)*size, j*size:(j+1)*size] = vel_rad
            granule_mask3x3[i * size:(i + 1) * size, j * size:(j + 1) * size] = granule_mask

    vel_rad2x2 = vel_rad3x3[int(0.5*size):int(2.5*size), int(0.5*size):int(2.5*size)]
    granule_mask2x2 = granule_mask3x3[int(0.5*size):int(2.5*size), int(0.5*size):int(2.5*size)]

    vec_field = np.zeros((size*2, size*2, 2))
    for row in range(int(0.5*size), int(1.5*size)):
        for col in range(int(0.5*size), int(1.5*size)):
            if not granule_mask2x2[row, col]:
                vec = np.array([0, 0])
            else:
                coord = np.array((row, col))
                dist = distance_from_px(granule_mask2x2, coord[0], coord[1])
                dist[granule_mask2x2] = np.inf
                min_dist_coords = np.array(np.unravel_index(dist.argmin(), dist.shape))
                vec = min_dist_coords - coord
                normalization = vel_rad2x2[row, col] / np.linalg.norm(vec)
                vec = vec * normalization
            print(row, col)
            vec_field[row, col] = vec

    coords = np.linspace(0, size*2, size*2, dtype=int)
    cols, rows = np.meshgrid(coords, coords)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(granule_mask2x2, cmap="hot")
    # imshow "thinks" in row and col (that is also the way the numpy array is defined)
    # but quiver wants x and y, that is essentially the opposite
    # i.e. row corresponds to y and col to x
    qu = ax.quiver(cols, rows, -vec_field[:, :, 1], vec_field[:, :, 0], np.linalg.norm(vec_field, axis=2), scale=1e5)
    fig.colorbar(qu, ax=ax, label="Horizontal Velocity [m/s]")
    out_dir = Path("/home/dspaeth/data/simulations/tmp_plots/")
    plt.savefig(out_dir / "vec_field.png", dpi=300)
    plt.show()

    # fig, ax = plt.subplots()
    # img = ax.imshow(dist, cmap="jet")
    # ax.scatter(105, 70, marker="x", color="black")
    # fig.colorbar(img, ax=ax, label="Velocity [m/s]")
    # plt.show()


def distance_from_px(img, row, col):
    """ Compute the distance from a pixel"""
    coords = np.linspace(0, img.shape[0], img.shape[0], dtype=int)
    cols, rows = np.meshgrid(coords, coords)
    dist = np.sqrt(np.square(rows - row) +
                   np.square(cols - col))
    return dist

if __name__ == "__main__":
    test_velocity_map()