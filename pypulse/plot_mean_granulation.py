import matplotlib.pyplot as plt
import numpy as np
from dataloader import granulation_map
from parse_ini import parse_global_ini
from physics import spectral_radiance_to_temperature


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
    granulation_spectral_radiance = granulation_map()
    temperature = spectral_radiance_to_temperature(granulation_spectral_radiance)
    temp = temperature[1000]
    v_gran_rad = np.zeros_like(temp)
    # Crude way to find the granular lanes and the granules
    dividing_temp = 5100
    granule_mask = temp >= dividing_temp
    granular_lane_mask = temp < dividing_temp
    v_lane = 4500
    v_granule = -1500
    v_gran_rad[granular_lane_mask] = v_lane * (dividing_temp - temp[granular_lane_mask]) / (dividing_temp - np.min(temp))
    v_gran_rad[granule_mask] = v_granule * (dividing_temp - temp[granule_mask]) / (dividing_temp - np.max(temp))
    i = 0
    while np.abs(np.mean(v_gran_rad)) > 0.001:
        i += 1
        print(i, np.abs(np.mean(v_gran_rad)), v_lane, v_granule)
        if i > 1e6:
            break
        dv = 0.1
        if np.mean(v_gran_rad) > 0:
            v_lane -= dv
            v_granule -= dv
        else:
            v_lane += dv
            v_granule += dv
        v_gran_rad[granular_lane_mask] = v_lane * (dividing_temp - temp[granular_lane_mask]) / (
                dividing_temp - np.min(temp))
        v_gran_rad[granule_mask] = v_granule * (dividing_temp - temp[granule_mask]) / (
                dividing_temp - np.max(temp))



    print(np.mean(v_gran_rad))
    fig, ax = plt.subplots()
    img = ax.imshow(v_gran_rad, cmap="jet")
    fig.colorbar(img, ax=ax, label="Velocity [m/s]")
    plt.show()

if __name__ == "__main__":
    test_velocity_map()