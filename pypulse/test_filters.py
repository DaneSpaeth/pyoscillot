import numpy as np
import matplotlib.pyplot as plt
from physics import planck


def get_integrated_flux(filter_file, T):
    filter_wave = []
    filter_transmission = []
    with open(filter_file, "r") as f:
        for line in f:
            columns = line.strip().split()
            try:
                filter_wave.append(float(columns[0]))
                filter_transmission.append(float(columns[1]))
            except Exception as e:
                print(e)

    filter_wave = np.array(filter_wave)
    filter_transmission = np.array(filter_transmission)
    flux = planck(filter_wave * 1e-9, T=T)
    flux *= filter_transmission

    plt.plot(filter_wave, filter_transmission, label=filter_file)

    return np.sum(flux)


if __name__ == "__main__":
    T = 4850
    V_flux = get_integrated_flux("NOT_V_filter.txt", T)
    R_flux = get_integrated_flux("NOT_R_filter.txt", T)

    print(f"V_flux: {V_flux}")
    print(f"R_flux: {R_flux}")

    wave = np.linspace(450e-9, 1000e-9)
    flux = planck(wave, T=T)
    flux = flux / np.max(flux) * 80
    plt.plot(wave * 1e9, flux, label="Stellar Spectrum")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Flux")
    plt.legend()
    plt.show()
