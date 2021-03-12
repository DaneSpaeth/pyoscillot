import numpy as np
from utils import create_circular_mask, interpolate_to_restframe
import dataloader as load
from plapy.constants import C
from scipy.special import sph_harm
from spherical_geometry import create_starmask, calculate_pulsation


class GridStar():
    """ Simulate a star with a grid."""

    def __init__(self, N=500, Teff=4800, vsini=1000):
        """ Initialize grid.

            :param int N: number of grid cells in each direction
            :param int Teff: Effective Temperature [K] (must be available)
            :param int vsini: V*sin(i) [m/s]
        """
        if not N % 2:
            N += 1
        self.orig_N = N
        self.Teff = Teff
        self.border = 0
        self.star = create_starmask(N=N, border=self.border)
        self.N = self.star.shape[0]
        self.grid = np.zeros(self.star.shape)
        print(self.grid.shape)
        self.rotation = self.grid
        self.pulsation = self.grid
        self.center = (
            int(self.star.shape[0] / 2), int(self.star.shape[1] / 2))

        self.spot = False
        self.add_rotation(vsini=vsini)
        self.temperature = Teff * self.star

    def add_rotation(self, vsini):
        """ Add Rotation, vsini in m/s"""
        pos_map = self.star
        rel_hor_dist = self.grid
        one_line = np.arange(0, self.N, dtype=int) - self.center[0]

        for i in range(rel_hor_dist.shape[0]):
            rel_hor_dist[i, :] = one_line

        self.rotation = (rel_hor_dist * self.star) / \
            np.max(rel_hor_dist) * vsini

    def add_spot(self, phase=0.25, radius=50, deltaT=1800, reset=True, ):
        """ Add a circular starspot at position x,y."""
        y = int(self.center[1])
        # x = int((phase * 2) % 2 * self.N)
        x = int(phase % 1 * self.N)
        print(f"Add circular spot with rad={radius} at {x},{y}")

        self.spot = create_circular_mask(
            self.N, self.N, center=(x, y), radius=radius)
        # Make sure to reset the temp before adding another spot
        if reset:
            self.temperature[self.star] = self.Teff
        self.temperature[self.spot] -= deltaT
        self.temperature[np.logical_not(self.star)] = 0

    def calc_spectrum(self, min_wave=5000, max_wave=12000):
        """ Return Spectrum (potentially Doppler broadened) from min to max.

            :param float min_wave: Minimum Wavelength (Angstrom)
            :param flota max_wave: Maximum Wavelength (Angstrom)

            :returns: Array of wavelengths, array of flux value
        """
        wavelength_range = (min_wave - 1, max_wave + 1)
        rest_wavelength, rest_spectrum, _ = load.phoenix_spectrum(
            Teff=self.Teff, logg=2.5, feh=-0.5, wavelength_range=wavelength_range)

        if np.any(self.spot):
            print("Star has a spot")
            _, spot_spectrum, _ = load.phoenix_spectrum(
                Teff=3000, logg=3.0, feh=0.0, wavelength_range=wavelength_range)

        # Calc v over c
        v_c_rot = self.rotation / C
        v_c_pulse = self.pulsation.real / C

        total_spectrum = np.zeros(len(rest_spectrum))
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                if self.star[row, col]:
                    if self.temperature[row, col] == 3000:
                        local_spectrum = spot_spectrum
                    else:
                        local_spectrum = rest_spectrum
                    if not v_c_rot[row, col] and not v_c_pulse[row, col]:
                        print(f"Skip Star Element {row, col}")
                        total_spectrum += local_spectrum
                    else:
                        print(f"Calculate Star Element {row, col}")

                        local_wavelength = rest_wavelength + \
                            v_c_rot[row, col] * rest_wavelength + \
                            v_c_pulse[row, col] * rest_wavelength
                        # Interpolate the spectrum to the same rest wavelenght grid
                        interpol_spectrum = interpolate_to_restframe(local_wavelength,
                                                                     local_spectrum, rest_wavelength)
                        total_spectrum += interpol_spectrum

        total_spectrum = total_spectrum / np.median(total_spectrum)

        self.wavelength = rest_wavelength
        self.spectrum = total_spectrum
        return rest_wavelength, total_spectrum

    def add_pulsation(self, l=2, m=2, phase=0):
        """ Add a pulsation to the star."""
        pulsation_period = 600  # days
        nu = 1 / pulsation_period
        t = phase * pulsation_period
        V_p = 400
        k = 1.2
        self.pulsation, p_rad, p_phi, p_theta = calculate_pulsation(
            l, m, V_p, k, nu, t, N=self.orig_N, border=self.border)
        print(self.pulsation.shape)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    star = GridStar(N=50, vsini=0)
    wave_rest, spec_rest = star.calc_spectrum(7000, 7050)
    star.add_pulsation(phase=0)

    wave_0, spec_0 = star.calc_spectrum(7000, 7050)

    star = GridStar(N=50, vsini=0)
    star.add_pulsation(phase=0.5)
    wave_05, spec_05 = star.calc_spectrum(7000, 7050)

    # plt.imshow(star.pulsation.real, cmap="seismic",
    #           origin="lower", vmin=-400, vmax=400)
    plt.plot(wave_rest, spec_rest, label="Restframe")
    plt.plot(wave_0, spec_0, label="Phase 0")
    plt.plot(wave_05, spec_05, label="Phase 0.5")
    plt.legend()
    plt.show()
