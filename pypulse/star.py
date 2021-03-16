import numpy as np
from utils import create_circular_mask, interpolate_to_restframe, gaussian, bisector
import dataloader as load
from plapy.constants import C
from scipy.special import sph_harm
from spherical_geometry import create_starmask, calculate_pulsation, calc_temp_variation, create_spotmask


class GridStar():
    """ Simulate a star with a grid."""

    def __init__(self, N_star=500, N_border=5, Teff=4800, vsini=1000):
        """ Initialize grid.

            :param int N_star: number of grid cells on the star in x and y
                               (will automatically be adjusted to be odd
                                to have a center coordinate)
            :param int N_border: number of pixels at the border (left and right
                                 each)
            :param int Teff: Effective Temperature [K] (must be available)
            :param int vsini: V*sin(i) [m/s]
        """
        if not N_star % 2:
            N_star += 1
        self.N_star = N_star
        self.N_grid = N_star + 2 * N_border
        self.N_border = N_border
        self.Teff = Teff
        self.star = create_starmask(N=N_star, border=N_border)
        self.grid = np.zeros((self.N_grid, self.N_grid))
        self.rotation = self.grid
        self.pulsation = self.grid
        self.center = (
            int(self.star.shape[0] / 2), int(self.star.shape[1] / 2))
        print(self.center)

        self.spot = False
        self.add_rotation(vsini=vsini)
        self.temperature = Teff * self.star

    def add_rotation(self, vsini):
        """ Add Rotation, vsini in m/s"""
        pos_map = self.star
        rel_hor_dist = self.grid
        one_line = np.arange(0, self.N_grid, dtype=int) - self.center[0]

        for i in range(rel_hor_dist.shape[0]):
            rel_hor_dist[i, :] = one_line

        self.rotation = (rel_hor_dist * self.star) / \
            np.max(rel_hor_dist) * vsini

    def add_spot(self, phase=0.25, altitude=90, radius=25, deltaT=1800, reset=True, ):
        """ Add a circular starspot at position x,y.



            :param phase: Phase ranging from 0 to 1 (0 being left edge,
                                                     1 being one time around)
            :param radius: Radius in degree
            :param altitude: Altitude on star in degree (90 is on equator)
            :param deltaT: Temp difference to Teff in K
            :param bool reset: If True, reset the temp before adding another spot
        """
        az = phase * 360
        self.spot = create_spotmask(
            radius, az, altitude, N=self.N_star, border=self.N_border)
        # Make sure to reset the temp before adding another spot
        if reset:
            self.temperature[self.star] = self.Teff
        self.temperature[self.spot] -= deltaT
        self.temperature[np.logical_not(self.star)] = 0

    def calc_spectrum(self, min_wave=5000, max_wave=12000, mode="phoenix"):
        """ Return Spectrum (potentially Doppler broadened) from min to max.

            :param float min_wave: Minimum Wavelength (Angstrom)
            :param flota max_wave: Maximum Wavelength (Angstrom)

            :returns: Array of wavelengths, array of flux value
        """
        wavelength_range = (min_wave - 1, max_wave + 1)
        if mode == "phoenix":
            rest_wavelength, rest_spectrum, _ = load.phoenix_spectrum(
                Teff=self.Teff, logg=2.5, feh=-0.5, wavelength_range=wavelength_range)
        elif mode == "gaussian":
            rest_wavelength = np.linspace(min_wave, max_wave, 10000)
            rest_spectrum = -1 * gaussian(rest_wavelength,
                                          (max_wave + min_wave) / 2,
                                          (max_wave - min_wave) / 100)

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
                        # Interpolate the spectrum to the same rest wavelength grid
                        interpol_spectrum = interpolate_to_restframe(local_wavelength,
                                                                     local_spectrum, rest_wavelength)
                        total_spectrum += interpol_spectrum

        total_spectrum = total_spectrum  # / np.abs(np.median(total_spectrum))

        self.wavelength = rest_wavelength
        self.spectrum = total_spectrum
        return rest_wavelength, total_spectrum

    def add_pulsation(self, l=2, m=2, phase=0):
        """ Add a pulsation to the star."""
        # TODO make these values adjustable
        self.l = l
        self.m = m
        self.pulsation_period = 600  # days
        nu = 1 / self.pulsation_period
        t = phase * self.pulsation_period
        V_p = 400
        k = 1.2
        self.pulsation, p_rad, p_phi, p_theta = calculate_pulsation(
            l, m, V_p, k, nu, t, N=self.N_star, border=self.N_border)

    def add_temp_variation(self, phase=0):
        t = phase * self.pulsation_period
        ampl = 100  # K
        phase_shift = 0  # radians
        temp_variation, self.rad_no_lineofsight = calc_temp_variation(
            self.l, self.m, ampl, t, phase_shift=phase_shift,
            N=self.N_star, border=self.N_border)
        self.temperature += temp_variation


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    star = GridStar(N_star=1000, N_border=3, vsini=0)

    phases = np.linspace(0, 1, 7)
    for p in phases:
        star.add_spot(p)
        plt.imshow(star.temperature, origin="lower")
        plt.show()
