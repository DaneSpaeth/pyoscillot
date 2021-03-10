import numpy as np
from utils import create_circular_mask, interpolate_to_restframe
import dataloader as load
from plapy.constants import C
from scipy.special import sph_harm


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
        self.N = N
        self.Teff = Teff
        self.grid = np.zeros((N, N))
        self.rotation = self.grid
        self.pulsation = self.grid
        self.center = (int(N / 2), int(N / 2))

        self.star = create_circular_mask(N, N)
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

    def spectrum(self, min_wave=5000, max_wave=12000):
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
        v_c = self.rotation / C

        total_spectrum = np.zeros(len(rest_spectrum))
        for row in range(self.N):
            for col in range(self.N):
                if self.star[row, col]:
                    if self.temperature[row, col] == 3000:
                        local_spectrum = spot_spectrum
                    else:
                        local_spectrum = rest_spectrum
                    if not v_c[row, col]:
                        # print(f"Skip Star Element {row, col}")
                        total_spectrum += local_spectrum
                    else:
                        # print(f"Calculate Star Element {row, col}")

                        local_wavelength = rest_wavelength + \
                            v_c[row, col] * rest_wavelength
                        # Interpolate the spectrum to the same rest wavelenght grid
                        interpol_spectrum = interpolate_to_restframe(local_wavelength,
                                                                     local_spectrum, rest_wavelength)
                        total_spectrum += interpol_spectrum

        total_spectrum = total_spectrum / np.median(total_spectrum)

        self.wavelength = rest_wavelength
        self.spectrum = total_spectrum
        return rest_wavelength, total_spectrum

    def add_pulsation(self, n=1, l=3, m=3):
        """ Add a pulsation to the star."""
        phi = np.linspace(0, np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)

        self.pulsation = np.zeros(self.grid.shape)
        for row in range(self.N):
            for col in range(self.N):
                if not self.star[row, col]:
                    self.pulsation[row, col] = 0
                else:
                    dx = col - self.center[0]
                    dy = row - self.center[1]
                    phi = np.arcsin(dy / self.N * 2)
                    theta = np.arcsin(dx / self.N * 2)

                    harm = sph_harm(m, l, phi, theta)

                    self.pulsation[row, col] = harm

        # The Cartesian coordinates of the unit sphere
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Calculate the spherical harmonic Y(l,m) and normalize to [0,1]
        harm = sph_harm(m, l, theta, phi).real
        print(harm.shape)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    star = GridStar(N=100, vsini=0)
    star.add_pulsation()

    plt.imshow(star.pulsation)
    plt.show()
