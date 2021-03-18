import numpy as np
from utils import (create_circular_mask, interpolate_to_restframe,
                   gaussian, bisector_new, adjust_resolution)
import dataloader as load
from plapy.constants import C
from scipy.special import sph_harm
from spherical_geometry import (create_starmask, calculate_pulsation,
                                calc_temp_variation, create_spotmask,
                                create_rotation)
from physics import get_ref_spectra, get_interpolated_spectrum


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

        self.spot = False
        self.add_rotation(vsini=vsini)
        self.temperature = (Teff * self.star).astype(float)

    def add_rotation(self, vsini):
        """ Add Rotation, vsini in m/s"""

        self.rotation = create_rotation(
            vsini, N=self.N_star, border=self.N_border, inclination=90,
            line_of_sight=True)

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
        wavelength_range = (min_wave - 0.25, max_wave + 0.25)

        # First get a wavelength grid and dictionarys of the available
        # Phoenix spectra that are close to the temperature in
        # self.temperature
        if mode == "phoenix":
            (rest_wavelength,
             ref_spectra,
             ref_headers) = get_ref_spectra(self.temperature,
                                            wavelength_range=wavelength_range)
        elif mode == "gaussian":
            print("Gaussian mode")
            # Does not work with Temperature variations at the moment
            center = 7000  # A

            rest_wavelength = np.linspace(
                center - 0.7, center + 0.7, 10000)
            spec = 1 - gaussian(rest_wavelength, center, 0.2)
            ref_spectra = {self.Teff: spec}
            ref_headers = {self.Teff: []}

        # if np.any(self.spot):
        #     print("Star has a spot")
        #     _, spot_spectrum, _ = load.phoenix_spectrum(
        #         Teff=3000, logg=3.0, feh=0.0, wavelength_range=wavelength_range)

        # Calc v over c
        print(np.nanmax(self.rotation))
        v_c_rot = self.rotation / C
        v_c_pulse = self.pulsation.real / C
        print(np.nanmin(v_c_pulse), np.nanmax(v_c_pulse))

        total_spectrum = np.zeros(len(rest_wavelength))
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                if self.star[row, col]:
                    T_local = self.temperature[row, col]
                    if mode == "gaussian":
                        local_spectrum = ref_spectra[self.Teff]
                    else:
                        _, local_spectrum, _ = get_interpolated_spectrum(T_local,
                                                                         ref_wave=rest_wavelength,
                                                                         ref_spectra=ref_spectra,
                                                                         ref_headers=ref_headers)
                    if not v_c_rot[row, col] and not v_c_pulse[row, col]:
                        # print(f"Skip Star Element {row, col}")
                        total_spectrum += local_spectrum
                    else:
                        # print(f"Calculate Star Element {row, col}")

                        local_wavelength = rest_wavelength + \
                            v_c_rot[row, col] * rest_wavelength + \
                            v_c_pulse[row, col] * rest_wavelength
                        # Interpolate the spectrum to the same rest wavelength grid
                        interpol_spectrum = interpolate_to_restframe(local_wavelength,
                                                                     local_spectrum, rest_wavelength)
                        total_spectrum += interpol_spectrum

        total_spectrum = total_spectrum  # / np.abs(np.median(total_spectrum))

        self.wavelength = rest_wavelength

        # Now adjust the resolution to Carmenes
        # total_spectrum = adjust_resolution(
        #    rest_wavelength, total_spectrum, R=90000)
        # if mode == "gaussian":
        total_spectrum += np.abs(total_spectrum.min())
        total_spectrum = total_spectrum / total_spectrum.max()
        self.spectrum = total_spectrum
        return rest_wavelength, total_spectrum

    def add_pulsation(self, l=2, m=2, k=1.2, phase=0):
        """ Add a pulsation to the star."""
        # TODO make these values adjustable
        self.l = l
        self.m = m
        self.pulsation_period = 600  # days
        self.nu = 1 / self.pulsation_period
        t = phase * self.pulsation_period
        V_p = 400
        # k = 1.2
        self.pulsation, p_rad, p_phi, p_theta = calculate_pulsation(
            l, m, V_p, k, self.nu, t, N=self.N_star, border=self.N_border)

        # TODO REMOVE
        # self.pulsation = p_rad / np.nanmax(p_rad.real) * V_p

        # print(np.nanmax(self.pulsation.real))

    def add_temp_variation(self, phase=0, reset=True):
        """ Add a temperature variation."""
        if reset:
            self.temperature[self.star] = self.Teff
        t = phase * self.pulsation_period
        ampl = 200  # K
        phase_shift = 0  # radians
        temp_variation, self.rad_no_lineofsight = calc_temp_variation(
            self.l, self.m, ampl, self.nu, t, phase_shift=phase_shift,
            N=self.N_star, border=self.N_border)
        temp_variation[np.isnan(temp_variation)] = 0
        self.temperature += temp_variation


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    save = False
    if save:
        star = GridStar(N_star=50, N_border=1, vsini=3000)
    else:
        cols = 5
        # fig, ax = plt.subplots(1, cols)
        fig, ax = plt.subplots(1, figsize=(4, 9))
    # line = 6254.29
    # wave, spec = star.calc_spectrum(line, line)
    # spec = spec / np.max(spec)
    # bis_wave, bis = bisector_new(wave, spec)

    # center = wave[spec.argmin()]
    # plt.plot(bis_wave, bis, marker=".", linestyle="None")
    # plt.show()
    # exit()
    # N = 12
    # phases = np.linspace(0, 1 - (1 / N), N - 1)
    phases = np.array([0, 0.25, 0.5, 0.75, 1.0])
    phases = np.array([0.25, 0.5, 0.75])
    for idx, p in enumerate(phases):
        line = 6254.29
        if save:
            star.add_pulsation(l=4, m=-4, k=0.15, phase=p)

            wave, spec = star.calc_spectrum(line, line)
            plt.imshow(star.pulsation.real, origin="lower",
                       vmin=-200, vmax=200, cmap="seismic")
            plt.savefig(f"arrays/{round(p, 4)}.pdf")
            # plt.show()
            # exit()
            plt.close()
            if not idx:
                np.save("arrays/wave.npy", wave)
            np.save(f"arrays/{round(p,4)}.npy", spec)
            continue
        if not idx:
            wave = np.load("arrays/wave.npy")
            # wave = wave[::100]
        spec = np.load(f"arrays/{round(p,4)}.npy")
        # print(wave
        # spec = spec[::100]

        # print(len(wave))

        row = int(idx / cols)
        col = idx % cols
        center = wave[spec.argmin()]

        # print((wave[-1] - wave[0]) / len(wave) * C / 1e3)
        bis_wave, bis, center = bisector_new(wave, spec)
        vs = ((bis_wave - line) / line) * C
        vs = vs + 280
        # ax[col].plot(vs, bis)
        # # ax[row, col].set_xlim(-600, 600)
        # # ax[row, col].plot((wave - center) / center * C,
        # #                   spec, label=(center - 7000) / center * C)
        # ax[col].set_xlabel("V [m/s]")
        # ax[col].set_xlim(-120, 120)
        # # ax[row, col].set_xlim(np.min(bis_wave), np.max(bis_wave))
        # # ax[row, col].axvline(center)
        # ax[col].legend()

        ax.plot(vs, bis, label=p - 0.25)

    ax.legend()
    plt.xlabel("V_r (m/s)")
    plt.ylabel("Relative Flux")
    plt.title("Bisector for l=4, m=-4, pulsation phases")
    plt.show()

    # plt.show()
