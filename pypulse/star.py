import numpy as np
from utils import (gaussian, adjust_resolution, interpolate_to_restframe)
from plapy.constants import C
from three_dim_star import ThreeDimStar, TwoDimProjector
from physics import get_ref_spectra, get_interpolated_spectrum


class GridSpectrumSimulator():
    """ Simulate a the spectrum of a star with a grid."""

    def __init__(self, N_star=500, N_border=5, Teff=4800, v_rot=3000,
                 inclination=90, T_var=0):
        """ Initialize grid.

            :param int N_star: number of grid cells on the star in x and y
                               (will automatically be adjusted to be odd
                                to have a center coordinate)
            :param int N_border: number of pixels at the border (left and right
                                 each)
            :param int Teff: Effective Temperature [K] (must be available)
            :param int vsini: V*sin(i) [m/s]
            :param float T_var: Temperature Variation of pulsation
        """
        self.three_dim_star = ThreeDimStar(
            Teff=Teff, T_var=T_var, V_p=20, k=100)
        self.three_dim_star.create_rotation(v_rot)
        # self.three_dim_star.add_granulation()
        self.projector = TwoDimProjector(self.three_dim_star,
                                         N=N_star, border=N_border,
                                         inclination=inclination,
                                         line_of_sight=True)
        self.spectrum = None
        self.flux = None

    def add_spot(self, phase=0.25, altitude=90, radius=25, T_spot=4300, ):
        """ Add a circular starspot at position x,y.

            :param phase: Phase ranging from 0 to 1 (0 being left edge,
                                                     1 being one time around)
            :param radius: Radius in degree
            :param altitude: Altitude on star in degree (90 is on equator)
            :param deltaT: Temp difference to Teff in K
            :param bool reset: If True, reset the temp before adding another spot
        """
        az = phase * 360.
        print(az)
        self.three_dim_star.add_spot(radius, phi_pos=az, T_spot=T_spot)

    def calc_spectrum(self, min_wave=5000, max_wave=12000, mode="phoenix"):
        """ Return Spectrum (potentially Doppler broadened) from min to max.

            :param float min_wave: Minimum Wavelength (Angstrom)
            :param flota max_wave: Maximum Wavelength (Angstrom)

            :returns: Array of wavelengths, array of flux value
        """

        wavelength_range = (min_wave - 0.25, max_wave + 0.25)

        self.temperature = self.projector.temperature()

        # First get a wavelength grid and dictionarys of the available
        # Phoenix spectra that are close to the temperature in
        if mode == "phoenix" or mode == "oneline":
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

        # Get the projected rotation and pulsation
        self.rotation = self.projector.rotation()
        self.pulsation = self.projector.pulsation()

        # Calc v over c
        v_c_rot = self.rotation / C
        v_c_pulse = self.pulsation / C

        total_spectrum = np.zeros(len(rest_wavelength))

        for row in range(self.temperature.shape[0]):
            for col in range(self.temperature.shape[1]):
                if not np.isnan(self.temperature[row, col]):
                    T_local = self.temperature[row, col]
                    if mode == "gaussian":
                        local_spectrum = ref_spectra[self.three_dim_star.Teff]
                    else:
                        _, local_spectrum, _ = get_interpolated_spectrum(T_local,
                                                                         ref_wave=rest_wavelength,
                                                                         ref_spectra=ref_spectra,
                                                                         ref_headers=ref_headers)

                    if not v_c_rot[row, col] and not v_c_pulse[row, col]:
                        # print(f"Skip Star Element {row, col}")
                        total_spectrum += local_spectrum
                    else:
                        print(f"Calculate Star Element {row, col}")

                        local_wavelength = rest_wavelength + \
                            v_c_rot[row, col] * rest_wavelength + \
                            v_c_pulse[row, col] * rest_wavelength
                        # Interpolate the spectrum to the same rest wavelength grid

                        interpol_spectrum = interpolate_to_restframe(local_wavelength,
                                                                     local_spectrum, rest_wavelength)
                        # interpol_spectrum = local_spectrum

                        total_spectrum += interpol_spectrum

        total_spectrum = total_spectrum  # / np.abs(np.median(total_spectrum))

        # Now adjust the resolution to Carmenes
        if mode == "oneline":
            total_spectrum += np.abs(total_spectrum.min())
            total_spectrum = total_spectrum / total_spectrum.max()
        else:
            total_spectrum = adjust_resolution(
                rest_wavelength, total_spectrum, R=90000)
        self.spectrum = total_spectrum

        # Also calculate the flux
        self.calc_flux()

        return rest_wavelength, total_spectrum

    def add_pulsation(self, l=2, m=2, phase=0):
        """ Add a pulsation to the star."""
        # TODO make these values adjustable
        t = phase / self.three_dim_star.nu
        self.three_dim_star.add_pulsation(t=t, l=l, m=m)

    def calc_flux(self):
        """ Calculate the local flux of the star.

            Sum up the total flux of the spectrum that you calculated.
        """
        if self.spectrum is None:
            return 0

        self.flux = np.sum(self.spectrum)
        return self.flux


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    star = GridSpectrumSimulator(N_star=30, N_border=1, v_rot=3000)
    star.calc_spectrum()
