import numpy as np
import plapy.constants as const
from dataloader import phoenix_spectrum, phoenix_spec_intensity
import matplotlib.pyplot as plt


def planck(wav, T):
    """ Return planck's law at wavelength and Temperature.

        :param float/np.array wav: Wavelength in m
        :param T: Temperature in K
    """
    a = 2.0 * const.H * const.C**2
    b = const.H * const.C / (wav * const.K_b * T)
    intensity = a / ((wav**5) * (np.exp(b) - 1.0))
    return intensity


def planck_ratio(wav, T_1, T_2):
    """ Return the ratio of the planck's law between T_1, T_2 per wavelength.

        R = S(T_1) / S(T_2)


        :param float/np.array wav: Wavelength in m
        :param T_1: Temp 1 in K
        :param T_2: Temp 2 in K
        :returns: int or np.array of ratio
    """
    int_1 = planck(wav, T_1)
    int_2 = planck(wav, T_2)
    return int_1 / int_2


# def get_interpolated_spectrum(T_local,
#                               ref_wave,
#                               ref_spectra,
#                               ref_headers=None):
#     """ Return a potentially interpolated spectrum. Returns the same format as
#         the phoenix spectrum.
#     """

#     T_low = int(np.floor(T_local / 100) * 100)
#     T_high = int(np.ceil(T_local / 100) * 100)

#     if T_low == T_high:
#         # print("No Temperature Interpolation")
#         spec = ref_spectra[T_low]
#         header = ref_headers[T_low]
#         wave = ref_wave
#         return wave, spec, header
#     else:

#         # print(f"Use the given Reference Spectra at T={T_close}")
#         assert ref_wave is not None, "Please add a Reference Wavelength using the ref_wave param"
#         assert ref_headers is not None, "Please add the Reference headers using the ref_headers param"
#         wave = ref_wave
#         spec_low = ref_spectra[T_low]
#         spec_high = ref_spectra[T_high]
#         header = ref_headers[T_low]

#         # Now interpolate with the contrast given by the Planck curves

#         ratio_high = 1 - (np.abs(T_high - T_local)) / 100
#         ratio_low = 1 - (np.abs(T_low - T_local)) / 100
#         spec_low_interpol = spec_low * \
#             planck_ratio(wave * 1e-10, T_local, T_low)
#         spec_high_interpol = spec_high * \
#             planck_ratio(wave * 1e-10, T_local, T_low)
#         spec = (spec_low_interpol * ratio_low +
#                 spec_high_interpol * ratio_high)

#     return wave, spec, header

def get_interpolated_spectrum(T_local,
                              ref_wave=None,
                              ref_spectra=None,
                              ref_headers=None,
                              spec_intensity=False,
                              mu_local=1,
                              ref_mu=None):
    """ Return a potentially interpolated spectrum. Returns the same format as
        the phoenix spectrum.

        At the moment:
        logg=3.0, feh=0.0
    """
    assert ref_spectra is not None, "Please add a Reference Wavelength using the ref_spectra param"
    assert ref_wave is not None, "Please add a Reference Wavelength using the ref_wave param"
    assert ref_headers is not None, "Please add the Reference headers using the ref_headers param"
    if spec_intensity:
        assert ref_mu is not None, "You have to provide mu if spec_intensity is True"

    T_close = int(round(T_local, -2))

    # Get closest spectrum
    wave = ref_wave
    header = ref_headers[T_close]
    if not spec_intensity:
        spec = ref_spectra[T_close]
    else:
        # The specific intensities are saved as a datacube
        spec_int_cube = ref_spectra[T_close]
        # Get the closest index of the mu
        idx = np.abs(ref_mu - mu_local).argmin()
        print(f"Closest mu to {mu_local} at idx={idx}")
        spec = ref_spectra[T_close][idx]

    assert wave.shape == spec.shape

    # Now interpolate with the contrast given by the Planck curves
    if int(T_local) != T_close:
        spec = spec * planck_ratio(wave * 1e-10, T_local, T_close)
    return wave, spec, header


def get_ref_spectra(T_grid, logg, feh, wavelength_range=(3000, 7000),
                    spec_intensity=False):
    """ Return a wavelength grid and a dict of phoenix spectra and a dict of
        pheonix headers.

        The dict has the form {Temp1:spectrum1, Temp2:spectrum2}

        This dict and wavelength grid can be used to interpolate the spectra
        for local T in a later step. Loading them beforhand as a dictionary
        reduces the amount of disk reading at later stages (i.e. if you
        read everytime you want to compute a local T spectrum)

        :param np.array T_grid: Temperature grid in K. The function
                                automatically determines the necessary ref spec
        :param float logg: log(g) for PHOENIX spectrum
        :param float feh: [Fe/H] for PHOENIX spectrum
        :param tuple wavelength_range: Wavelength range fro spectrum in A

        :param bool spec_intensity: If True it will return the spec intensity
                                    spectra cubes and the mu angle as the final
                                    parameter

        :return: tuple of (wavelength grid, T:spec dict, T:header dict, [mu])

    """
    T_grid = T_grid[~np.isnan(T_grid)]
    T_grid = T_grid[T_grid > 0]
    T_grid = np.round(T_grid, -2)
    T_unique = np.unique(T_grid)
    T_unique = T_unique.astype(int)

    # Append the next lowest and next highest values as well
    T_unique = np.insert(T_unique, 0, T_unique[0] - 100)
    T_unique = np.append(T_unique, T_unique[-1] + 100)

    # And now define a grid from the lowest to the highest value with all full 100s

    T_unique = np.linspace(np.min(T_unique), np.max(T_unique), int(
        (np.max(T_unique) - np.min(T_unique)) / 100) + 1, dtype=int)

    ref_spectra = {}
    ref_headers = {}
    if not spec_intensity:
        for T in T_unique:
            wave, ref_spectra[T], ref_headers[T] = phoenix_spectrum(
                Teff=float(T), logg=logg, feh=feh,
                wavelength_range=wavelength_range)
            # All waves are the same, so just return the last one

        return wave, ref_spectra, ref_headers
    else:
        for T in T_unique:
            wave, ref_spectra[T], mu, ref_headers[T] = phoenix_spec_intensity(
                Teff=float(T), logg=logg, feh=feh,
                wavelength_range=wavelength_range)
            # All waves are the same, also all mu should be the same
            # So just return the last one
        return wave, ref_spectra, ref_headers, mu
    
def spectral_radiance_to_temperature(spectral_radiance):
    """ Convert an array of spectral radiance to temperature.
    
    Following the formula from:
    https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law#Integration_of_intensity_derivation
    , i.e.: radiance = sigma / pi * T^4
    The factor pi in the end stems from the integration over the hemisphere onto which the energy is emitted
    but decreased by the integral over the cos(z) as the emission is Lambertian
    See: https://en.wikipedia.org/wiki/Radiance  under Description
    Not sure yet how to deal with the spectral part of it

    :param array_like or float spectral_radiance: Spectral radiance in erg/cm^2/s/Angstrom/sr
    """
    # stefan-boltzmann constant in erg/cm^2/s/K^4
    sigma = 5.670374e-5
    temperature = np.power(spectral_radiance / sigma * np.pi, 1 / 4)
    return temperature


if __name__ == "__main__":
    T_grid = np.ones((100, 100)) * 3100
    T_grid[0:20, :] = 3001
    T_grid[50:60, :] = 3200

    wave, ref_spectra, ref_headers = get_ref_spectra(
        T_grid, logg=2.0, feh=0.0, wavelength_range=(3000, 12000))

    wave, spec1, header3 = get_interpolated_spectrum(3050,
                                                     ref_wave=wave,
                                                     ref_spectra=ref_spectra,
                                                     ref_headers=ref_headers)
    wave, spec2, header3 = get_interpolated_spectrum(3060,
                                                     ref_wave=wave,
                                                     ref_spectra=ref_spectra,
                                                     ref_headers=ref_headers)
    plt.plot(wave, spec2)
    plt.plot(wave, spec1)
    plt.show()
