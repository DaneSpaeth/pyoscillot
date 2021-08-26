import numpy as np
import plapy.constants as const
from dataloader import phoenix_spectrum
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


def get_interpolated_spectrum(T_local,
                              wavelength_range=(3000, 7000),
                              ref_wave=None,
                              ref_spectra=None,
                              ref_headers=None):
    """ Return a potentially interpolated spectrum. Returns the same format as
        the phoenix spectrum.

        At the moment:
        logg=3.0, feh=0.0
    """

    T_close = int(round(T_local, -2))

    # Get closest spectrum
    if ref_spectra is None:
        print("I AM HERE")
        wave, spec, header = phoenix_spectrum(
            T_close, logg=3.0, feh=0.0, wavelength_range=wavelength_range)
    else:
        # print(f"Use the given Reference Spectra at T={T_close}")
        assert ref_wave is not None, "Please add a Reference Wavelength using the ref_wave param"
        assert ref_headers is not None, "Please add the Reference headers using the ref_headers param"
        wave = ref_wave
        spec = ref_spectra[T_close]
        header = ref_headers[T_close]
        assert wave.shape == spec.shape

        # Now interpolate with the contrast given by the Planck curves
    if int(T_local) != T_close:
        spec = spec * planck_ratio(wave * 1e-10, T_local, T_close)
    return wave, spec, header


def get_ref_spectra(T_grid, logg, feh, wavelength_range=(3000, 7000)):
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

        :return: tuple of (wavelength grid, T:spec dict, T:header dict)

    """
    T_grid = T_grid[~np.isnan(T_grid)]
    T_grid = T_grid[T_grid > 0]
    T_grid = np.round(T_grid, -2)
    T_unique = np.unique(T_grid)
    T_unique = T_unique.astype(int)

    ref_spectra = {}
    ref_headers = {}
    for T in T_unique:
        wave, ref_spectra[T], ref_headers[T] = phoenix_spectrum(
            Teff=float(T), logg=logg, feh=feh, wavelength_range=wavelength_range)
        # All waves are the same, so just return the last one

    return wave, ref_spectra, ref_headers


if __name__ == "__main__":
    T_grid = np.ones((100, 100)) * 4700.0
    T_grid[0:20, :] = 3000
    T_grid[50:60, :] = 4500
    T_grid[20:30, :] = 5201

    wave, ref_spectra, ref_headers = get_ref_spectra(
        T_grid, logg=2.0, feh=0.0, wavelength_range=(3000, 12000))
    print(ref_spectra.keys())

    # wave, ref_spectra, ref_headers = get_ref_spectra(
    #     T_grid, wavelength_range=(3000, 12000))

    # wave3, spec3, header3 = get_interpolated_spectrum(3000,
    #                                                   ref_wave=wave,
    #                                                   ref_spectra=ref_spectra,
    #                                                   ref_headers=ref_headers)
    # # plt.plot(wave3, spec3)
    # # plt.plot(wave2, spec2)
    # plt.plot(wave3, spec3)
    # plt.show()
