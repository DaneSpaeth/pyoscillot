import numpy as np
import plapy.constants as const
from dataloader import phoenix_spectrum
import matplotlib.pyplot as plt


def planck(wav, T):
    """ Return planck's law at wavelength and Temperature.

        :param float/np.array wav: Wavelength in m
        :param T: Temperature in K
    """
    a = 2.0 * const.h * const.C**2
    b = const.h * const.C / (wav * const.k_b * T)
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


def get_interpolated_spectrum(T_local, wavelength_range=(3000, 7000)):
    """ Return a potentially interpolated spectrum. Returns the same format as
        the phoenix spectrum.

        At the moment:
        logg=3.0, feh=0.0
    """
    if not 4450 < T_local < 5150:
        print("Temperature is not in grid")
        exit()

    T_close = round(T_local, -2)
    print(T_close)
    # Sadly the 5000K phoenix spectrum is not available
    T_missing = 5000
    if T_close == T_missing:
        if T_local < T_missing:
            T_close -= 100
        else:
            T_close += 100

    # Get closest spectrum
    wave, spec, header = phoenix_spectrum(
        T_close, logg=3.0, feh=0.0, wavelength_range=wavelength_range)

    # Now interpolate with the contrast given by the Planck curves
    if T_local != T_close:
        spec = spec * planck_ratio(wave * 1e-10, T_local, T_close)
    return wave, spec, header
