import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel
from scipy.interpolate import CubicSpline
from dataloader import phoenix_spectrum


def create_circular_mask(h, w, center=None, radius=None):
    """ Create a circular mask.

        From https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    """

    if center is None:
        # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:
        # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask


def gaussian(x, mu=0, sigma=0.001):
    """ Return a gaussian at position x.

        :param float or array x: Position or array of positions at which to
                                 evaluate the Gaussian.
    """
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.square(x - mu) / (2 * sigma**2))


def bisector(wavelength, spectrum):
    """ Calculate the bisector of the line.

        Still work in progress. One must be careful what to pass here.

        :param wavelength: Array of wavelengths
        :param spectrum: Array of spectrum

        :returns: Array of bisector wavelengths, array of bisector flux
    """
    bisector_waves = []
    bisector_flux = []
    search_for = np.linspace(np.max(spectrum) - 0.05 * np.abs((np.max(spectrum) - np.min(spectrum))),
                             np.min(spectrum) + 0.01, 50)
    max_idx = np.argmin(spectrum)
    left_spectrum = spectrum[0:max_idx]
    # print(wavelength[max_idx])
    left_wavelength = wavelength[0:max_idx]
    right_spectrum = spectrum[max_idx:]
    right_wavelength = wavelength[max_idx:]
    for s in search_for:
        # print(s)
        #diff_left = np.abs(left_spectrum - s)
        # diff_right = np.abs(right_spectrum - s)
        #left_idx = np.argmin(diff_left)
        #s = left_spectrum[left_idx]
        # right_idx = np.argmin(diff_right)
        #left_wave = left_wavelength[np.argmin(diff_left)]
        # right_wave = right_wavelength[np.argmin(diff_right)]
        cs = CubicSpline(left_spectrum[::-1], left_wavelength[::-1])
        left_wave = cs(s)

        cs = CubicSpline(right_spectrum, right_wavelength,)
        right_wave = cs(s)

        bisector_wave = (right_wave + left_wave) / 2
        bisector_flux.append(s)
        bisector_waves.append(bisector_wave)

    return np.array(bisector_waves), np.array(bisector_flux)


def bisector_new(wave, spec, skip=2):
    """ Calculate the bisector of the line.

        Still work in progress. One must be careful what to pass here.

        :param wavelength: Array of wavelengths
        :param spectrum: Array of spectrum
        :param int skip: Number of datapoints to skip from the bottom of the line

        Must be normalized (1 continuum)

        :returns: Array of bisector wavelengths, array of bisector flux
    """
    bisector_waves = []
    bisector_flux = []

    center_idx = spec.argmin()
    # Amount of datapoints to skip from the bottom
    left_wave = wave[:center_idx - skip]
    left_spec = spec[:center_idx - skip]
    right_wave = wave[center_idx + skip:]
    right_spec = spec[center_idx + skip:]

    threshold = 0.03
    right_mask = right_spec < 1 - threshold / 2
    right_spec = right_spec[right_mask]
    right_wave = right_wave[right_mask]

    for w, s in zip(reversed(left_wave), reversed(left_spec)):
        if s > 1 - threshold:
            break
        cs = CubicSpline(right_spec, right_wave)
        right = cs(s)

        bisector_flux.append(s)
        bisector_waves.append((right + w) / 2)

    return np.array(bisector_waves), np.array(bisector_flux), wave[center_idx]


def add_doppler_shift(rest_spectrum, rest_wavelength, doppler_shift):
    """ Return shifted spectrum

        Wavelengths and shift in Angstrom.

        Assume that the spectrum outside the shifted area must not be correct.
    """
    if doppler_shift == 0:
        return rest_spectrum
    elif doppler_shift > 0:
        shift_idx = np.argmin(
            np.abs(rest_wavelength - (rest_wavelength[0] + doppler_shift)))
        if not shift_idx:
            return rest_spectrum
        shift_spectrum = np.zeros(len(rest_spectrum) + shift_idx)
        shift_spectrum[shift_idx:] = rest_spectrum
        shift_spectrum = shift_spectrum[:-shift_idx]
        return shift_spectrum
    elif doppler_shift < 0:
        shift_idx = np.argmin(
            np.abs(rest_wavelength - (rest_wavelength[-1] + doppler_shift)))
        if shift_idx == len(rest_spectrum) - 1:
            return rest_spectrum
        shift_spectrum = np.zeros(
            len(rest_spectrum) + len(rest_spectrum) - shift_idx)
        shift_spectrum[:shift_idx] = rest_spectrum[len(
            rest_spectrum) - shift_idx:]
        shift_spectrum = shift_spectrum[:len(rest_spectrum)]
        return shift_spectrum


def cut_to_maxshift(spectrum, wavelength, min_wave, max_wave):
    """ Cut the spectrum such that the maximal shift is masked."""
    min_idx = np.argmin(np.abs(wavelength - min_wave))
    max_idx = np.argmin(np.abs(wavelength - max_wave))
    spectrum = spectrum[min_idx:max_idx]
    wavelength = wavelength[min_idx:max_idx]

    return spectrum, wavelength


def interpolate_to_restframe(wavelength, spectrum, rest_wavelength):
    """ Interpolate the wavelength and spectrum to the rest_wavelength."""
    # CAUTION: At the moment I allow to extrapolate here
    func = interpolate.interp1d(wavelength, spectrum, fill_value="extrapolate")
    shift_spec = func(rest_wavelength)

    return shift_spec


def adjust_resolution(wave, spec, R, w_sample=1):
    '''
    Smears a model spectrum with a gaussian kernel to the given resolution, R.

    Modified from https://github.com/spacetelescope/pysynphot/issues/78
    Parameters
    -----------

    sp: spectrum


    R: int
        The resolution (dL/L) to smear to

    w_sample: int
        Oversampling factor for smoothing

    Returns
    -----------

    sp: PySynphot Source Spectrum
        The smeared spectrum
    '''

    # Save original wavelength grid and units
    w_grid = wave

    # Generate logarithmic wavelength grid for smoothing
    w_logmin = np.log10(np.nanmin(w_grid))
    w_logmax = np.log10(np.nanmax(w_grid))
    n_w = np.size(w_grid) * w_sample
    w_log = np.logspace(w_logmin, w_logmax, num=n_w)

    # Find stddev of Gaussian kernel for smoothing
    R_grid = (w_log[1:-1] + w_log[0:-2]) / (w_log[1:-1] - w_log[0:-2]) / 2
    sigma = np.median(R_grid) / R
    if sigma < 1:
        sigma = 1

    # Interpolate on logarithmic grid
    f_log = np.interp(w_log, w_grid, spec)

    # Smooth convolving with Gaussian kernel
    gauss = Gaussian1DKernel(stddev=sigma)
    f_conv = convolve_fft(f_log, gauss)

    # Interpolate back on original wavelength grid
    f_sm = np.interp(w_grid, w_log, f_conv)

    # Write smoothed spectrum back into Spectrum object
    return f_sm


def adjust_snr(spec, wave_template, spec_template, sig_template, snr=None):
    """ Adjust the signal to noise ratio of the spectrum.

        if snr is None the same as for the CARMENES template is taken

        :param np.array spec: 1d numpy array of spectrum values
        :param np.array wave_template: 1d numpy array of template wavelength
        :param np.array spec_template: 1d numpy array of template spectrum
        :param np.array sig_template: 1d numpy array of template error (noise)
    """
    if snr is None:
        snr = np.nanmedian(spec_template / sig_template)
    print("SNR:", snr)

    # We first need to make sure there are no nans in the sig_template
    if np.any(np.isnan(sig_template)):
        nan_idx = np.isnan(sig_template)
        sig_template[nan_idx] = np.interp(wave_template[nan_idx],
                                          wave_template[~nan_idx],
                                          sig_template[~nan_idx])
    # Now smooth the noise
    sig_template = gaussian_filter1d(sig_template, 40)

    # Now rescale the spec to have the desired snr
    current_snr = np.nanmedian(spec / sig_template)
    factor = snr / current_snr

    spec = spec * np.abs(factor)

    return spec


if __name__ == "__main__":
    from plapy.constants import C

    line = 6254.29
    wave, spec, _ = phoenix_spectrum(
        wavelength_range=(line - 0.25, line + 0.25))
    spec = spec / np.max(spec)
    bis_wave, bis = bisector_new(wave, spec)
    # plt.plot(wave, spec, marker=".", linestyle="None")
    plt.plot((bis_wave - np.median(bis_wave)) / np.median(bis_wave) * C, bis)
    plt.show()
