import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel
from scipy.interpolate import CubicSpline, interp1d
from dataloader import phoenix_spectrum, telluric_mask
from scipy.signal import periodogram


import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use('TkAgg')



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
        # diff_left = np.abs(left_spectrum - s)
        # diff_right = np.abs(right_spectrum - s)
        # left_idx = np.argmin(diff_left)
        # s = left_spectrum[left_idx]
        # right_idx = np.argmin(diff_right)
        # left_wave = left_wavelength[np.argmin(diff_left)]
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

    threshold = 0.1
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

def bisector_on_line(wave, spec, line_center, width=1, skip=0, outlier_clip=0.1):
    """ Calculate the bisector of a line centered around line_center with width width.

        :param np.array wave: Array of wavelengths. Same unit as line_center and outlier_clip
        :param np.array spec: Normalized spectrum
        :param float line_center: Central wavelength of line
        :param float width: Rough width of line in same units as wave
        :param int skip: Number of pixels to skip from the bottom
        :param float outlier_clip: Clip outliers larger than value from bisector
    """
    bisector_waves = []
    bisector_flux = []

    num_widths = 2

    print(num_widths*width)
    mask = np.logical_and(wave >= line_center - num_widths * np.abs(width),
                          wave <= line_center + num_widths * np.abs(width))

    mask_sp = spec < 0.93
    mask = np.logical_and(mask, mask_sp)

    wave_line = wave[mask]
    spec_line = spec[mask]

    center_idx = np.argmin(np.abs(wave_line - line_center))
    # Amount of datapoints to skip from the bottom
    left_wave = wave_line[:center_idx - skip]
    left_spec = spec_line[:center_idx - skip]
    right_wave = wave_line[center_idx + skip:]
    right_spec = spec_line[center_idx + skip:]

    # threshold = 0.1
    # right_mask = right_spec < 1 - threshold / 2
    # right_spec = right_spec[right_mask]
    # right_wave = right_wave[right_mask]

    # make both array strictly increasing for the right part
    incr_mask = np.diff(right_spec) > 0
    while not np.all(incr_mask[:-1]):
        incr_mask = np.diff(right_spec) > 0
        # We choose to not use the final point
        incr_mask = np.append(incr_mask, False)
        right_spec = right_spec[incr_mask]
        right_wave = right_wave[incr_mask]

    # And decreasing for the left
    decr_mask = np.diff(left_spec) < 0
    while not np.all(decr_mask[:-1]):
        decr_mask = np.diff(left_spec) < 0
        # We choose to not use the final point
        decr_mask = np.append(decr_mask, False)
        left_spec = left_spec[decr_mask]
        left_wave = left_wave[decr_mask]


    left_cs = interp1d(np.flip(left_spec), np.flip(left_wave), fill_value="extrapolate")
    right_cs = interp1d(right_spec, right_wave, fill_value="extrapolate")
    lin_sp = np.linspace(np.min(spec_line), np.max(spec_line), 35)

    left = left_cs(lin_sp)
    right = right_cs(lin_sp)
    bisector_waves = (left + right) / 2
    bisector_flux = lin_sp

    # Now mask out outliers
    outlier_mask = np.abs(line_center - bisector_waves) <= outlier_clip

    bisector_waves = bisector_waves[outlier_mask]
    bisector_flux = bisector_flux[outlier_mask]


    return bisector_waves, bisector_flux, left_wave, left_spec, right_wave, right_spec


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

    shift_spec = np.interp(rest_wavelength, wavelength, spectrum)

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
    print(sigma)
    if sigma < 1:
        sigma = 1

    # Interpolate on logarithmic grid
    f_log = np.interp(w_log, w_grid, spec)

    # Smooth convolving with Gaussian kernel
    gauss = Gaussian1DKernel(stddev=sigma)
    f_conv = convolve_fft(f_log, gauss)

    # Interpolate back on original wavelength grid
    f_sm = np.interp(w_grid, w_log, f_conv)

    # Return smoothed spectrum
    return f_sm


def _overplot_telluric_mask(ax):
    xlim_low = ax.get_xlim()[0]
    xlim_high = ax.get_xlim()[1]

    telluric_m = telluric_mask()
    telluric_w = telluric_m[:, 0]
    telluric_m = telluric_m[:, 1]

    mask_low =  telluric_w >= xlim_low
    mask_high = telluric_w <= xlim_high
    mask = np.logical_and(mask_low, mask_high)
    telluric_m = telluric_m[mask]
    telluric_w = telluric_w[mask]

    for w, w_next, m, m_next in zip(telluric_w[0:-1], telluric_w[1:], telluric_m[0:-1], telluric_m[1:]):
        if m:
            if m_next:
                ax.axvspan(w, w_next, alpha=0.5, color='grey')


def rebin(wold, sold, wnew):
    """Interpolates OR integrates a spectrum onto a new wavelength scale, depending
    on whether number of pixels per angstrom increases or decreases. Integration
    is effectively done analytically under a cubic spline fit to old spectrum.

    Ported to from rebin.pro (IDL) to Python by Frank Grundahl (FG).
    Original program written by Jeff Valenti.

    IDL Edit History:
    ; 10-Oct-90 JAV Create.
    ; 22-Sep-91 JAV Translated from IDL to ANA.
    ; 27-Aug-93 JAV Fixed bug in endpoint check: the "or" was essentially an "and".
    ; 26-Aug-94 JAV Made endpoint check less restrictive so that identical old and
    ;       new endpoints are now allowed. Switched to new Solaris library
    ;       in call_external.
    ; Nov01 DAF eliminated call_external code; now use internal idl fspline
    ; 2008: FG replaced fspline with spline

    :param wold: Input wavelength vector.
    :type wold: ndarray[nr_pix_in]
    :param sold: Input spectrum to be binned.
    :type sold: ndarray[nr_pix_in]
    :param wnew: New wavelength vector to bin to.
    :type wnew: ndarray[nr_pix_out]

    :return: Newly binned spectrum.
    :rtype: ndarray[nr_pix_out]
    """

    def idl_rebin(a, shape):
        sh = shape[0], a.shape[0] // shape[0], shape[1], a.shape[1] // shape[1]
        return a.reshape(sh).mean(-1).mean(1)

    # Determine spectrum attributes.
    nold = np.int32(len(wold))  # Number of old points
    nnew = np.int32(len(wnew))  # Number of new points
    psold = (wold[nold - 1] - wold[0]) / (nold - 1)  # Old pixel scale
    psnew = (wnew[nnew - 1] - wnew[0]) / (nnew - 1)  # New pixel scale

    # Verify that new wavelength scale is a subset of old wavelength scale.
    if (wnew[0] < wold[0]) or (wnew[nnew - 1] > wold[nold - 1]):
        logging.warning('New wavelength scale not subset of old.')

    # Select integration or interpolation depending on change in dispersion.

    if psnew < psold:

        # Pixel scale decreased ie, finer pixels
        # Interpolating onto new wavelength scale.
        dum = interp1d(wold, sold)  # dum  = interp1d( wold, sold, kind='cubic' ) # Very slow it seems.
        snew = dum(wnew)

    else:

        # Pixel scale increased ie more coarse
        # Integration under cubic spline - changed to interpolation.

        xfac = np.int32(psnew / psold + 0.5)  # pixel scale expansion factor

        # Construct another wavelength scale (W) with a pixel scale close to that of
        # the old wavelength scale (Wold), but with the additional constraint that
        # every XFac pixels in W will exactly fill a pixel in the new wavelength
        # scale (Wnew). Optimized for XFac < Nnew.

        dw = 0.5 * (wnew[2:] - wnew[:-2])  # Local pixel scale

        pre = np.float(2.0 * dw[0] - dw[1])
        post = np.float(2.0 * dw[nnew - 3] - dw[nnew - 4])

        dw = np.append(dw[::-1], pre)[::-1]
        dw = np.append(dw, post)
        w = np.zeros((nnew, xfac), dtype='float')

        # Loop thru subpixels
        for i in range(0, xfac):
            w[:, i] = wnew + dw * (np.float(2 * i + 1) / (2 * xfac) - 0.5)  # pixel centers in W

        nig = nnew * xfac  # Elements in interpolation grid
        w = np.reshape(w, nig)  # Make into 1-D

        # Interpolate old spectrum (Sold) onto wavelength scale W to make S. Then
        # sum every XFac pixels in S to make a single pixel in the new spectrum
        # (Snew). Equivalent to integrating under cubic spline through Sold.

        # dum    = interp1d( wold, sold, kind='cubic' ) # Very slow!
        # fill_value in interp1d added to deal with w-values just outside the interpolation range
        dum = interp1d(wold, sold, fill_value="extrapolate")
        s = dum(w)
        s = s / xfac  # take average in each pixel
        sdummy = s.reshape(nnew, xfac)
        snew = xfac * idl_rebin(sdummy, [nnew, 1])
        snew = np.reshape(snew, nnew)

    return snew


if __name__ == "__main__":
    num = 10000
    wave = np.linspace(6000, 6500, num)
    spec = np.ones_like(wave)

    interval = 1000
    peak_idx = np.arange(int(0+interval/2), num, interval)
    spec[peak_idx] += 0.1

    spec_res = adjust_resolution(wave, spec, R=115000, w_sample=100)




    plt.plot(wave, spec_res)
    plt.xlim(6124, 6126)
    plt.ylim(0.97, 1.05)
    plt.show()
