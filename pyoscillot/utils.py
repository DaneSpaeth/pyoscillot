import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft, convolve
from astropy.convolution import Gaussian1DKernel
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import periodogram
from scipy.optimize import curve_fit
import subprocess
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from pathlib import Path
import cfg
from dataloader import phoenix_spectrum, telluric_mask, phoenix_spec_intensity, Rassine_outputs, Zhao_bis_polynomials, continuum
from physics import delta_relativistic_doppler
import copy


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

def neg_gaussian(x, mu=0, sigma=0.001):
    """ Return a negative gaussian at position x.

        :param float or array x: Position or array of positions at which to
                                 evaluate the Gaussian.
    """
    return 1 - gaussian(x, mu, sigma)


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

def _gauss_continuum(x, mu, sigma, amplitude, continuum):
    return continuum - (amplitude * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2))

def bisector_on_line(wave, spec, line_center, width=1, skip=0, outlier_clip=0.1, continuum=1.0, cutoff_high=0.97):
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

    num_widths = 2.5

    mask = np.logical_and(wave >= line_center - num_widths * np.abs(width),
                          wave <= line_center + num_widths * np.abs(width))

    mask_sp = spec < cutoff_high
    mask = np.logical_and(mask, mask_sp)
    

    wave_line = copy.deepcopy(wave)#[mask]
    spec_line = copy.deepcopy(spec)#[mask]
    
    # print(wave_line)
    # print(spec_line)
    # Let's fit the minimum again to cut off the lower 10%
    # Try to only fit the very center of the line
    cutoff = 0.2
    expected = (line_center, 0.05, cutoff, cutoff)
    min_sp = np.min(spec_line)
    mask = spec_line < min_sp + cutoff
    try:
        params, cov = curve_fit(_gauss_continuum, wave_line[mask], spec_line[mask], expected)
    
        lin_wv = np.linspace(np.min(wave_line[mask]), np.max(wave_line[mask]), 1000)
        min_flux = np.min(_gauss_continuum(lin_wv, *params))
        center = params[0]
    except RuntimeError:
        min_flux = np.min(spec_line)
    center = line_center
    

    # Amount of datapoints to skip from the bottom
    left_wave = wave_line[wave_line < center]
    left_spec = spec_line[wave_line < center]
    right_wave = wave_line[wave_line > center]
    right_spec = spec_line[wave_line > center]

    # make both array strictly increasing for the right part
    incr_mask = np.diff(right_spec) > 0
    if not incr_mask.all():
        max_true_idx = np.argmin(incr_mask)
        incr_mask[max_true_idx:] = False
    incr_mask = np.append(incr_mask, False)
    right_spec = right_spec[incr_mask]
    right_wave = right_wave[incr_mask]

    # And stricly increasing from right to left for the left
    incr_mask = np.diff(np.flip(left_spec)) > 0
    if not incr_mask.all():
        min_true_idx = np.argmax(incr_mask)
        # For the case that the first element is False and all others are True
        max_true_idx = np.argmin(incr_mask[min_true_idx:]) + min_true_idx
        if max_true_idx == min_true_idx:
            max_true_idx = -1
        incr_mask[max_true_idx:] = False
            
    incr_mask = np.append(incr_mask, False)
    left_spec = left_spec[np.flip(incr_mask)]
    left_wave = left_wave[np.flip(incr_mask)]


    interpolation="linear"
    if interpolation == "cubic_spline":
        left_cs = CubicSpline(np.flip(left_spec), np.flip(left_wave), extrapolate=True)
        right_cs = CubicSpline(right_spec, right_wave, extrapolate=True)
    elif interpolation == "linear":
        left_cs = interp1d(np.flip(left_spec), np.flip(left_wave), fill_value="extrapolate")
        right_cs = interp1d(right_spec, right_wave, fill_value="extrapolate")
    lin_sp = np.linspace(np.min(spec_line), np.max(spec_line), 75)
    # lin_sp = np.flip(left_wave)

    left = left_cs(lin_sp)
    right = right_cs(lin_sp)
    bisector_waves = (left + right) / 2
    bisector_flux = lin_sp

    # Now mask out outliers
    outlier_mask = np.abs(line_center - bisector_waves) <= outlier_clip

    bisector_waves[~outlier_mask] = np.nan
    bisector_flux[~outlier_mask] = np.nan
    
    mask = np.logical_and(bisector_flux >= min_flux +  0.03, bisector_flux < 0.9 * continuum)
    bisector_waves[~mask] = np.nan
    bisector_waves[~mask] = np.nan


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

def adjust_resolution(wave, spec, R=100000):
    """ Basic resolution adjustment using a convolution with a Gaussian kernel.
    
        :param np.array wave: Wavelength array in Angstrom
        :param np.array spec: Spectrum arry
        :param float R: resolution to smooth to
    """
    mid_px = int(len(wave)/2)
    center = wave[mid_px]
    
    sigma_inst = center / (2*np.sqrt(2*np.log(2)) * R)
    
    # convert that to pixel
    pixel_scale = wave[mid_px] - wave[mid_px - 1]
    sigma_px = sigma_inst / pixel_scale
    
    kernel = Gaussian1DKernel(stddev=sigma_px)
    sp_conv = convolve_fft(spec, kernel)
    
    return sp_conv

def adjust_resolution_per_pixel(wave, spec, R=100000):
    """ Improved resolution adjustment function working per pixel and taking care of jumps.
    
        :param np.array wave: Wavelength array in Angstrom
        :param np.array spec: Spectrum arry
        :param float R: resolution to smooth to
    """
    # Define the scale jumps present in PHOENIX
    scale_jumps = [0, 5000, 10000, 15000, 20000]
    pixel_scales_dict = {0: None,
                            5000: 0.006,
                            10000: 0.01,
                            15000: 0.03,
                            20000: None}
    scale_jumps = [sj for sj in scale_jumps if sj < wave[-1] + 5000 and sj > wave[0] - 5000]
    scale_jump_px = [(np.abs(wave-sj)).argmin() for sj in scale_jumps]
    
    last_idx = 0
    spec_conv = np.zeros_like(wave)
    
    
    # Calculate the local smoothing kernel for each wavelength point
    sigma_inst = wave / (2*np.sqrt(2*np.log(2)) * R)
    for jump_interval, idx in enumerate(scale_jump_px):
        # In case you are at the right border, make sure to include the last point
        if idx == len(wave) - 1:
            idx += 1

        # # Make arrays that run exactly to the jump but do not include it
        if jump_interval == 0:
            continue
        wave_local = wave[last_idx:idx]
        spec_local = spec[last_idx:idx]
        # wave_start = wave_local[0]
        # wave_stop = wave_local[-1]
        pixel_scale_local = pixel_scales_dict[scale_jumps[jump_interval]]
        pixel_scale_local = wave_local[50] - wave_local[49]
        
        sigma_px_local = sigma_inst[last_idx:idx] / pixel_scale_local
        
        # Let's first calculate the largest width in the current segment
        max_dpx = np.max(sigma_px_local)
        
        # And define 25 times as a overhead
        px_over = int(np.ceil(max_dpx*7))
        
        spec_conv_local = np.zeros_like(wave_local)
        spec_loops = []
        
        rowmask = np.zeros(len(wave_local), dtype=bool)
        for i in range(0, len(wave_local)):
            # Check if you overlap into the last interval on the left
            if (i - px_over) < 0:
                # Check if you're in the first interval, i.e. you cannot interpolate
                # Set the left side to 0 then
                if jump_interval == 1:
                    spec_loop = np.zeros(np.abs(i-px_over))
                    spec_loop = np.append(spec_loop, spec_local[:i+px_over+1])
                else:
                    di = i - px_over
                    # We have to interpolate into the last range
                    prev_interval_wave = wave[scale_jump_px[jump_interval-2]:scale_jump_px[jump_interval-1]]
                    prev_interval_spec = spec[scale_jump_px[jump_interval-2]:scale_jump_px[jump_interval-1]]
                    
                    lin_wave = np.linspace(wave_local[0] - np.abs(di)*pixel_scale_local,
                                            wave_local[0],
                                            np.abs(di))
                    interp_spec = np.interp(lin_wave, prev_interval_wave, prev_interval_spec)
                    # Now you have the interpolated spectrum in the new sampling range
                    # Now stitch together
                    spec_loop = interp_spec
                    spec_loop = np.append(spec_loop, spec_local[:i+px_over+1])
            # Check if you overlap on the right side        
            elif (i + px_over) >= len(wave_local):
                # Check if you're in the last interval, i.e. you cannot interpolate
                # Set the right side to 0 then
                if not len(scale_jump_px) > jump_interval + 1:
                    spec_loop = spec_local[i - px_over:]
                    spec_loop = np.append(spec_loop, np.zeros(i + px_over - len(wave_local) + 1))
                else:
                    di = i + px_over - len(wave_local) + 1
                    # We have to interpolate into the last range
                    next_interval_wave = wave[scale_jump_px[jump_interval]:scale_jump_px[jump_interval+1]]
                    next_interval_spec = spec[scale_jump_px[jump_interval]:scale_jump_px[jump_interval+1]]
                    
                    lin_wave = np.linspace(wave_local[-1] + pixel_scale_local,
                                            wave_local[-1] + np.abs(di)*pixel_scale_local, np.abs(di))
                    interp_spec = np.interp(lin_wave, next_interval_wave, next_interval_spec)
                    # Now you have the interpolated spectrum in the new sampling range
                    # Now stitch together
                    spec_loop = spec_local[i - px_over:]
                    spec_loop = np.append(spec_loop, interp_spec)
            else:
                px_start = i - px_over
                px_stop = i + px_over + 1
                spec_loop = spec_local[px_start : px_stop]
            
            
            rowmask[i] = True
            
            spec_loop = np.array(spec_loop)
            spec_loops.append(spec_loop)
        
        
        # Compute the kernels
        lin_px = np.linspace(-px_over, px_over, 2*px_over+1)
        lin_px = np.array([lin_px for i in range(len(wave_local))])
        sigma_px_local = np.array([sigma_px_local for i in range(2*px_over+1)]).T
        
        kernels = gaussian(lin_px, 0., sigma_px_local)
        
        spec_loops = np.array(spec_loops)
        kernels = kernels[rowmask,:]

        spec_conv_local[rowmask] = np.sum(spec_loops * kernels, axis=1)
        del kernels
        del spec_loops
        spec_conv[last_idx:idx] = spec_conv_local
        last_idx = idx  
        
    return spec_conv 
        
    


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

def get_ref_spectra(T_grid, logg, feh, wavelength_range=(3000, 7000),
                    spec_intensity=False, change_bis=False):
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
            # Create a new dictionary with the spectra for each mu
            mu_dict = {}
            wave, spec, ref_headers[T] = phoenix_spectrum(
                Teff=float(T), logg=logg, feh=feh,
                wavelength_range=wavelength_range)
            # All waves are the same, so just return the last one
            # if change_bis:
            #     print("Fit and Remove the PHOENIX bisectors")
            #     spec_corr, _, _, _, _, _ = remove_phoenix_bisector(wave, spec, T, logg, feh)
            #     spec = spec_corr
                
            #     # Now let's add in the bisectors
            #     # available_mus = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
            #     # TODO remove
            #     # available_mus = [1.0]
                
            #     # bis_polynomial_dict = Zhao_bis_polynomials()
            #     # Test out the Pollux bisectors
            #     bis_polynomial_dict = simple_alpha_boo_CB_model()
                
            #     for mu in bis_polynomial_dict.keys():
            #         spec_add, _, _, _, _ = add_bisector(wave, 
            #                                             copy.deepcopy(spec_corr), 
            #                                             bis_polynomial_dict[mu],
            #                                             T, 
            #                                             logg, 
            #                                             feh, 
            #                                             debug_plot=False,
            #                                             mu=mu)
            #         # TODO remove
            #         # spec_add = spec_corr
            #         mu_dict[mu] = spec_add
            # else:
            mu_dict[1.0] = spec
            ref_spectra[T] = mu_dict
                
            # ref_spectra[T] = spec

        return wave, ref_spectra, ref_headers
    else:
        for T in T_unique:
            wave, ref_spectra[T], mu, ref_headers[T] = phoenix_spec_intensity(
                Teff=float(T), logg=logg, feh=feh,
                wavelength_range=wavelength_range)
            # All waves are the same, also all mu should be the same
            # So just return the last one
        return wave, ref_spectra, ref_headers, mu


def plot_individual_fit(ax, line, wv, sp, bis_wave, bis, left_wv, left_sp, right_wv, right_sp, gauss_params):
    """ Add an individual fitted line to existing ax."""
    ax.plot(wv, sp, marker="o", markersize=6)
    ax.plot(left_wv, left_sp, linestyle="None", marker="o", markersize=5, color="green")
    ax.plot(right_wv, right_sp, linestyle="None", marker="o", markersize=5, color="red")

    ax.plot(bis_wave, bis, marker="o", markersize=5)
    
    # ax.plot(wv, _gauss_continuum(wv, *gauss_params), marker="None", color="pink")
    ylims = ax.get_ylim()
    ax.vlines(line, ylims[0], ylims[1], color="tab:red", linestyle="dashed")
    ax.set_ylim(ylims)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(rf"FeI {line}$\AA$")
    ax.ticklabel_format(useOffset=False)
    
    return ax


def normalize_phoenix_spectrum_Rassine(wave, spec, Teff, logg, feh, run=False, debug_plot=False):
    """ Normalize a PHOENIX Spectrum using Rassine.
    
        All results are linearly interpolated back onto the original wavelength grid.
    
        :param np.array wave_vac: Wavelength in Angstrom
        :param np.array spec: Unnormalized Spectrum
        
        :returns: np.array wave_norm, normalized wavelength
        :retunrs np.array continuum, The fitted continuum
    """
    # Now create the Rassine fit
    if run:
        spec_df = pd.DataFrame({'wave':wave,'flux':spec})
        pickle_path = "/home/dspaeth/pyoscillot/pyoscillot/phoenix_spec_rassine.p"
        spec_df.to_pickle(pickle_path)
        subprocess.run(["python3",
                    "/home/dspaeth/Rassine_public/Rassine.py",
                    pickle_path],
                    timeout=60)

        rassine_df = pd.read_pickle("/home/dspaeth/pyoscillot/pyoscillot/RASSINE_phoenix_spec_rassine.p")
    else:
        rassine_df = Rassine_outputs(Teff, logg, feh)
    continuum = rassine_df["output"]["continuum_linear"]
    
    wave_rassine = rassine_df["wave"]
    print(f"Wavelength Range from Rassine: {wave_rassine[0]}:{wave_rassine[-1]}")
    if not run:
        assert wave[0] > wave_rassine[0], f"Your wavelength array starts below the Rassine array limit {wave[0]} < {wave_rassine[0]}" 
        assert wave[-1] < wave_rassine[-1], f"Your wavelength array ends above the Rassine array limit {wave[-1]} > {wave_rassine[-1]}" 
    
    # Normalize
    continuum_interp = np.interp(wave, wave_rassine, continuum)
    
    spec_norm = spec / continuum_interp
    
    if debug_plot:
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = cfg.conf_dict["datapath"] / "plots/phoenix_bisectors/debug"
        savename = f"{Teff}K_{logg}_{feh}_norm.png"
        out_file = out_root / savename
        if not out_file.is_file(): 
            fig, ax = plt.subplots(1, figsize=(6.35, 3.5))
            ax.plot(wave, spec, lw=0.25, color="tab:blue")
            ax.plot(wave, continuum_interp, lw=0.5, color="tab:red")
            ax.set_xlabel(r"Wavelength [$\AA$]")
            ax.set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
            print(f"Save debug plot to {out_root}/{savename}")
            fig.set_tight_layout(True)
            plt.savefig(out_file, dpi=600)
            plt.close()
    
    return wave, spec_norm, continuum_interp


def normalize_phoenix_spectrum_precomputed(wave, spec, Teff, logg, feh, 
                                           limb_dark_continuum=None, 
                                           debug_plot=False):
    """ Normalize a PHOENIX Spectrum using a precomupted continuum.
    
        All results are linearly interpolated back onto the original wavelength grid.
    
        :param np.array wave_vac: Wavelength in Angstrom
        :param np.array spec: Unnormalized Spectrum
        :param int Teff: Effective Temperature (must be one of PHOENIX grid)
        :param float logg: log(g) (must be one of PHOENIX grid)
        :param float feh: [Fe/H] (must be one of PHOENIX grid)
        :param np.array limb_dark_continuum: (Optional) Additional Limb Darkening continuum to remove
        
        :returns: np.array wave_norm, normalized wavelength
        :returns np.array continuum, The saved continuum
    """
    # Load the precomputed continuum (this will take a precomputed PHOENIX continuum)
    # Without knowing anything about the limb darkening correction
    wave_cont, cont = continuum(Teff, logg, feh, wavelength_range=(wave[0], wave[-1]))
    
    # Check that wavelengths are correct
    assert wave[0] >= wave_cont[0], f"Your wavelength array starts below the Rassine array limit {wave[0]} < {wave_cont[0]}" 
    assert wave[-1] <= wave_cont[-1], f"Your wavelength array ends above the Rassine array limit {wave[-1]} > {wave_cont[-1]}" 
    
    # If we have removed the averaged limb darkening we must also adjust the continuum here
    if limb_dark_continuum is not None:
        print("Use the Limb Darkening Continuum during Continuum correction")
        cont = cont / limb_dark_continuum
    
    # Normalize
    spec_norm = spec / cont
    
        
    
    if debug_plot:
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = cfg.conf_dict["datapath"] / "plots/continuum"
            if not out_root.is_dir():
                out_root.mkdir()
        savename = f"{Teff}K_{logg}_{feh}_norm.png"
        out_file = out_root / savename
        if not out_file.is_file(): 
            fig, ax = plt.subplots(1, figsize=(6.35, 3.5))
            ax.plot(wave, spec, lw=0.25, color="tab:blue")
            ax.plot(wave, cont, lw=0.5, color="tab:red")
            ax.set_xlabel(r"Wavelength [$\AA$]")
            ax.set_ylabel("Flux [arb. units]")
            print(f"Save debug plot to {out_root}/{savename}")
            fig.set_tight_layout(True)
            plt.savefig(out_file, dpi=600)
            plt.close()
    
    return wave, spec_norm, cont

def get_phoenix_bisector(wave, spec, Teff, logg, FeH, 
                         limb_dark_continuum=None,
                         debug_plot=False, bis_plot=False, ax=None):
    """ Get the mean PHOENIX bisector as described by Zhao & Dumusque (2023) Fig. A.1
    
        :returns: numpy.Polynomial fit results to easily apply to every line depth
    """
    Fe_lines = [5250.2084, 5250.6453, 5434.5232, 6173.3344, 6301.5008]

    
    wave_rassine, spec, _ = normalize_phoenix_spectrum_precomputed(wave, 
                                                                   spec, 
                                                                   Teff, 
                                                                   logg, 
                                                                   FeH, 
                                                                   debug_plot=False, 
                                                                   limb_dark_continuum=limb_dark_continuum)
    wave_air = wave_rassine / (1.0 + 2.735182E-4 + 131.4182 / wave**2 + 2.76249E8 / wave**4)

    if debug_plot:
        fig, debug_ax = plt.subplots(2,3, figsize=(6.35, 3.5))
    # Compute the individual bisectors per line
    bis_vs = []
    biss = []
    for idx, line in enumerate(Fe_lines):
        # Choose an interval around the theoretical line center to look at
        interval = 0.25
        mask = np.logical_and(wave_air >= line - interval, wave_air <= line + interval)

        wv = wave_air[mask]
        sp = spec[mask]
    
        # Fit the width and center for an inital guess
        expected = (line, 0.05, 0.9, 1.0)
        try:
            params, cov = curve_fit(_gauss_continuum, wv, sp, expected)
            width = params[1]
            continuum = params[-1]
        except:
            width = 0.05
            continuum = 1.0
    
    
        try:
            bis_wave, bis, left_wv, left_sp, right_wv, right_sp = bisector_on_line(wv, 
                                                                                   sp, 
                                                                                   line,
                                                                                   width=width,
                                                                                   outlier_clip=0.005,
                                                                                   continuum=continuum)
        
            # Convert to velocities
            bis_v = (bis_wave - line) / bis_wave * 3e8
        
            # Make debug plot
            if debug_plot:
                plot_individual_fit(debug_ax.flatten()[idx], line, wv, sp, bis_wave, bis, left_wv, left_sp, right_wv, right_sp, params)
        
        except Exception as e:
            raise e
    
        bis_vs.append(bis_v)
        biss.append(bis)
        
    
    if debug_plot:
        fig.delaxes(debug_ax[-1,-1])
        fig.set_tight_layout(True)
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = cfg.conf_dict["datapath"] / "plots/phoenix_bisectors/debug"
        savename = f"{Teff}K_{logg}_{FeH}_Fe_lines.png"
        print(f"Save debug plot to {out_root}/{savename}")
        plt.savefig(out_root / savename, dpi=600)
        plt.close()

    bis_vs = np.array(bis_vs)
    biss = np.array(biss)

    
    # Now let's compute the mean bisector
    avg_bis = np.linspace(0.2, 0.8, 13)
    step_bis = avg_bis[1] - avg_bis[0]
    avg_v = np.zeros_like(avg_bis)
    for idx, abis in enumerate(avg_bis):
        mask = np.logical_and(biss.flatten() >= abis - step_bis, biss.flatten() < abis + step_bis)
        avg_v[idx] = np.nanmean(bis_vs.flatten()[mask])

    # Fit a second order polynomial
    mean_v =  np.nanmean(avg_v)
    avg_v -= mean_v
    poly_fit = Polynomial.fit(avg_bis, avg_v, 2)
    lin_bis = np.linspace(0.0, 1.0, 100)
    poly_v = poly_fit(lin_bis)

    # Now we want to have everything in the end centered around the fitted poly bisector
    # mean_v = np.mean(poly_v)
    bis_vs -= mean_v
    # avg_v -= mean_v
    # poly_v -= mean_v
    
    if bis_plot:
        if ax is None:
            _ax = None
            fig, ax = plt.subplots(1, figsize=(6.35, 3.5), dpi=600)
        colors = ["green", "cyan", "purple", "orange", "yellow"]
        for bis_v, bis, color, line in zip(bis_vs, biss, colors, Fe_lines): 
            ax.plot(bis_v, bis, color=color, marker="o", markersize=5, label=rf"FeI {line}$\AA$")
        ax.plot(avg_v, avg_bis, color="red", marker="o", markersize=7, linestyle="None")

        ax.plot(poly_v, lin_bis, color="blue", linewidth=8, alpha=0.7)
        
        ax.set_ylim(0, 1)
        ax.set_xlim(-200, 200)
        ax.set_xlabel("Velocity [m/s]")
        ax.set_ylabel("Normalized Flux")

        fig.set_tight_layout(True)
        ax.legend()
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = cfg.conf_dict["datapath"] / "plots/phoenix_bisectors"
        savename = f"{Teff}K_{logg}_{FeH}_bis_fit.png"
        plt.savefig(f"{out_root}/{savename}", dpi=600)
        plt.close()
    
    return poly_fit

def remove_phoenix_bisector(wave, spec, Teff, logg, FeH,
                            limb_dark_continuum=None,
                            debug_plot=False, line=5728.65,):
    """ Fit and Remove the phoenix bisector.
    
        :param np.array limb_dark_continuum: (Optional) Additional Limb Darkening continuum to remove
    """
    poly_fit = get_phoenix_bisector(wave, 
                                    spec, 
                                    Teff, 
                                    logg, 
                                    FeH, 
                                    debug_plot=False,
                                    bis_plot=True, 
                                    limb_dark_continuum=limb_dark_continuum)
    
    # First normalize the spectrum
    _, spec_norm, continuum = normalize_phoenix_spectrum_precomputed(wave, 
                                                                     spec, 
                                                                     Teff, 
                                                                     logg, 
                                                                     FeH, 
                                                                     limb_dark_continuum=limb_dark_continuum)
    
    delta_v = poly_fit(spec_norm)
    delta_wave = delta_relativistic_doppler(wave, v=delta_v)
    wave_corr = wave - delta_wave
    
    # Interpolate back on original wavelength grid?
    spec_corr = np.interp(wave, wave_corr, spec)
    
    # Also make a normalized version for debugging
    spec_corr_norm = np.interp(wave, wave_corr, spec_norm)
    # FOR DEBUGGING
    if debug_plot:
        interval = 0.25
        mask = np.logical_and(wave >= line - interval, wave <= line + interval)
        
        fig, ax = plt.subplots(1, 2, figsize=(6.35, 3.5))
        ax[0].plot(wave[mask], spec_norm[mask], color="tab:blue",marker="o", label="Original PHOENIX spectrum")
        ax[0].plot(wave[mask], spec_corr_norm[mask], color="tab:red",marker="o", label="Removed Bisector")
        ax[0].legend()
        ax[0].set_xlabel(r"Wavelength [$\AA$]")
        ax[0].set_ylabel("Normalized Flux")
        ax[0].set_title(rf"{line}$\AA$")
        ax[0].ticklabel_format(useOffset=False)
        
        # Fit the bisectors for both lines
        mean_bis_v = None
        for sp, color in zip((spec_norm[mask], spec_corr_norm[mask]), ("tab:blue", "tab:red")):
            # Fit the width and center for an inital guess
            expected = (line, 0.05, 0.9, 1.0)
            try:
                params, cov = curve_fit(_gauss_continuum, wave[mask], sp, expected)
                width = params[1]
                continuum = params[-1]

            except:
                width = 0.05
                continuum = 1.0
                
        
        
            try:
                bis_wave, bis, left_wv, left_sp, right_wv, right_sp = bisector_on_line(wave[mask], 
                                                                                    sp, 
                                                                                    line,
                                                                                    width=width,
                                                                                    outlier_clip=0.1,
                                                                                    continuum=continuum)
            
                # Convert to velocities
                bis_v = (bis_wave - line) / bis_wave * 3e8
                
                if mean_bis_v is None:
                    mean_bis_v = np.nanmean(bis_v)
                ax[0].plot(bis_wave, bis, color=color, marker="o")
                ax[1].plot(bis_v - mean_bis_v, bis, color=color, marker="o")
            except Exception as e:
                pass
        
        
        
        # ax[1].plot(wave[mask], delta_v[mask])
        ax[0].legend(loc="lower left")
        
        # Add the BIS polynomial
        lin_spec = np.linspace(0, 1, 100)
        ax[1].plot(poly_fit(lin_spec), lin_spec, color="black", alpha=0.7, label=f"Fitted Mean Bisector")
        ax[1].legend(loc="lower left")
        
        
        
        # ax[1].plot(wave[mask], delta_v[mask])
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = cfg.conf_dict["datapath"] / "plots/phoenix_bisectors"
        savename = f"{Teff}K_{logg}_{FeH}_bis_removed.png"
        fig.set_tight_layout(True)
        print(f"Save to {out_root}/{savename}")
        plt.savefig(f"{out_root}/{savename}", dpi=600)
        plt.close()
    
    return spec_corr, spec_corr_norm, spec_norm, poly_fit, delta_v, delta_wave

def add_bisector(wave, spec, bis_polynomial, Teff, logg, FeH,
                 mu=None, limb_dark_continuum=None,
                 debug_plot=False, line=5728.65):
    """ Add in a bisector to the data.
    
        Ideally the wave and spec should be cleaned of any previous bisectors.
        
        :param np.array wave: The Wavelength array
        :param np.array spec: The Spectrum Flux array (not normalized)
        :param np.polynomial.Polynomial: A numpy Polynomial describing the Bisector 
        :param np.array limb_dark_continuum: (Optional) Additional Limb Darkening continuum to remove
        
        
        
        :returns: Corrected Spectrum, normalized corrected spectrum
    """
    print(f"ADD IN BIS")
    _, spec_norm, continuum = normalize_phoenix_spectrum_precomputed(wave, 
                                                                     spec, 
                                                                     Teff, 
                                                                     logg, 
                                                                     FeH, 
                                                                     limb_dark_continuum=limb_dark_continuum)
    
    delta_v = bis_polynomial(spec_norm)
    # delta_v *= 5
    delta_wave = delta_relativistic_doppler(wave, v=delta_v)
    wave_corr = wave + delta_wave
    
    # Interpolate back on original wavelength grid?
    spec_corr = np.interp(wave, wave_corr, spec)
    
    # Also make a normalized version for debugging
    spec_corr_norm = np.interp(wave, wave_corr, spec_norm)
    
    # FOR DEBUGGING
    if debug_plot:
        interval = 0.25
        mask = np.logical_and(wave >= line - interval, wave <= line + interval)
        
        fig, ax = plt.subplots(1,2, figsize=(6.35, 3.5))
        ax[0].plot(wave[mask], spec_norm[mask], color="tab:blue", marker="o", label="Bisector removed PHOENIX spectrum")
        # ax[0].plot(wave_corr[mask], spec[mask], color="tab:red", marker="o", label="Added Bisector")
        ax[0].plot(wave[mask], spec_corr_norm[mask], color="tab:red", marker="o", label="Added Bisector")
        ax[0].ticklabel_format(useOffset=False)
        # ax[1].plot(delta_v[mask], spec_norm[mask])
        
        
        # Fit the bisectors for both lines
        mean_bis_v = None
        for sp, color in zip((spec_norm[mask], spec_corr_norm[mask]), ("tab:blue", "tab:red")):
            # Fit the width and center for an inital guess
            expected = (line, 0.05, 0.9, 1.0)
            try:
                params, cov = curve_fit(_gauss_continuum, wave[mask], sp, expected)
                width = params[1]
                continuum = params[-1]

            except:
                width = 0.05
                continuum = 1.0
                
        
        
            try:
                bis_wave, bis, left_wv, left_sp, right_wv, right_sp = bisector_on_line(wave[mask], 
                                                                                    sp, 
                                                                                    line,
                                                                                    width=width,
                                                                                    outlier_clip=0.1,
                                                                                    continuum=continuum)
            
                # Convert to velocities
                bis_v = (bis_wave - line) / bis_wave * 3e8
                
                if mean_bis_v is None:
                    mean_bis_v = np.nanmean(bis_v)
                ax[0].plot(bis_wave, bis, color=color, marker="o")
                ax[1].plot(bis_v - mean_bis_v, bis, color=color, marker="o")
            except Exception as e:
                pass
        
        
        
        
        # ax[1].plot(wave[mask], delta_v[mask])
        ax[0].legend(loc="lower left")
        
        ax[0].set_xlabel(r"Wavelength [$\AA$]")
        ax[0].set_ylabel("Normalized Flux")
        ax[0].set_title(rf"{line}$\AA$, $\mu$={mu}")
        ax[1].set_xlabel("Velocity [m/s]")
        ax[1].set_ylabel("Normalized Flux")
        
        # Add the BIS polynomial
        lin_spec = np.linspace(0, 1, 100)
        ax[1].plot(bis_polynomial(lin_spec), lin_spec, color="black", alpha=0.7, label=f"Convective Blueshift Model (µ={mu})")
        ax[1].legend(loc="lower left")
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = cfg.conf_dict["datapath"] / "plots/phoenix_bisectors"
        savename = f"{Teff}K_{logg}_{FeH}_mu{mu}_bis_added.png"
        fig.set_tight_layout(True)
        plt.savefig(f"{out_root}/{savename}", dpi=600)
        plt.close()
        
    return spec_corr, spec_corr_norm, spec_norm, delta_v, delta_wave

def calc_limb_dark_intensity(wave, mu):
    alpha = np.zeros_like(wave)
    mask_UV = np.logical_and(wave >= 3033, wave <= 3570)
    
    u = 1
    real_limit = 4160
    test_limit = 3570
    mask_VIS = np.logical_and(wave >= test_limit, wave <= 10990)
    
    # wave has to be given in µm but it currently in Angstroms
    # hence the factor 1e4
    alpha[mask_UV] = -0.507 + 0.441 * (1 / (wave[mask_UV] / 1e4))
    alpha[mask_VIS] = -0.023 + 0.292 * (1 / (wave[mask_VIS] / 1e4)) 
    intensity = 1 - u * (1 - mu**alpha)
    # intensity is now an array with the dimension of wave
    # i.e. going along the wavelength giving a multiplication factor
    # the intensity array should be propely normalized
    
    return intensity

def add_limb_darkening(wave, spec, mu):
    """ Add the limb darkening law based on Hestroffer & Magnan 1998.
    
        Intended to be used for a single mu angle.
    
        :param np.array wave: Wavelength array
        :param np.array spec: Spectrum array
        :param float mu: Mu angle within [0, 1]
    """
    spec = copy.deepcopy(spec)
    
    intensity = calc_limb_dark_intensity(wave, mu)
    
    spec_limb = spec * intensity
    
    return intensity, spec_limb

def calc_mean_limb_dark(wave, mu_array, load_precalc=True, N=150):
    """ Calculate a mean limb darkening continuum for a mu_array.
    
        :param np.array wave: A wavelength array in Angstrom
        :param np.array mu_array: Usually a 2D array containing mu angles for the star
        :param bool load_precalc: If True, a precalculated LD will be loaded if existing
        
        :returns: Averaged limb darkening continuum with the same shape as wave
    """
    wave_start = np.min(wave)
    wave_stop = np.max(wave)
    outname = f"mean_LD_N{N}_wave{wave_start}_{wave_stop}.npy"
    outroot = cfg.conf_dict["datapath"] / "mean_limb_darkening" 
    outfile = outroot / outname
    run = True
    if load_precalc:
        if outfile.is_file():
            print(f"Load precalculated mean Limb Darkening from {outfile}")
            mean_ld_intensity = np.load(outfile)
            run = False
        else:
            wave_start = 3500.004
            wave_stop = 17499.99
            larger_outname = f"mean_LD_N{N}_wave{wave_start}_{wave_stop}.npy"
            outfile = outroot / larger_outname
            if outfile.is_file():
                mean_ld_intensity = np.load(outfile)
                wave_file = outroot / f"wave_LD_N{N}_wave{wave_start}_{wave_stop}.npy"
                wave_saved = np.load(wave_file)
                mask = np.logical_and(wave_saved >= np.min(wave), 
                                      wave_saved <= np.max(wave))
                mean_ld_intensity = mean_ld_intensity[mask]
                run = False
        
    if run:
        mus = copy.deepcopy(mu_array)
        mus = mus.flatten()
        mus = mus[~np.isnan(mus)]
        ld_intensities = np.zeros_like(wave)
        
        unique_mus, counts = np.unique(mus, return_counts=True)
        
        
        for idx, (mu, count) in enumerate(zip(unique_mus, counts)):
            print(idx, len(unique_mus))
            print(count)
            ld_intensity = calc_limb_dark_intensity(wave, mu)
            ld_intensities += (ld_intensity * count)
            
        mean_ld_intensity = ld_intensities / len(mus)
        np.save(outfile, mean_ld_intensity)
        
        outname = f"wave_LD_N{N}_wave{wave_start}_{wave_stop}.npy"
        outfile = cfg.conf_dict["datapath"] / "mean_limb_darkening" / outname
        
        np.save(outfile, wave)
    return mean_ld_intensity
        
    
    

# def add_isotropic_convective_broadening(wave, spec, v_macro, wave_dependent=True, debug_plot=False, wave_step=0.5, per_pixel=False):
#     """ Add the effect of macroturbulence, i.e. convective broadening, via convolution.
    
#         This function assumes an isotropic broadening term, i.e. a constant
#         convolution kernel across the stellar disk.
        
#         :param np.array wave: Wavelength array in Angstrom
#         :param np.array spec: Spectrum array 
#         :param float v_macro: Macroturbulent velocity (eta) in m/s
#     """
#     if not wave_dependent:
#         center_idx = int(len(wave) / 2)
#         delta_wave = delta_relativistic_doppler(wave[center_idx], v_macro)
#         # this corresponds to the FWHM of the Gaussian kernel, so we need the conversion factor
#         delta_wave /= 2*np.sqrt(2*np.log(2))
#         pixel_scale = wave[center_idx] - wave[center_idx - 1]
        
#         #TODO: check if pixel scale is constant
        
        
        
#         sigma_px = delta_wave / pixel_scale
#         kernel = Gaussian1DKernel(stddev=sigma_px)
#         spec_conv = convolve_fft(spec, kernel)
#     else:
#         center_idx = int(len(wave) / 2)
#         delta_wave = delta_relativistic_doppler(wave, v_macro)
#         delta_wave /= 2*np.sqrt(2*np.log(2))
#         pixel_scale = wave[1:] - wave[:-1]
#         pixel_scale = np.insert(pixel_scale, 0, pixel_scale[0])
        
#         # sigma_px = delta_wave / pixel_scale
#         # The pixel scale is constant but has jumps at 5000, 10000 and 15000 A
#         scale_jumps = [0, 5000, 10000, 15000, 20000]
#         pixel_scales = [None, 0.006, 0.01, 0.03, None]
#         scale_jumps = [sj for sj in scale_jumps if sj < wave[-1] + 5000]
#         scale_jump_px = [(np.abs(wave-sj)).argmin() for sj in scale_jumps]
        
#         last_idx = 0
        
#         spec_conv = np.zeros_like(wave)
#         for jump_interval, idx in enumerate(scale_jump_px):
#             # Make arrays that run exactly to the jump but do not include it
#             if jump_interval == 0:
#                 continue
#             wave_local = wave[last_idx:idx]
#             spec_local = spec[last_idx:idx]
            
#             if not len(wave_local):
#                 continue
#             # pixel_scale_local = pixel_scale[last_idx:idx]
#             pixel_scale_local = pixel_scales[jump_interval]
#             delta_wave_local = delta_wave[last_idx:idx]
#             sigma_px_local = delta_wave_local / pixel_scale_local
            
            
#             # Let's first calculate the largest width in the current segment
#             max_dw = np.max(delta_wave_local)
#             # Convert it to pixel
#             max_dpx = max_dw / pixel_scale_local
#             # And define 30 times as a overhead
#             if not per_pixel:
#                 px_step = int(wave_step / pixel_scale_local / 2) 
#                 px_over = int(np.ceil(max_dpx*20))
#             else:
#                 px_step = 0
#                 px_over = int(np.ceil(max_dpx*20))
            
#             spec_conv_local = np.zeros_like(wave_local)
            
#             for i in range(px_step, len(wave_local), max(px_step,1)):
#                 # print(f"\r{i}, {len(wave_local)}", end="")
#                 di_high = 0
#                 if (i - px_step - px_over) < 0:
#                     if jump_interval == 1:
#                         # Cannot interpolate
#                         continue
#                     di = i - px_step - px_over
#                     # We have to interpolate into the last range
#                     prev_interval_wave = wave[scale_jump_px[jump_interval-2]:scale_jump_px[jump_interval-1]]
#                     prev_interval_spec = spec[scale_jump_px[jump_interval-2]:scale_jump_px[jump_interval-1]]
                    
#                     lin_wave = np.linspace(wave_local[0] - np.abs(di)*pixel_scale_local,
#                                             wave_local[0],
#                                             np.abs(di))
#                     interp_spec = np.interp(lin_wave, prev_interval_wave, prev_interval_spec)
#                     # Now you have the interpolated spectrum in the new sampling range
#                     # Now stitch together
#                     spec_loop = interp_spec
#                     spec_loop = np.append(spec_loop, spec_local[:i+px_step+px_over+1])
#                 elif (i + px_step + px_over) > len(wave_local):
#                     if not len(scale_jump_px) > jump_interval + 1:
#                         # Cannot interpolate
#                         continue
                    
                        
#                     di = i + px_step + px_over - len(wave_local) + 1
#                     # We have to interpolate into the last range
#                     next_interval_wave = wave[scale_jump_px[jump_interval]:scale_jump_px[jump_interval+1]]
#                     next_interval_spec = spec[scale_jump_px[jump_interval]:scale_jump_px[jump_interval+1]]
                    
#                     lin_wave = np.linspace(wave_local[-1] + pixel_scale_local,
#                                             wave_local[-1] + np.abs(di)*pixel_scale_local, np.abs(di))
#                     interp_spec = np.interp(lin_wave, next_interval_wave, next_interval_spec)
#                     # Now you have the interpolated spectrum in the new sampling range
#                     # Now stitch together
#                     spec_loop = spec_local[i - px_step - px_over:]
#                     spec_loop = np.append(spec_loop, interp_spec)
#                     di_high = di - px_over
                    
#                 else:
#                     spec_loop = spec_local[i - (px_step+px_over):i + (px_step+px_over) + 1]
                
                
#                 kernel = Gaussian1DKernel(stddev=sigma_px_local[i])
#                 spec_conv_loop = convolve_fft(spec_loop, kernel)
                
#                 if di_high > 0:
#                     spec_conv_local[i-px_step:i+px_step+1] = spec_conv_loop[px_over:px_over+2*px_step+1-di_high]
#                 else:
#                     spec_conv_local[i-px_step:i+px_step+1] = spec_conv_loop[px_over:px_over+2*px_step+1]
                    
                    

            
#             spec_conv[last_idx:idx] = spec_conv_local
#             last_idx = idx      
    
#     if debug_plot:
#         if cfg.debug_dir is not None:
#             out_root = cfg.debug_dir
#         else:
#             out_root = Path("/home/dspaeth/pyoscillot/data/plots/macroturbulence/")
#         savename = f"macroturbulence.png"
#         outfile = out_root / savename
#         # Only save one debug plot (otherwise you would have that for every cell)
#         if not outfile.is_file():    
#             fig, ax = plt.subplots(1, figsize=(6.35, 3.5))
#             ax.plot(wave, spec, label="Simulated Spectrum")
#             ax.plot(wave, spec_conv, label=f"Broadend Spectrum by v_macro={v_macro}m/s")
#             ax.set_xlim(wave[center_idx]-5, wave[center_idx]+5)
#             ax.set_xlabel(r"Wavelength $[\AA]$")
#             ax.set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
#             ax.legend()
#             fig.set_tight_layout(True)
#             plt.savefig(f"{out_root}/{savename}", dpi=600)
#             plt.close()
        
        
    
#     return spec_conv


def add_isotropic_convective_broadening(wave, spec, v_macro, wave_dependent=True, debug_plot=False, wave_step=0.5, per_pixel=True, convolution=False, old=False):
    """ Add the effect of macroturbulence, i.e. convective broadening, via convolution.
    
        This function assumes an isotropic broadening term, i.e. a constant
        convolution kernel across the stellar disk.
        
        :param np.array wave: Wavelength array in Angstrom
        :param np.array spec: Spectrum array 
        :param float v_macro: Macroturbulent velocity (eta) in m/s
    """
    print(f"Add isotropic macroturbulence, wave_dependent={wave_dependent}, per_pixel={per_pixel}, convolution={convolution}")
    if not wave_dependent:
        center_idx = int(len(wave) / 2)
        delta_wave = delta_relativistic_doppler(wave[center_idx], v_macro)
        # this corresponds to the FWHM of the Gaussian kernel, so we need the conversion factor
        delta_wave /= 2*np.sqrt(2*np.log(2))
        pixel_scale = wave[center_idx] - wave[center_idx - 1]
        
        sigma_px = delta_wave / pixel_scale
        kernel = Gaussian1DKernel(stddev=sigma_px)
        spec_conv = convolve(spec, kernel)
    else:
        center_idx = int(len(wave) / 2)
        delta_wave = delta_relativistic_doppler(wave, v_macro)
        delta_wave /= 2*np.sqrt(2*np.log(2))
        
        # The pixel scale is constant but has jumps at 5000, 10000 and 15000 A
        scale_jumps = [0, 5000, 10000, 15000, 20000]
        pixel_scales_dict = {0: None,
                             5000: 0.006,
                             10000: 0.01,
                             15000: 0.03,
                             20000: None}
        scale_jumps = [sj for sj in scale_jumps if sj < wave[-1] + 5000 and sj > wave[0] - 5000]
        
        scale_jump_px = [(np.abs(wave-sj)).argmin() for sj in scale_jumps]
        
        last_idx = 0
        spec_conv = np.zeros_like(wave)
        for jump_interval, idx in enumerate(scale_jump_px):
            # Make arrays that run exactly to the jump but do not include it
            if jump_interval == 0:
                continue
            wave_local = wave[last_idx:idx]
            spec_local = spec[last_idx:idx]
            wave_start=wave_local[0]
            wave_stop=wave_local[-1]
            pixel_scale_local = pixel_scales_dict[scale_jumps[jump_interval]]
        
            
            delta_wave_local = delta_wave[last_idx:idx]
            sigma_px_local = delta_wave_local / pixel_scale_local
            
            
            # Let's first calculate the largest width in the current segment
            max_dw = np.max(delta_wave_local)
            # Convert it to pixel
            max_dpx = max_dw / pixel_scale_local
            # And define 50 times as a overhead
            if not per_pixel:
                px_step = int(wave_step / pixel_scale_local / 2) 
                px_over = int(np.ceil(max_dpx*25))
            else:
                px_step = 0
                px_over = int(np.ceil(max_dpx*25))
            
            spec_conv_local = np.zeros_like(wave_local)
            
            spec_loops = []
            rowmask = np.zeros(len(wave_local), dtype=bool)
            
            for i in range(px_step, len(wave_local), max(px_step,1)):
                # print(f"\r{i}, {len(wave_local)}", end="")
                di_high = 0
                if (i - px_step - px_over) < 0:
                    if jump_interval == 1:
                        # Cannot interpolate
                        continue
                    di = i - px_step - px_over
                    # We have to interpolate into the last range
                    prev_interval_wave = wave[scale_jump_px[jump_interval-2]:scale_jump_px[jump_interval-1]]
                    prev_interval_spec = spec[scale_jump_px[jump_interval-2]:scale_jump_px[jump_interval-1]]
                    
                    lin_wave = np.linspace(wave_local[0] - np.abs(di)*pixel_scale_local,
                                            wave_local[0],
                                            np.abs(di))
                    interp_spec = np.interp(lin_wave, prev_interval_wave, prev_interval_spec)
                    # Now you have the interpolated spectrum in the new sampling range
                    # Now stitch together
                    spec_loop = interp_spec
                    spec_loop = np.append(spec_loop, spec_local[:i+px_step+px_over+1])
                elif (i + px_step + px_over) >= len(wave_local):
                    if not len(scale_jump_px) > jump_interval + 1:
                        # Cannot interpolate
                        continue
                    
                    di = i + px_step + px_over - len(wave_local) + 1
                    # We have to interpolate into the last range
                    next_interval_wave = wave[scale_jump_px[jump_interval]:scale_jump_px[jump_interval+1]]
                    next_interval_spec = spec[scale_jump_px[jump_interval]:scale_jump_px[jump_interval+1]]
                    
                    lin_wave = np.linspace(wave_local[-1] + pixel_scale_local,
                                            wave_local[-1] + np.abs(di)*pixel_scale_local, np.abs(di))
                    interp_spec = np.interp(lin_wave, next_interval_wave, next_interval_spec)
                    # Now you have the interpolated spectrum in the new sampling range
                    # Now stitch together
                    spec_loop = spec_local[i - px_step - px_over:]
                    spec_loop = np.append(spec_loop, interp_spec)
                    di_high = di - px_over
                    
                else:
                    spec_loop = spec_local[i - (px_step+px_over):i + (px_step+px_over) + 1]
                    
                
                if convolution:
                    kernel = Gaussian1DKernel(stddev=sigma_px_local[i])
                    spec_conv_loop = convolve(spec_loop, kernel)
                    
                    
                    if di_high > 0:
                        spec_conv_local[i-px_step:i+px_step+1] = spec_conv_loop[px_over:px_over+2*px_step+1-di_high]
                    else:
                        spec_conv_local[i-px_step:i+px_step+1] = spec_conv_loop[px_over:px_over+2*px_step+1]
                    
                else:
                    if old:
                        kernel = gaussian(np.linspace(-(px_step+px_over), (px_step+px_over), len(spec_loop)), 0., sigma_px_local[i])
                        spec_conv_loop = np.sum(spec_loop * kernel)
                        spec_conv_local[i] = spec_conv_loop
                    if not old:
                        rowmask[i] = True
                        spec_loops.append(spec_loop)
                    
            if not old:
                root = cfg.conf_dict["datapath"] / "macroturbulence_kernels"
                kernels_file = root / Path(f"kernels_v{v_macro:.1f}_w{wave_start}_{wave_stop}.npy")
                if False:
                # if kernels_file.is_file():
                    kernels = np.load(kernels_file)
                else:
                    # Lets try to precompute the kernels in an array
                    lin_px = np.linspace(-px_over, px_over, 2*px_over+1)
                    # kernels = np.array([gaussian(lin_px, 0., sigma) for sigma in sigma_px_local])
                    lin_px = np.array([lin_px for i in range(len(wave_local))])
                    sigma_px_local = np.array([sigma_px_local for i in range(2*px_over+1)]).T
                    kernels = gaussian(lin_px, 0., sigma_px_local)
                    np.save(kernels_file, kernels)
                    
                spec_loops = np.array(spec_loops)
                kernels = kernels[rowmask,:]
        
                spec_conv_local[rowmask] = np.sum(spec_loops * kernels, axis=1)
                del kernels
                del spec_loops
            spec_conv[last_idx:idx] = spec_conv_local
            last_idx = idx      
    
    if debug_plot:
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = cfg.conf_dict["datapath"] / "/plots/macroturbulence"
        savename = f"macroturbulence.png"
        outfile = out_root / savename
        # Only save one debug plot (otherwise you would have that for every cell)
        if not outfile.is_file():    
            fig, ax = plt.subplots(1, figsize=(6.35, 3.5))
            ax.plot(wave, spec, label="Simulated Spectrum")
            ax.plot(wave, spec_conv, label=f"Broadend Spectrum by v_macro={v_macro}m/s")
            ax.set_xlim(wave[center_idx]-5, wave[center_idx]+5)
            ax.set_xlabel(r"Wavelength $[\AA]$")
            ax.set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
            ax.legend()
            fig.set_tight_layout(True)
            plt.savefig(f"{out_root}/{savename}", dpi=600)
        
        
        plt.close()
    
    return spec_conv

def measure_bisector_on_line(wave, spec, line):
    """ Convienience function to measure a bisector on a line."""
    # Choose an interval around the theoretical line center to look at
    interval = 0.25
    mask = np.logical_and(wave >= line - interval, wave <= line + interval)

    wv = wave[mask]
    sp = spec[mask]

    # Fit the width and center for an inital guess
    expected = (line, 0.05, 0.9, 1.0)
    
    params, cov = curve_fit(_gauss_continuum, wv, sp, expected)
    width = params[1]
    continuum = params[-1]
    
    print(params)
    center = params[0]
    


    
    bis_wave, bis, left_wv, left_sp, right_wv, right_sp = bisector_on_line(wv, 
                                                                        sp, 
                                                                        center,
                                                                        width=width,
                                                                        outlier_clip=0.05,
                                                                        continuum=continuum)
    
    print(bis_wave, bis)
    
    bis_v = (bis_wave - line) / bis_wave * 3e8
    
    return bis_wave, bis_v, bis


def oversampled_wave_interpol(rest_wave, wave, spec):
    """ Take a shifted wave, spec array pair, oversample it and interpolate back onto
        a rest_wave
    """
    # Define a fine wavegrid in the shifted frame
    fine_wave = np.arange(wave[0], wave[-1], 0.001)
    # Calc the oversampled shifted spec
    cs_raw = CubicSpline(wave, spec)
    fine_spec = cs_raw(fine_wave)
    spec_interpol = np.interp(rest_wave, fine_wave, fine_spec)
    
    return spec_interpol
    


if __name__ == "__main__":
    wave, spec, header = phoenix_spectrum()
    
    mask = np.logical_and(wave>4900, wave<5100)
    # wave = wave[mask]
    # spec = spec[mask]
    spec_R = add_isotropic_convective_broadening(wave, spec, v_macro=5000)
    
    fig, ax = plt.subplots(1, figsize=(6.35, 3.5))
    ax.plot(wave, spec, "bx")
    ax.plot(wave, spec_R, "r*")
    
    mask = np.logical_and(wave>4999, wave<5001)
    wave = wave[mask]
    spec = spec[mask]
    spec_R = spec_R[mask]
    print(wave[spec_R.argmin()])
    print(wave[spec.argmin()])
    
    ax.set_xlim(4999, 5001)
    
    
    
    plt.savefig("dbug.png", dpi=300)