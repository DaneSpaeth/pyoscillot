import numpy as np
from astropy.time import Time
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from utils import adjust_resolution
from dataloader import carmenes_template


def interpolate(spectrum, wavelength, template_file=None,
                target_max_snr=300, adjust_snr=True, add_noise=True,
                snr_profile=None):
    """ Interpolate to the Carmenes spectrum."""
    if template_file is not None:
        (spec_templ, cont_templ, sig_templ,
         wave_templ) = carmenes_template(template_file)
    else:
        (spec_templ, cont_templ, sig_templ,
         wave_templ) = carmenes_template()

    if snr_profile is None:
        snr_per_order = np.nanmedian(spec_templ / sig_templ, axis=1)
        # snr_per_order /= np.nanmax(snr_per_order)
    else:
        # If a snr profile is given it should be given as a normalized quantity
        # Adjust to target_max_snr
        snr_per_order = snr_profile * target_max_snr

    new_spec = []
    spectrum = adjust_resolution(wavelength, spectrum, R=90000, w_sample=5)
    for order in range(len(wave_templ)):

        order_spec = []
        func = interp1d(wavelength, spectrum, kind="linear")
        order_spec = func(wave_templ[order])

        # Reduce the level to something similar to CARMENES
        order_spec = order_spec * \
            np.nanmean(spec_templ[order]) / np.nanmean(order_spec)

        # Do not correct for cont anymore
        # order_cont = cont[order] / np.mean(cont[order])
        # order_spec = order_spec * order_cont

        # Adjust the signal to noise ratio and also adds noise if add_noise
        # is True
        if adjust_snr:
            order_spec = adjust_snr_order(
                order_spec,
                spec_templ[order],
                sig_templ[order],
                wave_templ[order],
                add_noise,
                new_median_snr=snr_per_order[order])

        # Set the old orders that were nan back to nan
        nan_mask = np.isnan(sig_templ[order])
        order_spec[nan_mask] = np.nan

        new_spec.append(order_spec)
    new_spec = np.array(new_spec)

    return new_spec, wave_templ


def get_new_header(time, bc=None, bjd=None, snr_profile=None,
                   target_max_snr=None):
    """ Create the new header for the fake Carmenes spectrum.

        :param time: Time of observation
        :param bc: Barycentric Correction to write into DRS
        :param bjd: Barycentric Julian Date to write into DRS

        Add only keys that should be new.
    """
    time = Time(time, scale="utc")
    header_dict = {"DATE-OBS": time.isot.split(".")[0],
                   "CARACAL DATE-OBS": time.isot.split(".")[0],
                   "MJD-OBS": time.mjd,
                   "CARACAL MJD-OBS": time.mjd,
                   "CARACAL JD": time.jd - 2400000,
                   "CARACAL HJD": time.jd - 2400000}
    # HJD is wrong but not so important at the moment
    if bc is not None:
        header_dict["CARACAL BERV"] = bc / 1000
    if bjd is not None:
        header_dict["CARACAL BJD"] = bjd - 2400000

    if snr_profile is not None and target_max_snr is not None:
        for order, snr_ratio in enumerate(snr_profile):
            header_dict[f"CARACAL FOX SNR {order}"] = snr_ratio * \
                target_max_snr

    return header_dict


def adjust_snr_order(sp, sp_templ, sig_templ, wave_templ, add_noise,
                     new_median_snr=None):
    """ Adjust the SNR for one order.

        :param 1d array sp: Spectrum
        :param 1d array sp_templ: Template spectrum
        :param 1d array sig_templ: Template Sigma (Error)
        :param 1d array wave_templ: Template Wavelength grid
        :param bool add_noise: If True add noise to new spectrm

        :returns: SNR and noise adjusted array
    """
    # Do not change the original array
    sig_templ = sig_templ.copy()
    # Use the template SNR if None is given
    if new_median_snr is None:
        new_median_snr = np.nanmedian(sp_templ / sig_templ)

    # We first need to make sure there are no nans in the sig_template
    if np.any(np.isnan(sig_templ)):
        nan_idx = np.isnan(sig_templ)
        sig_templ[nan_idx] = np.interp(wave_templ[nan_idx],
                                       wave_templ[~nan_idx],
                                       sig_templ[~nan_idx])
    # Now smooth the noise
    filter_width = 40
    sig_templ = gaussian_filter1d(sig_templ, sigma=filter_width)
    smooth_sp = gaussian_filter1d(sp, sigma=filter_width)

    # Now rescale the spec to have the desired snr
    current_snr = np.nanmedian(smooth_sp / sig_templ)
    factor = new_median_snr / current_snr
    sp = sp * np.abs(factor)

    # Now add some noise
    # global_snr = gaussian_filter1d(spec, sigma=filter_width) / sig_template
    if add_noise:
        noise = np.random.normal(loc=np.zeros(
            len(sig_templ)), scale=sig_templ)
        noisy_sp = sp + noise
        return noisy_sp
    else:
        return sp
