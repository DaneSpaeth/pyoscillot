import numpy as np
from astropy.time import Time
from scipy.interpolate import interp1d
from utils import adjust_resolution
from dataloader import carmenes_template


def interpolate(spectrum, wavelength):
    """ Interpolate to the Carmenes spectrum."""
    (spec, cont, sig, wave) = carmenes_template()

    new_spec = []
    spectrum = adjust_resolution(wavelength, spectrum, R=90000, w_sample=5)
    for order in range(len(wave)):

        order_spec = []
        func = interp1d(wavelength, spectrum, kind="linear")
        order_spec = func(wave[order])

        # Reduce the level to something similar to CARMENES
        order_spec = order_spec * \
            np.nanmean(spec[order]) / np.nanmean(order_spec)

        order_cont = cont[order] / np.mean(cont[order])
        order_spec = order_spec * order_cont

        # orig_resolution=90000
        # order_spec = adjust_resolution(
        #    wave[order], order_spec, R=120000, w_sample=50)

        # order_spec = adjust_snr(order_spec, wave[order], spec[order], sig[order],
        #                         snr=3 * np.nanmedian(spec[order] / sig[order]))

        # order_spec = gaussian_filter1d(order_spec, 5)

        # Set the old orders that were nan back to nan
        nan_mask = np.isnan(sig[order])
        order_spec[nan_mask] = np.nan

        new_spec.append(order_spec)
    new_spec = np.array(new_spec)

    return new_spec, wave


def get_new_header(time, bc=None, bjd=None):
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

    return header_dict
