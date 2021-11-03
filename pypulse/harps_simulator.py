import numpy as np
from scipy.interpolate import interp1d
from astropy.time import Time
from dataloader import harps_template
from utils import adjust_resolution


def interpolate(spectrum, wavelength):
    """ Interpolate to the HARPS spectrum."""
    (spec, wave, blaze) = harps_template()

    interpol_spec = []
    spectrum = adjust_resolution(wavelength, spectrum, R=115000, w_sample=5)
    for order in range(len(wave)):

        order_spec = []
        func = interp1d(wavelength, spectrum, kind="linear")
        order_spec = func(wave[order])

        # Adjust for the blaze
        order_spec *= blaze[order]

        # Reduce the level to something similar to HARPS
        order_spec = order_spec * \
            np.nanmax(spec[order]) / np.nanmax(order_spec)

        # At the moment do not correct for the continuum
        # order_cont = cont[order] / np.mean(cont[order])
        # order_spec = order_spec * order_cont

        interpol_spec.append(order_spec)
    interpol_spec = np.array(interpol_spec)

    return interpol_spec, wave


def get_new_header(time, bc=None, bjd=None):
    """ Create the new header for the fake Carmenes spectrum.

        :param time: Time of observation
        :param bc: Barycentric Correction to write into DRS
        :param bjd: Barycentric Julian Date to write into DRS

        Add only keys that should be new.
    """
    time = Time(time, scale="utc")
    header_dict = {"DATE": time.isot,
                   "DATE-OBS": time.isot,
                   "MJD-OBS": time.mjd,
                   "JD": time.jd - 2400000,
                   "HJD": time.jd - 2400000,
                   "PI-COI": "SpaethSim",
                   "OBJECT": "Sim",
                   "TEXPTIME": 100.0}

    # serval uses the comment of MJD-OBS to retrieve a timeid
    comment_dict = {"MJD-OBS": f"MJD start ({time.isot})"}
    # HJD is wrong but not so important at the moment
    if bc is not None:
        header_dict["HIERARCH ESO DRS BERV"] = bc / 1000
    if bjd is not None:
        header_dict["HIERARCH ESO DRS BJD"] = bjd - 2400000

    # others: RA, DEC, UTC, LST, MJD-END

    return header_dict, comment_dict
