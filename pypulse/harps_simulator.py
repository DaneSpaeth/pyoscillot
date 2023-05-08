import numpy as np
from scipy.interpolate import interp1d
from astropy.time import Time
from dataloader import harps_template
from utils import adjust_resolution


def interpolate(spectrum, wavelength):
    """ Interpolate to the HARPS spectrum.


        :param np.array spectrum: The calculated spectrum to interpolate to HARPS
        :param np.array wavelength: The calculated wavelength to interpolate to HARPS
    """
    # Load the template spectra and wavelength
    (tmpl_spec, tmpl_wave, tmpl_blaze) = harps_template(spec_filename="HARPS_template_ngc4349_127_e2ds_A.fits")

    interpol_spec = []
    R_real = 115000
    # R_test = 130000
    print("Adjusting Resolution")
    for order in range(len(tmpl_wave)):

        # Cut the calculated wavelengths and spectra to the order
        local_wave_mask = np.logical_and(wavelength > tmpl_wave[order][0] - 100,
                                         wavelength < tmpl_wave[order][-1] + 100)

        local_wavelength = wavelength[local_wave_mask]
        local_spectrum = spectrum[local_wave_mask]

        # Adjust the resolution per order
        # This will use one kernel per order
        local_spectrum = adjust_resolution(local_wavelength, local_spectrum, R=R_real, w_sample=20)

        # Interpolate the calculated spectra onto the tmpl_wave grid
        func = interp1d(local_wavelength, local_spectrum, kind="linear")
        order_spec = func(tmpl_wave[order])

        # Adjust for the blaze
        order_spec *= tmpl_blaze[order]

        # Reduce the level to something similar to HARPS
        order_spec = order_spec * \
                     np.nanmean(tmpl_spec[order]) / np.nanmean(order_spec)

        interpol_spec.append(order_spec)
    interpol_spec = np.array(interpol_spec)

    return interpol_spec, tmpl_wave


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
                   "MJD-OBS": time.mjd + 2400000,
                   #"JD": time.jd - 2400000,
                   #"HJD": time.jd - 2400000,
                   "PI-COI": "SpaethSim",
                   "OBJECT": "Sim",
                   "TEXPTIME": 100.0}

    # serval uses the comment of MJD-OBS to retrieve a timeid
    comment_dict = {"MJD-OBS": f"MJD start ({time.isot})"}
    # HJD is wrong but not so important at the moment
    if bc is not None:
        header_dict["HIERARCH ESO DRS BERV"] = bc / 1000
    if bjd is not None:
        header_dict["HIERARCH ESO DRS BJD"] = bjd

    # others: RA, DEC, UTC, LST, MJD-END

    return header_dict, comment_dict

if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    num = 1000000
    wave = np.linspace(3000, 9000, num)
    spec = np.ones_like(wave)

    interval = 100
    peak_idx = np.arange(int(0 + interval / 2), num, interval)
    spec[peak_idx] -= 0.5

    interpol_spec, tmpl_wave = interpolate(spec, wave)

    order = 30
    plt.figure(dpi=300)
    plt.plot(tmpl_wave[order], interpol_spec[order])

    w = tmpl_wave[order]
    s = interpol_spec[order]

    mins_idx = s < 0.973

    plt.plot(w[mins_idx], s[mins_idx])
    plt.show()


