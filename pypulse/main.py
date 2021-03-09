from datetime import datetime, date, timedelta
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from dataloader import phoenix_spectrum, carmenes_template
from datasaver import save_spectrum
from exopy import observatories
from barycorrpy import get_BC_vel
from astropy.time import Time


def create_rv_series(P=600, N=20, K=0, spot=False):
    """ Create a fake RV series.

        :param P: period in days
        :param N: Number of datapoints
        :param K: Amplitude in m/s
    """
    # At the moment, fix today as last observation date
    today = date.today()
    stop = datetime.combine(today, datetime.min.time())

    # Sample one Period
    P_sample = np.linspace(0, P, N, dtype=int)
    time_sample = np.array([stop - timedelta(days=int(d))
                            for d in P_sample[::-1]])
    phase_sample = 1 / P * P_sample
    K_sample = K * np.sin(2 * np.pi * phase_sample)

    hip = 73620
    K_sample = add_barycentric_correction(K_sample, time_sample, hip)

    # Load one rest_spectrum, all units in Angstrom
    min_wave = 5000
    max_wave = 12000
    center_wave = (max_wave + min_wave) / 2
    wavelength_range = (min_wave - 10, max_wave + 10)
    rest_spectrum, rest_wavelength, _ = phoenix_spectrum(
        Teff=4800, wavelength_range=wavelength_range)

    # Shift in Angstom. Approximation, fixed wavelength
    # shift_sample = K_sample / 3e8 * center_wave

    # shift_wavelengths = []
    # for shift in shift_sample:
    #     shift_wavelengths.append(rest_wavelength + shift)
    c = 299792458  # m/s
    # Add the Doppler shifts
    shift_wavelengths = []
    for v in K_sample:
        shift_wavelengths.append(rest_wavelength + v / c * rest_wavelength)

    new_specs = []
    for shift_wavelength in shift_wavelengths:
        spec, wave = interpolate_carmenes(rest_spectrum, shift_wavelength)
        new_specs.append(spec)

    for idx, time in enumerate(time_sample):
        new_header = get_new_header(time)
        timestr = time.strftime("%Y%m%dT%Hh%Mm%Ss")
        filename = f"car-{timestr}-sci-fake-vis_A.fits"

        save_spectrum(new_specs[idx], new_header, filename)


def get_new_header(time):
    """ Create the new header for the fake Carmenes spectrum.

        Add only keys that should be new.
    """
    time = Time(time, scale="utc")
    header_dict = {"DATE-OBS": time.isot.split(".")[0],
                   "CARACAL DATE-OBS": time.isot.split(".")[0],
                   "MJD-OBS": time.mjd,
                   "CARACAL MJD-OBS": time.mjd}

    return header_dict


def add_barycentric_correction(K_array, time_list, star):
    """ Add the barycentric correction to the K_list."""
    tmean = 53.0455
    print(time_list)
    time_list = [t + timedelta(seconds=tmean) for t in time_list]
    print(time_list)
    jdutc_times = [Time(t, scale="utc") for t in time_list]
    for jdutc in jdutc_times:
        jdutc.format = "jd"

    caha = observatories.calar_alto
    lat = float(caha["lat"].replace(" N", ""))
    lon = -float((caha["lon"].replace(" W", "")))
    alt = 2168.

    bcs = []
    for jdutc in jdutc_times:

        result = get_BC_vel(JDUTC=jdutc, hip_id=star, lat=lat, longi=lon,
                            alt=alt, ephemeris='de430')
        bcs.append(float(result[0]))
    bcs = np.array(bcs)
    print("BARYCENTRIC CORRECTIONS")
    print(bcs)
    return K_array - bcs


def interpolate_carmenes(spectrum, wavelength):
    """ Interpolate to the Carmenes spectrum."""
    (spec, cont, sig, wave) = carmenes_template()

    new_spec = []
    for order in range(len(wave)):

        order_spec = []
        func = interpolate.interp1d(wavelength, spectrum)
        order_spec = func(wave[order])

        # Reduce the level to something similar to CARMENES
        order_spec = order_spec * \
            np.nanmean(spec[order]) / np.nanmean(order_spec)

        order_cont = cont[order] / np.mean(cont[order])
        order_spec = order_spec * order_cont

        order_spec = gaussian_filter1d(order_spec, 5)

        # Set the old oders that were nan back to nan
        nan_mask = np.isnan(sig[order])
        order_spec[nan_mask] = np.nan

        new_spec.append(order_spec)
    new_spec = np.array(new_spec)
    return new_spec, wave


if __name__ == "__main__":
    create_rv_series()
