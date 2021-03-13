from datetime import datetime, date, timedelta
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from barycorrpy import get_BC_vel
from astropy.time import Time
from plapy import observatories
from plapy.constants import C
from dataloader import phoenix_spectrum, carmenes_template
from datasaver import save_spectrum
from star import GridStar


# TODO Make adjustable
HIP = 73620
MIN_WAVE = 5000
MAX_WAVE = 12000
# End TODO


def sample_phase(P, N):
    """ Return a phase sample and the corresponding time sample.

        Phase ranges from 0 to 1
    """
    # At the moment, fix today as last observation date
    stop = datetime.combine(date.today(), datetime.min.time())

    # Sample one Period
    P_sample = np.linspace(0, P, N, dtype=int)
    time_sample = np.array([stop - timedelta(days=int(d))
                            for d in P_sample[::-1]])
    phase_sample = 1 / P * P_sample

    return phase_sample, time_sample


def get_planet_spectra(P=600, N=20, K=0):
    """ Return a list of wavelengths and fluxes for a planetary signal."""
    phase_sample, time_sample = sample_phase(P, N)
    K_sample = K * np.sin(2 * np.pi * phase_sample)

    K_sample = add_barycentric_correction(K_sample, time_sample, HIP)

    # Load one rest_spectrum, all units in Angstrom
    wavelength_range = (MIN_WAVE - 10, MAX_WAVE + 10)
    rest_wavelength, rest_spectrum, _ = phoenix_spectrum(
        Teff=4800, wavelength_range=wavelength_range)

    # Add the Doppler shifts
    shift_wavelengths = []
    spectra = []
    for v in K_sample:
        shift_wavelengths.append(rest_wavelength + v / C * rest_wavelength)
        spectra.append(rest_spectrum)

    return shift_wavelengths, spectra, time_sample


def get_spot_spectra(P=30, N=20):
    """ Simulate the spot spectra."""
    phase_sample, time_sample = sample_phase(P, N)

    # TODO REMOVE
    phase_sample = phase_sample[:-1]
    time_sample = time_sample[:-1]
    # END TODO
    # At the moment assume that there is no planetary signal present
    # But still create K_sample for barycentric correction

    K_sample = np.zeros(len(time_sample))

    K_sample = add_barycentric_correction(K_sample, time_sample, HIP)

    shift_wavelengths = []
    spectra = []
    i = 0

    import matplotlib.pyplot as plt
    for v, phase in zip(K_sample, phase_sample):
        print(f"Calculate star {i}")
        star = GridStar(N_star=100, vsini=3000)
        star.add_spot(phase=phase, radius=25)

        # plt.imshow(star.temperature)
        # plt.savefig(
        #     f"/home/dane/Documents/PhD/pypulse/plots/{round(phase,3)}.pdf")
        # continue

        # Wavelength in restframe of phoenix spectra but already perturbed by
        # spot
        rest_wavelength, rest_spectrum = star.calc_spectrum(
            MIN_WAVE - 10, MAX_WAVE + 10)

        # Add doppler shift due to barycentric correction
        shift_wavelengths.append(rest_wavelength + v / C * rest_wavelength)
        spectra.append(rest_spectrum)

    # exit()
    return shift_wavelengths, spectra, time_sample


def create_rv_series(P=600, N=20, K=200, mode="RV"):
    """ Create a fake RV series.

        :param P: period in days
        :param N: Number of datapoints
        :param K: Amplitude in m/s
    """
    print(mode)
    if mode == "RV":
        shift_wavelengths, spectra, time_sample = get_planet_spectra(
            P=P, N=N, K=K)
    elif mode == "spot":
        shift_wavelengths, spectra, time_sample = get_spot_spectra(P=P, N=N)
    elif mode == "pulsation":
        shift_wavelengths, spectra, time_sample = get_pulsation_spectra(
            P=P, N=N)

    new_specs = []
    for shift_wavelength, spectrum in zip(shift_wavelengths, spectra):
        spec, wave = interpolate_carmenes(spectrum, shift_wavelength)
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
    time_list = [t + timedelta(seconds=tmean) for t in time_list]
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


def get_pulsation_spectra(P=600, N=20):
    """ Simulate the pulsation spectra."""
    phase_sample, time_sample = sample_phase(P, N)

    # TODO REMOVE
    phase_sample = phase_sample[:-1]
    time_sample = time_sample[:-1]
    # END TODO
    # At the moment assume that there is no planetary signal present
    # But still create K_sample for barycentric correction

    K_sample = np.zeros(len(time_sample))

    K_sample = add_barycentric_correction(K_sample, time_sample, HIP)

    shift_wavelengths = []
    spectra = []
    i = 0

    import matplotlib.pyplot as plt
    for v, phase in zip(K_sample, phase_sample):
        print(f"Calculate star {i}")
        star = GridStar(N_star=100, vsini=3000)
        star.add_pulsation(l=2, m=2, phase=phase)

        plt.imshow(star.pulsation.real, origin="lower",
                   cmap="seismic", vmin=-400, vmax=400)
        plt.savefig(
            f"/home/dane/Documents/PhD/pypulse/plots/pulsation/{round(phase,3)}.pdf")
        plt.close()

        # Wavelength in restframe of phoenix spectra but already perturbed by
        # pulsation
        rest_wavelength, rest_spectrum = star.calc_spectrum(
            MIN_WAVE - 10, MAX_WAVE + 10)

        # Add doppler shift due to barycentric correction
        shift_wavelengths.append(rest_wavelength + v / C * rest_wavelength)
        spectra.append(rest_spectrum)

    return shift_wavelengths, spectra, time_sample


if __name__ == "__main__":
    create_rv_series(N=21, mode="pulsation")
