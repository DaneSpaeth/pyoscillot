from datetime import datetime, date, timedelta
import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from barycorrpy import get_BC_vel, utc_tdb
from astropy.time import Time
from plapy.obs import observatories
from plapy.constants import C
from dataloader import phoenix_spectrum, carmenes_template
from datasaver import save_spectrum
from star import GridSpectrumSimulator
from utils import adjust_snr, adjust_resolution


# TODO Make adjustable
HIP = 73620
MIN_WAVE = 5000
MAX_WAVE = 12000
# End TODO


def sample_phase(P, N, N_phases=1):
    """ Return a phase sample and the corresponding time sample.

        Phase ranges from 0 to 1
    """
    # At the moment, fix today as last observation date
    stop = datetime.combine(date.today(), datetime.min.time())

    # Sample one Period
    P_sample = np.linspace(0, N_phases * P, N, dtype=int)
    time_sample = np.array([stop - timedelta(days=int(d))
                            for d in P_sample[::-1]])
    phase_sample = (1 / P * P_sample) % 1

    return phase_sample, time_sample


def get_planet_spectra(P=600, N=20, K=0):
    """ Return a list of wavelengths and fluxes for a planetary signal."""
    phase_sample, time_sample = sample_phase(P, N)
    K_sample = K * np.sin(2 * np.pi * phase_sample)

    K_sample, bcs, bjds = add_barycentric_correction(
        K_sample, time_sample, HIP)

    # Load one rest_spectrum, all units in Angstrom
    wavelength_range = (MIN_WAVE - 10, MAX_WAVE + 10)
    rest_wavelength, rest_spectrum, _ = phoenix_spectrum(
        Teff=4800, wavelength_range=wavelength_range)

    # Add the Doppler shifts
    shift_wavelengths = []
    spectra = []
    for v in K_sample:
        vo = v
        ve = 0
        c = 299792458.0
        a = (1.0 + vo / c) / (1.0 + ve / c)
        print(a)
        shift_wavelengths.append(np.exp(np.log(rest_wavelength) + np.log(a)))
        spectra.append(rest_spectrum)

    return shift_wavelengths, spectra, time_sample, bcs, bjds


def get_spot_spectra(P=30, N=20):
    """ Simulate the spot spectra."""
    phase_sample, time_sample = sample_phase(P, N)

    # TODO REMOVE
    phase_sample = phase_sample[:-1]
    time_sample = time_sample[:-1]
    # END TODO
    # At the moment assume that there is no planetary signal present
    # But still create K_sample for barycentric correction

    with open("phases.txt", "w") as f:
        for p in phase_sample:
            f.write(f"{p}\n")

    K_sample = np.zeros(len(time_sample))

    K_sample, bcs, bjds = add_barycentric_correction(
        K_sample, time_sample, HIP)

    shift_wavelengths = []
    spectra = []
    i = 0

    import matplotlib.pyplot as plt
    for v, phase in zip(K_sample, phase_sample):
        print(f"Calculate star {i}")
        star = GridSpectrumSimulator(N_star=100, v_rot=3000, inclination=60)
        star.add_spot(phase=phase, radius=20)

        # Wavelength in restframe of phoenix spectra but already perturbed by
        # spot
        rest_wavelength, rest_spectrum = star.calc_spectrum(
            MIN_WAVE - 10, MAX_WAVE + 10)

        # Add doppler shift due to barycentric correction
        shift_wavelengths.append(rest_wavelength + v / C * rest_wavelength)
        spectra.append(rest_spectrum)

    return shift_wavelengths, spectra, time_sample, bcs, bjds


def create_rv_series(P=600, N=20, K=200, mode="RV"):
    """ Create a fake RV series.

        :param P: period in days
        :param N: Number of datapoints
        :param K: Amplitude in m/s
    """
    if mode == "RV":
        shift_wavelengths, spectra, time_sample, bcs, bjds = get_planet_spectra(
            P=P, N=N, K=K)
    elif mode == "spot":
        shift_wavelengths, spectra, time_sample, bcs, bjds = get_spot_spectra(
            P=P, N=N)
    elif mode == "pulsation":
        shift_wavelengths, spectra, time_sample, bcs, bjds = get_pulsation_spectra(
            P=P, N=N)
    else:
        print("Select a Mode")
        exit()

    new_specs = []
    for shift_wavelength, spectrum in zip(shift_wavelengths, spectra):
        spec, wave = interpolate_carmenes(spectrum, shift_wavelength)
        new_specs.append(spec)

    for idx, time in enumerate(time_sample):
        new_header = get_new_header(time, bcs[idx], bjds[idx])
        timestr = time.strftime("%Y%m%dT%Hh%Mm%Ss")
        filename = f"car-{timestr}-sci-fake-vis_A.fits"

        save_spectrum(new_specs[idx], new_header, filename)


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


def add_barycentric_correction(K_array, time_list, star, set_0=True):
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
    bjds = []
    for jdutc in jdutc_times:

        # result = get_BC_vel(JDUTC=jdutc, hip_id=star, lat=lat, longi=lon,
        #                     alt=alt, ephemeris='de430')
        result = get_BC_vel(JDUTC=jdutc,
                            ra=225.72515818125,
                            dec=2.0913040080555554,
                            epoch=2451545.0,
                            pmra=-54.89,
                            pmdec=13.34,
                            px=0.0,
                            lat=37.2236,
                            longi=-2.5463,
                            alt=2168.0)
        bjd_result = utc_tdb.JDUTC_to_BJDTDB(JDUTC=jdutc,
                                             ra=225.72515818125,
                                             dec=2.0913040080555554,
                                             epoch=2451545.0,
                                             pmra=-54.89,
                                             pmdec=13.34,
                                             px=0.0,
                                             lat=37.2236,
                                             longi=-2.5463,
                                             alt=2168.0)
        bcs.append(float(result[0]))
        bjds.append(float(bjd_result[0]))

    bcs = np.array(bcs)
    if set_0:
        bcs *= 0
        print(bcs)
    return K_array - bcs, bcs, bjds


def interpolate_carmenes(spectrum, wavelength):
    """ Interpolate to the Carmenes spectrum."""
    (spec, cont, sig, wave) = carmenes_template()

    new_spec = []
    spectrum = adjust_resolution(wavelength, spectrum, R=90000, w_sample=5)
    for order in range(len(wave)):

        order_spec = []
        func = interpolate.interp1d(wavelength, spectrum, kind="linear")
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


def get_pulsation_spectra(P=600, N=20):
    """ Simulate the pulsation spectra."""
    phase_sample, time_sample = sample_phase(P, N, N_phases=1)

    K_sample = np.zeros(len(time_sample))

    K_sample, bcs, bjds = add_barycentric_correction(
        K_sample, time_sample, HIP)

    shift_wavelengths = []
    spectra = []
    i = 0

    import matplotlib.pyplot as plt
    for v, phase in zip(K_sample, phase_sample):
        print(f"Calculate star {i} at phase {phase}")
        i += 1
        star = GridSpectrumSimulator(
            N_star=100, N_border=3, Teff=4800, v_rot=3000, T_var=50, inclination=60)

        star.add_pulsation(l=4, m=4, phase=phase)

        plt.imshow(star.projector.pulsation(), origin="lower",
                   cmap="seismic")
        plt.savefig(
            f"/home/dane/Documents/PhD/pypulse/plots/new_pulsations/{round(phase,3)}.pdf")
        plt.close()

        # Wavelength in restframe of phoenix spectra but already perturbed by
        # pulsation
        rest_wavelength, rest_spectrum = star.calc_spectrum(
            MIN_WAVE - 10, MAX_WAVE + 10)

        # Add doppler shift due to barycentric correction
        shift_wavelengths.append(rest_wavelength + v / C * rest_wavelength)
        spectra.append(rest_spectrum)

    return shift_wavelengths, spectra, time_sample, bcs, bjds


if __name__ == "__main__":
    create_rv_series(P=600, N=30, K=10, mode="RV")
