import numpy as np
import matplotlib.pyplot as plt
from plapy.constants import C
from PyAstronomy import pyasl
import time
from utils import interpolate_to_restframe
from pathlib import Path
from scipy.optimize import least_squares, curve_fit
from numpy.polynomial.polynomial import Polynomial


def calc_theoretical_results(ref_wave, ref_spec, wave, spec, bjd):
    """ Calculate all theoretical results, i.e. RVs, CRX, dLW

        :param list shift_wavelengths: All shifted wavelengths
        :param list spectra: List of spectra
        :param list bjds: List of BJDs

    """

    fig, ax = plt.subplots(1)

    # rv = calc_rv(ref_wave, ref_spec, wave, spec)
    least_square_rvfit(ref_wave, ref_spec, wave, spec)
    exit()
    ax.plot(ref_wave, ref_spec, label="Reference")
    ax.plot(wave, spec, label="Pulsation")

    ax.legend()
    plt.show()
    exit()

    print(ref_wave - wave)

    print("Run Cross-correlation global with mode doppler")
    rv, cc = pyasl.crosscorrRV(
        wave, spec, ref_wave, ref_spec, -1.00, -0.3, 0.001, skipedge=100, mode="doppler")
    # Find the index of maximum cross-correlation function
    maxind = np.argmax(cc)
    print("Cross-correlation function is maximized at dRV = ",
          rv[maxind], " km/s")
    if rv[maxind] > 0.0:
        print("  A red-shift with respect to the template")
    else:
        print("  A blue-shift with respect to the template")
    rv_best = rv[maxind]
    print(rv_best)

    plt.plot(rv, cc, 'bp-')
    plt.plot(rv[maxind], cc[maxind], 'ro')
    plt.show()
    exit()

    wave_chunks, rv_chunks = calc_rv_chunks(ref_wave, ref_spec, wave, spec)
    print(np.mean(rv_chunks))
    fig, ax = plt.subplots(1)
    ax.scatter(wave_chunks, rv_chunks)
    plt.show()


def calc_rv(ref_wave, ref_spec, wave, spec):
    """ Calculate the RV between Reference Spectrum  and Spectrum."""
    # Carry out the cross-correlation.
    # The RV-range is -30 - +30 km/s in steps of 0.6 km/s.
    # The first and last 20 points of the data are skipped.
    print("Calculate RV via CCF")
    start = time.time()
    rv, cc = pyasl.crosscorrRV(
        wave, spec, ref_wave, ref_spec, -0.500, 0.500, 0.0001, skipedge=20)
    stop = time.time()
    print(f"CCF took {round(stop-start,2)}s")

    # Find the index of maximum cross-correlation function
    maxind = np.argmax(cc)

    print("Cross-correlation function is maximized at dRV = ",
          rv[maxind], " km/s")
    if rv[maxind] > 0.0:
        print("  A red-shift with respect to the template")
    else:
        print("  A blue-shift with respect to the template")
    rv_best = rv[maxind]
    return rv_best


def calc_rv_chunks(ref_wave, ref_spec, wave, spec):
    """ Calculate RV in chunks which allows to calculate a CRX."""
    # You'll get twice the chunks since we will also overlap in between
    n_chunks = 50
    border = 100

    chunk_size_px = int((len(ref_wave) - 2 * border) / n_chunks)

    rv_chunks = []
    wave_chunks = []
    for n_chunk in range(n_chunks):
        idx_range = slice(border + n_chunk * chunk_size_px,
                          (n_chunk + 1) * chunk_size_px)
        ref_wave_chunk = ref_wave[idx_range]
        ref_spec_chunk = ref_spec[idx_range]
        wave_chunk = wave[idx_range]
        spec_chunk = spec[idx_range]

        ref_spec_chunk = ref_spec_chunk * \
            np.median(spec_chunk) / np.median(ref_spec_chunk)

        print(f"Calculate RV via least_square_rvfit for chunk {n_chunk}")
        start = time.time()
        rv_chunk = least_square_rvfit(
            ref_wave_chunk, ref_spec_chunk, wave_chunk, spec_chunk)
        stop = time.time()

        print(f"RV={round(rv_chunk, 3)}m/s. Fit took {round(stop-start,2)}s")

        rv_chunks.append(rv_chunk)
        wave_chunks.append(np.mean(ref_wave_chunk))

    rv_chunks = np.array(rv_chunks)
    wave_chunks = np.array(wave_chunks)

    return wave_chunks, rv_chunks


def least_square_rvfit(ref_wave, ref_spec, wave, spec):
    def func(v_shift, ref_wave, ref_spec, wave, spec):
        shift_wave = wave * (1 + v_shift[0] / C)
        # Interpolate back to the ref (and therefore simulated wavelength grid)
        interpol_spec = interpolate_to_restframe(shift_wave,
                                                 spec,
                                                 ref_wave)

        return interpol_spec - ref_spec

    x0 = np.array([10])
    # bounds = (-1000, 1000)
    res = least_squares(func, x0, method="lm",
                        args=(ref_wave, ref_spec, wave, spec))

    return -res.x[0]


def fit_crx(wave, rv):
    """ Fit the chromatic index."""
    def func(log_wave, alpha, crx):
        return alpha + log_wave * crx
    popt, pcov = curve_fit(func, np.log(wave), rv)

    alpha, crx = popt

    print(alpha, crx)

    return crx, alpha


def test():
    sim = "tmp"
    wave_files = Path(sim).glob("wave_*.npy")
    spec_files = Path(sim).glob("spectrum_*.npy")

    waves = []
    specs = []
    vs = []
    bjds = []

    for wave_file, spec_file in zip(sorted(list(wave_files)),
                                    sorted(list(spec_files))):
        v = float(spec_file.name.split("spectrum_")
                  [-1].split(".npy")[0].split("_")[0])
        bjd = float(spec_file.name.split("spectrum_")
                    [-1].split(".npy")[0].split("_")[-1])
        if v == 0:
            ref_wave = np.load(wave_file)
            ref_spec = np.load(spec_file)
            wave_mask_low = ref_wave > 5200
            wave_mask_high = ref_wave < 9600
            wave_mask = np.logical_and(wave_mask_low, wave_mask_high)
            ref_wave = ref_wave[wave_mask]
            ref_spec = ref_spec[wave_mask]
        else:
            wave = np.load(wave_file)
            wave_mask_low = wave > 5200
            wave_mask_high = wave < 9600
            wave_mask = np.logical_and(wave_mask_low, wave_mask_high)
            waves.append(wave[wave_mask])
            specs.append(np.load(spec_file)[wave_mask])
            vs.append(v)
            bjds.append(bjd)

    v_fits = []
    crxs = []
    for wave, spec, v_theo in zip(waves, specs, vs):
        wave_chunks, rv_chunks = calc_rv_chunks(ref_wave, ref_spec, wave, spec)
        fig, ax = plt.subplots(1)
        ax.semilogx(wave_chunks, rv_chunks, "bo")
        crx, alpha = fit_crx(wave_chunks, rv_chunks)
        lin_wave = np.linspace(np.min(wave_chunks),
                               np.max(wave_chunks))
        # print(np.log(lin_wave) * crx)
        # exit()

        ax.semilogx(lin_wave, np.log(lin_wave) * crx + alpha)
        plt.show()
        crxs.append(crx)

        v_fit = least_square_rvfit(ref_wave, ref_spec, wave, spec)
        print(
            f"v_fit={round(v_fit,3)}, v_theo={round(v_theo,3)}, delta_v={round(v_fit - v_theo,3)}")
        v_fits.append(v_fit)

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(bjds, v_fits, "bo")
    ax[1].plot(bjds, crxs, "bo")
    plt.show()


if __name__ == "__main__":
    test()
