import numpy as np
import matplotlib.pyplot as plt
from plapy.constants import C
from PyAstronomy import pyasl
import time


def calc_theoretical_results(ref_wave, ref_spec, wave, spec, bjd):
    """ Calculate all theoretical results, i.e. RVs, CRX, dLW

        :param list shift_wavelengths: All shifted wavelengths
        :param list spectra: List of spectra
        :param list bjds: List of BJDs

    """

    fig, ax = plt.subplots(1)

    # rv = calc_rv(ref_wave, ref_spec, wave, spec)
    # ax[0].scatter(bjd, rv_best)
    ax.plot(ref_wave, ref_spec, label="Reference")
    ax.plot(wave, spec, label="Pulsation")
    ax.legend()
    plt.show()

    wave_chunks, rv_chunks = calc_rv_chunks(ref_wave, ref_spec, wave, spec)
    ax.scatter(wave_chunks, rv_chunks)
    plt.show()
    exit()


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
    n_chunks = 100
    border = 100

    print(len(ref_wave))

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

        print(f"Calculate RV via CCF for chunk {n_chunk}")
        start = time.time()
        rv, cc = pyasl.crosscorrRV(wave_chunk, spec_chunk,
                                   ref_wave_chunk, ref_spec_chunk,
                                   -0.500, 0.500, 0.0001, skipedge=20)
        maxind = np.argmax(cc)
        rv_chunk = rv[maxind]
        stop = time.time()
        print(f"RV={round(1e3*rv_chunk,3)}m/s. CCF took {round(stop-start,2)}s")

        rv_chunks.append(rv_chunk)
        wave_chunks.append(np.mean(ref_wave_chunk))

    rv_chunks = np.array(rv_chunks)
    wave_chunks = np.array(wave_chunks)

    return wave_chunks, rv_chunks
