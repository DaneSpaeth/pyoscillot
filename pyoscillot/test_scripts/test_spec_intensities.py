import numpy as np
import matplotlib.pyplot as plt
from constants import C
from utils import interpolate_to_restframe
from pathlib import Path
from scipy.optimize import least_squares, curve_fit
from scipy.ndimage import gaussian_filter1d
from cfg import parse_global_ini
import time


def _read_in_arrays(name, min_wave=None, max_wave=None, ref=False):
    """ Read in all arrays for given simulation name."""

    dataroot = parse_global_ini()["datapath"]
    sim = dataroot / "fake_spectra" / name / "RAW"
    wave_files = list(Path(sim).glob("wave_*.npy"))
    spec_files = list(Path(sim).glob("spec_*.npy"))

    if not len(wave_files):
        print(f"No files could be found in {sim}")

    waves = []
    specs = []
    vs = []
    bjds = []

    for wave_file, spec_file in zip(sorted(wave_files),
                                    sorted(spec_files)):
        old = False
        if old:
            v = float(spec_file.name.split("spec_")
                      [-1].split(".npy")[0].split("_")[0])
            bjd = float(spec_file.name.split("spec_")
                        [-1].split(".npy")[0].split("_")[-1])
        else:
            v = float(spec_file.name.split("spec_")
                      [-1].split(".npy")[0].split("_")[-1])
            bjd = float(spec_file.name.split("spec_")
                        [-1].split(".npy")[0].split("_")[0])
        if ref:
            if v == 0:
                ref_wave = np.load(wave_file)
                ref_spec = np.load(spec_file)
                if min_wave is not None and max_wave is not None:
                    ref_wave, ref_spec = _cut_to_waverange(ref_wave, ref_spec,
                                                           min_wave, max_wave)
                continue

        wave = np.load(wave_file)

        spec = np.load(spec_file)
        if min_wave is not None and max_wave is not None:
            wave, spec = _cut_to_waverange(wave, spec,
                                           min_wave, max_wave)

        waves.append(wave)
        specs.append(spec)
        vs.append(v)
        bjds.append(bjd)

    if not ref:
        ref_wave = waves[0]
        ref_spec = specs[0]

    return (ref_wave, ref_spec,
            np.array(waves), np.array(specs),
            np.array(vs), np.array(bjds))


def _cut_to_waverange(wave, spec, min_wave, max_wave):
    """ Convenience function to cut to a wavelength range."""
    wave_mask_low = wave > min_wave
    wave_mask_high = wave < max_wave
    wave_mask = np.logical_and(wave_mask_low, wave_mask_high)
    wave = wave[wave_mask]
    spec = spec[wave_mask]

    return wave, spec


def calc_theoretical_results(name, min_wave=None, max_wave=None, ref=False, plot=True):
    """ Calculate all theoretical results, i.e. RVs, CRX, dLW

        :param str name: Name of Simulation
    """

    ref_wave, ref_spec, waves, specs, vs, bjds = _read_in_arrays(name,
                                                                 min_wave=min_wave,
                                                                 max_wave=max_wave,
                                                                 ref=ref)

    v_fits = []
    crxs = []
    for wave, spec, v_theo, bjd in zip(waves, specs, vs, bjds):
        wave_chunks, rv_chunks, chunk_sizes = calc_rv_chunks(
            ref_wave.copy(), ref_spec.copy(), wave.copy(), spec.copy())
        fig, ax = plt.subplots()
        ax.semilogx(wave_chunks, rv_chunks, "bo")
        crx, alpha = fit_crx(wave_chunks, rv_chunks)
        lin_wave = np.linspace(np.min(wave_chunks),
                               np.max(wave_chunks))

        ax.semilogx(lin_wave, np.log(lin_wave) * crx +
                    alpha, label=f"CRX={round(crx,2)}")
        ax.set_title(f"BJD={bjd}")
        ax.legend()
        ax.set_ylabel("RV [m/s]")
        ax.set_xlabel("Wavelength [A]")
        plt.savefig(f"tmp_plots/{min_wave}-{max_wave}_{bjd}.pdf")
        plt.close()
        crxs.append(crx)

        # v_fit = least_square_rvfit(ref_wave, ref_spec, wave, spec)
        # print(
        #     f"v_fit={round(v_fit,3)}, v_theo={round(v_theo,3)}, delta_v={round(v_fit - v_theo,3)}")
        v_fit = np.mean(rv_chunks)
        v_fits.append(v_fit)

    if plot:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(bjds, v_fits, "bo")
        ax[1].plot(bjds, crxs, "bo")
        ax[0].set_xlabel("BJD")
        ax[0].set_ylabel("RV [m/s]")
        ax[1].set_xlabel("BJD")
        ax[1].set_ylabel("CRX [m/s/Np]")
        plt.savefig(f"{name}_theoretical.pdf")
    else:
        return bjds, v_fits, crxs


def calc_rv_chunks(ref_wave, ref_spec, wave, spec):
    """ Calculate RV in chunks which allows to calculate a CRX."""
    n_chunks = 30
    border = 0

    chunk_size_px = int((len(ref_wave) - 2 * border) / n_chunks)

    rv_chunks = []
    wave_chunks = []
    chunk_sizes = []
    for n_chunk in range(n_chunks):
        idx_range = slice(border + n_chunk * chunk_size_px,
                          (n_chunk + 1) * chunk_size_px)
        ref_wave_chunk = ref_wave[idx_range]
        ref_spec_chunk = ref_spec[idx_range]
        wave_chunk = wave[idx_range]
        spec_chunk = spec[idx_range]

        print(f"Chunk size={len(wave_chunk)}")

        # try to fit the continuum
        filter_width = 1000
        smoothed_ref_spec_chunk = gaussian_filter1d(
            ref_spec_chunk, sigma=filter_width)
        smoothed_ref_spec_chunk = smoothed_ref_spec_chunk * \
            np.median(ref_spec_chunk) / np.median(smoothed_ref_spec_chunk)
        res = np.polynomial.polynomial.Polynomial.fit(
            ref_wave_chunk, smoothed_ref_spec_chunk, 1)
        coef = res.convert().coef

        ref_spec_chunk = ref_spec_chunk * \
            np.median(spec_chunk) / np.median(ref_spec_chunk)
        continuum = coef[0] + coef[1] * \
            ref_wave_chunk  # + coef[2] * lin_wave**2
        ref_spec_chunk /= continuum
        spec_chunk /= continuum

        chunk_sizes.append(wave_chunk[-1] - wave_chunk[0])

        print(f"Calculate RV via least_square_rvfit for chunk {n_chunk}")
        start = time.time()
        rv_chunk = least_square_rvfit(
            ref_wave_chunk, ref_spec_chunk, wave_chunk, spec_chunk)
        stop = time.time()

        # print(f"RV={round(rv_chunk, 3)}m/s. Fit took {round(stop-start,2)}s")

        rv_chunks.append(rv_chunk)
        wave_chunks.append(np.mean(ref_wave_chunk))

    rv_chunks = np.array(rv_chunks)
    wave_chunks = np.array(wave_chunks)
    chunk_sizes = np.array(chunk_sizes)

    # Add sigma clipping
    sigma_clip = False
    if sigma_clip:
        clip_mask = np.abs(rv_chunks - np.mean(rv_chunks)
                           ) < 1 * np.std(rv_chunks)
        rv_chunks = rv_chunks[clip_mask]
        wave_chunks = wave_chunks[clip_mask]
        chunk_sizes = chunk_sizes[clip_mask]

    return wave_chunks, rv_chunks, chunk_sizes


def least_square_rvfit(ref_wave, ref_spec, wave, spec):
    def func(v_shift, ref_wave, ref_spec, wave, spec):
        shift_wave = wave * 1 / (1 - v_shift[0] / C)
        # Interpolate back to the ref (and therefore simulated wavelength grid)
        interpol_spec = interpolate_to_restframe(shift_wave,
                                                 spec,
                                                 ref_wave)

        return interpol_spec - ref_spec

    x0 = np.array([10])
    res = least_squares(func, x0, method="lm",
                        args=(ref_wave, ref_spec, wave, spec))

    return -res.x[0]


def fit_crx(wave, rv):
    """ Fit the chromatic index."""
    def func(log_wave, alpha, crx):
        return alpha + log_wave * crx
    popt, pcov = curve_fit(func, np.log(wave), rv)

    alpha, crx = popt

    print(f"CRX={crx}")

    return crx, alpha


if __name__ == "__main__":
    max_wave_CARM = 10000
    # min_wave_CARM = 5612

    min_wave_CARM = 6000

    name = "TEST_PLANET_SPEC_NEW"
    fig, ax = plt.subplots(2)
    bjds, v_fits, crxs = calc_theoretical_results(
        name, min_wave=min_wave_CARM, max_wave=max_wave_CARM, ref=False, plot=False)

    label = f"Theoretical CARM_VIS ({min_wave_CARM}A,{max_wave_CARM}A)"
    ax[0].plot(bjds, v_fits, "go", label=label)
    ax[1].plot(bjds, crxs, "go", label=label)
    print(crxs)
    ax[0].set_xlabel("BJD")
    ax[0].set_ylabel("RV [m/s]")
    ax[1].set_xlabel("BJD")
    ax[1].set_ylabel("CRX [m/s/Np]")

    plt.savefig(f"{name}_theoretical.pdf")
