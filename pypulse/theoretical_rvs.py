import numpy as np
import matplotlib.pyplot as plt
from plapy.constants import C
from utils import interpolate_to_restframe
from pathlib import Path
from scipy.optimize import least_squares, curve_fit
from scipy.ndimage import gaussian_filter1d
from parse_ini import parse_global_ini
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


def calc_theoretical_results(name, min_wave=None, max_wave=None, ref=False, out_dir=None):
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
        crx, alpha = fit_crx(wave_chunks, rv_chunks)

        if out_dir is not None:
            fig, ax = plt.subplots(1)
            ax.semilogx(wave_chunks, rv_chunks, "bo")
            lin_wave = np.linspace(np.min(wave_chunks),
                                   np.max(wave_chunks))
            ax.semilogx(lin_wave, np.log(lin_wave) * crx +
                        alpha, label=f"CRX={round(crx,2)}")
            ax.set_title(f"BJD={bjd}")
            ax.legend()
            ax.set_ylabel("RV [m/s]")
            ax.set_xlabel("Wavelength [A]")
            plot_dir = out_dir / "RV_chunks"
            if not plot_dir.is_dir():
                plot_dir.mkdir()
            plt.savefig(plot_dir / f"{min_wave}-{max_wave}_{bjd}.pdf")
            plt.close()

        # v_fit = least_square_rvfit(ref_wave, ref_spec, wave, spec)
        # print(
        #     f"v_fit={round(v_fit,3)}, v_theo={round(v_theo,3)}, delta_v={round(v_fit - v_theo,3)}")
        v_fit = np.mean(rv_chunks)
        v_fits.append(v_fit)
        crxs.append(crx)

    return bjds, v_fits, crxs


def calc_rv_chunks(ref_wave, ref_spec, wave, spec):
    """ Calculate RV in chunks which allows to calculate a CRX."""
    n_chunks = 40
    border = 10

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


def save_results(file, bjd, rv, crx):
    """ Save the results to a file."""
    with open(file, "w") as f:
        for b, r, c in zip(bjd, rv, crx):
            f.write(f"{b}    {r}    {c}\n")


def theoretical_main(name):
    """ Calculate and save all the theoretical results."""
    # Create the folder that will contain the plots
    dataroot = parse_global_ini()["datapath"]
    out_dir = dataroot / "fake_spectra" / name / "theoretical_RVs"
    if not out_dir.is_dir():
        out_dir.mkdir()

    # Define the wavelength ranges for CARMENES and HARPS
    max_wave_CARM = 9204
    min_wave_CARM = 5612
    min_wave_HARPS = 3830
    max_wave_HARPS = 6930

    fig, ax = plt.subplots(2, figsize=(16, 9))
    bjd, rv, crx = calc_theoretical_results(
        name, min_wave=min_wave_CARM, max_wave=max_wave_CARM, ref=False, out_dir=out_dir)
    out_file = out_dir / "CARMENES_theoretical.txt"
    save_results(out_file, bjd, rv, crx)

    label = f"Theoretical CARM_VIS ({min_wave_CARM}A,{max_wave_CARM}A)"
    ax[0].plot(bjd, rv - np.median(rv), "go", label=label)
    ax[1].plot(bjd, crx, "go", label=label)

    bjd, rv, crx = calc_theoretical_results(
        name, min_wave=min_wave_HARPS, max_wave=max_wave_HARPS, ref=False, out_dir=out_dir)
    out_file = out_dir / "HARPS_theoretical.txt"
    save_results(out_file, bjd, rv, crx)
    label = f"Theoretical HARPS ({min_wave_HARPS}A,{max_wave_HARPS}A)"
    ax[0].plot(bjd, rv - np.median(rv),
               marker="o", color="purple", label=label, linestyle="None")
    ax[1].plot(bjd, crx,
               marker="o", color="purple", label=label, linestyle="None")

    # Labels and Legends
    ax[0].set_xlabel("BJD")
    ax[0].set_ylabel("RV [m/s]")
    ax[1].set_xlabel("BJD")
    ax[1].set_ylabel("CRX [m/s/Np]")
    ax[0].legend()
    ax[1].legend()
    fig.set_tight_layout(True)
    plt.savefig(out_dir / f"theoretical_RV.pdf")
