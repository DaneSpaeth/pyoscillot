import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import bisector_new
from constants import C
from star import GridStar

ARRAY_ROOT = Path("/home/dane/Documents/PhD/pyoscillot/pyoscillot/arrays")


def bisector_plot_Hatzes():
    """ Cretate a bisector plot as in Hatzes."""
    phases = np.array([0.25, 0.5, 0.75])
    ms = np.array([2, 4, 6])

    fig, ax = plt.subplots(1, 3, figsize=(12, 9))
    for idx, m in enumerate(ms):
        folder = ARRAY_ROOT / f"pulse_real_line_m{m}_k015"
        for phase in phases:
            if not idx:
                wave = np.load(folder / "wave.npy")
            spec = np.load(folder / f"{phase}.npy")
            bis_wave, bis, center = bisector_new(wave, spec)
            line = 6254.29
            vs = (bis_wave - line) / line * C
            vs += 280
            ax[idx].plot(vs, bis, label=f"Phase {phase-0.25}")
        ax[idx].legend()
        ax[idx].set_xlabel("V [m/s]")
        ax[idx].set_ylabel("Relative Flux")
        ax[idx].set_title(f"Sectoral G-mode (k=1.2), l={m}, m={-m}")

    plt.show()


def create_g_mode_arrays():
    """ Create the arrays for the g-mode."""
    phases = np.array([0.25, 0.5, 0.75])
    ms = np.array([2, 4, 6])

    for m in ms:
        star = GridStar(N_star=50, N_border=1, vsini=3000)
        for idx, p in enumerate(phases):
            star.add_pulsation(l=m, m=-m, k=1.2, phase=p)
            line = 6254.29
            wave, spec = star.calc_spectrum(line, line, mode="oneline")

            folder = ARRAY_ROOT / f"pulse_real_line_m{m}_k12"
            if not folder.is_dir():
                folder.mkdir()
            if not idx:
                np.save(folder / "wave.npy", wave)
            np.save(folder / f"{p}.npy", spec)
            plt.imshow(star.pulsation.real, origin="lower",
                       vmin=-200, vmax=200, cmap="seismic")
            plt.savefig(folder / f"{round(p, 4)}.pdf")
            plt.close()


if __name__ == "__main__":
    # create_g_mode_arrays()
    bisector_plot_Hatzes()
