import numpy as np
import matplotlib.pyplot as plt
from utils import adjust_resolution, adjust_resolution_per_pixel, _gauss_continuum, neg_gaussian
from dataloader import phoenix_spectrum
from scipy.optimize import curve_fit
from astropy.convolution import Gaussian1DKernel
from scipy.signal import deconvolve
import pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline, interp1d
plt.rcParams['axes.formatter.useoffset'] = False


LB_root = Path("/data/dspaeth/pypulse_data/resolution_tests")
df = pd.read_csv(LB_root / "LB_R700000.csv")

wave_LB_700000 = np.array(df["x"])
spec_LB_700000 = np.array(df[" y"])

df = pd.read_csv(LB_root / "LB_R100000.csv")

wave_LB_100000 = np.array(df["x"])
spec_LB_100000 = np.array(df[" y"])

fig, ax = plt.subplots(1, figsize=(6.35, 3.5))

lin_wave = np.linspace(wave_LB_700000.min(), wave_LB_700000.max(), 1000)

cs_700000 = interp1d(wave_LB_700000[np.argsort(wave_LB_700000)], spec_LB_700000[np.argsort(wave_LB_700000)])

ax.plot(lin_wave, cs_700000(lin_wave), color="black", lw=1, label="R=700000 (Löhner-Böttcher et al. 2018)")

ax.plot(wave_LB_100000, spec_LB_100000, color="tab:blue", lw=1, label="R=100000 (Löhner-Böttcher et al. 2018)")




lin_spec_70000 = cs_700000(lin_wave)


spec_res_dane  = adjust_resolution(lin_wave, lin_spec_70000, R=100000)
ax.plot(lin_wave, spec_res_dane, color="tab:orange", lw=1, label="R=100000 (pypulse)")

spec_res_per_pixel = adjust_resolution_per_pixel(lin_wave, lin_spec_70000, R=100000)
ax.plot(lin_wave, spec_res_dane, color="tab:red", lw=1, label="R=100000 (pypulse, per pixel)")

ax.legend()
ax.set_ylabel("Normalized Intensity")
ax.set_xlabel(r"Air wavelength [$\AA$]")
ax.set_xlim(6301.35, 6301.67)
print(np.min(lin_spec_70000))
print(np.min(spec_res_dane))
fig.set_tight_layout(True)
plt.savefig("PhD_plots/resolution_test_LB.png", dpi=600)
