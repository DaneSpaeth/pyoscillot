import numpy as np
import matplotlib.pyplot as plt
from dataloader import phoenix_spectrum
from plapy.constants import C
from physics import delta_relativistic_doppler
from scipy.interpolate import CubicSpline
from utils import oversampled_wave_interpol

wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(5000, 6000))

v = 3000
v_c = v / C

fig, ax = plt.subplots(2, 1, figsize=(6.35, 3.5), sharex=True)
ax[0].plot(wave, spec, marker=".", lw=0.5, color="tab:blue", label="Original", markersize=1)
ax[0].set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
ax[1].set_xlabel(r"Wavelength [$\AA$]")

wave_shift = wave + delta_relativistic_doppler(wave, v_c=v_c)
ax[0].plot(wave_shift, spec, color="tab:orange", label="Shifted (not interpolated yet)", marker=".", lw=0.5, markersize=1)

spec_interp = np.interp(wave, wave_shift, spec)

ax[0].plot(wave, spec_interp, color="tab:red", label="Shifted + Interpolated", marker=".", lw=0.5, markersize=1)


spec_oversampled = oversampled_wave_interpol(wave, wave_shift, spec)

ax[0].plot(wave, spec_oversampled, color="tab:green", label="Oversampled, shifted, interpolated", marker=".", lw=0.5, markersize=1)




fine_wave = np.arange(5549, 5561, 0.001)
cs_shifted = CubicSpline(wave_shift, spec)
cs_interp = CubicSpline(wave, spec_interp)
cs_oversampled = CubicSpline(wave, spec_oversampled)


spec_shift_cs = cs_shifted(fine_wave)
spec_interp_cs = cs_interp(fine_wave)
spec_oversampled_cs = cs_oversampled(fine_wave)

ax[1].plot(fine_wave, (spec_interp_cs - spec_shift_cs)/spec_shift_cs, color="tab:red")
ax[1].plot(fine_wave, (spec_oversampled_cs - spec_shift_cs)/spec_shift_cs, color="tab:green")
ax[1].set_ylabel("Relative difference")

ax[0].set_xlim(5550, 5560)
ax[0].legend()
fig.subplots_adjust(hspace=0, left=0.15, right=0.97, top=0.95, bottom=0.15)
plt.savefig("test_interpol.png", dpi=300)

