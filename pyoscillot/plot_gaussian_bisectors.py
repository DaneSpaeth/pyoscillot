from star import GridSpectrumSimulator
from utils import bisector_on_line, bisector, adjust_resolution
import matplotlib.pyplot as plt
import numpy as np
from PyAstronomy import pyasl

print("Test")
Teff = 4500
logg = 2
feh = 0.0
fig, ax = plt.subplots(1, 3, sharey=True)
for idx, l in enumerate((2, 4, 6)): 
    for t, color, label in zip((150, 300, 450),
                               ("tab:blue", "tab:orange", "tab:red"),
                               (r"$\phi=$0.25", r"$\phi=$0.5", r"$\phi=$0.75")):
        line = 6254.29
        # line = 6301.5008
        # line = 6432.62
        # line = pyasl.airtovac2(line)
        star = GridSpectrumSimulator(N_star=120, Teff=Teff, logg=logg, feh=feh, limb_darkening=True, convective_blueshift=False, v_macro=0, v_rot=3046, inclination=80)
        star.add_pulsation(t=t, l=l, m=-l, v_p=400, k=0.15, T_var=0)
        wave, spec, _ = star.calc_spectrum(min_wave=line-10, max_wave=line+10, mode="phoenix", skip_V_flux=True)
        
        sp_conv = adjust_resolution(wave, spec, R=120000)
        # sp_conv = spec
        
        
        
        mask = np.logical_and(wave > line-0.3, wave < line+0.3)
        wave = wave[mask]
        spec = sp_conv[mask]
        
        spec = spec / np.max(spec)

        # plt.plot(wave, spec)
        # plt.savefig("dbug.png")
        # exit()
        bisector_wave, bisector_flux, left_wave, left_spec, right_wave, right_spec = bisector_on_line(wave, spec, line, 1.0, expected_width=0.15, num_widths=3.0, continuum_lim=0.94, low_lim=0.01, nr_points=75)
        # bisector_wave, bisector_flux = bisector(wave, spec)
        bisectors_vs = (bisector_wave - line) / line * 3e8
        
        # ax[0].plot(wave, spec)
        ax[idx].plot(bisectors_vs, bisector_flux, color=color, label=label)
    ax[idx].legend()
    ax[idx].set_xlabel("Velocity [m/s]")
    ax[idx].set_ylabel("Bisector Flux")
fig.set_tight_layout(True)
plt.savefig("dbug.png")