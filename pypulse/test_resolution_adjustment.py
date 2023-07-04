import numpy as np
import matplotlib.pyplot as plt
from utils import adjust_resolution, adjust_resolution_dane, _gauss_continuum
from dataloader import phoenix_spectrum
from scipy.optimize import curve_fit
from astropy.convolution import Gaussian1DKernel
from scipy.signal import deconvolve

wave, spec, header = phoenix_spectrum(5800, 4.5, 0.0, wavelength_range=(6000, 6500))

wave_air = wave / (1.0 + 2.735182E-4 + 131.4182 / wave**2 + 2.76249E8 / wave**4)
wave = wave_air


# wave = np.linspace(4999, 5001, 10000)
# spec = _gauss_continuum(wave, 5000, 0.01, 0.9, 1.0)



spec_res = adjust_resolution(wave, spec, R=100000, w_sample=100)
spec_res_dane  = adjust_resolution_dane(wave, spec, R=499999)


line = 6301.5
interval = 0.2
mask = np.logical_and(wave > line - interval, wave < line + interval)
wave = wave[mask]
spec = spec[mask]
spec_res = spec_res[mask]
spec_res_dane = spec_res_dane[mask]

norm = np.max(spec)
spec /= norm
spec_res /= norm
spec_res_dane /= norm


for sp in (spec, spec_res, spec_res_dane):
    params, cov = curve_fit(_gauss_continuum, wave, sp, p0=(line, 0.25, 0.3, 1.0))
    center = params[0]
    sigma = params[1]
    fwhm = 2*np.sqrt(2*np.log(2))*sigma
    
    R = center / fwhm
    print(R)


plt.plot(wave, spec)
plt.plot(wave, spec_res)
# plt.plot(wave, _gauss_continuum(wave, *params), color="black")
plt.plot(wave, spec_res_dane)


plt.savefig("dbug.png")