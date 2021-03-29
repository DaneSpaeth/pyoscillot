import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from dataloader import phoenix_spectrum
from physics import planck_ratio, planck, get_interpolated_spectrum
from astropy.convolution import convolve_fft
from astropy.convolution import Gaussian1DKernel

wave, spec, h = phoenix_spectrum()
print(np.abs(wave[0] - wave[-1]) / len(wave))
