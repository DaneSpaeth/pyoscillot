import numpy as np
from pathlib import Path
from cfg import parse_global_ini
from dataloader import phoenix_spectrum, phoenix_wave


def calc_2nd_derivs_spline(x: list, y: list, yp1=np.inf, ypn=np.inf):
    """Calculates the 2nd derivatives for a spline.
    Python conversion from the C++ code in chapter "Cubic Spline Interpolation" in the
    "Numerical Recipes in C++, 2nd Edition".
    
    Taken from spexxy -> Husser
    https://github.com/thusser/spexxy/blob/master/spexxy/interpolator/spline.py

    Args:
        x: Input x values.
        y: Input y values.
        yp1: First derivative at point 0. If set to np.inf, use natural boundary condition and set 2nd deriv to 0.
        ypn: First derivative at point n-1. np.inf means the same as for yp1.

    Returns:
        Second derivates for all points given by x and y.
    """

    # get number of elements
    n = len(x)

    # create arrays for u and 2nd derivs
    if hasattr(y[0], '__iter__'):
        y2 = np.zeros((n, len(y[0])))
        u = np.zeros((n, len(y[0])))
    else:
        y2 = np.zeros((n))
        u = np.zeros((n))

    # derivatives for point 0 given?
    if not np.isinf(yp1):
        y2[0] += -0.5
        u[0] += (3. / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)

    # decomposition loop of the tridiagonal algorithm
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p

    # derivatives for point n-1 given?
    if not np.isinf(ypn):
        qn = 0.5
        un = (3. / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))
        y2[n - 1] += (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.)

    # backsubstitution loop of the tridiagonal algorithm
    for k in range(n - 2, 0, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]

    # finished
    return y2


def precompute_second_derivative_grid(logg=2.0, feh=0.0):
    """ Precompute a second derivative array"""
    # Get the output directory
    global_dict = parse_global_ini()
    out_root = global_dict["datapath"]
    
    teffs = np.array(range(2300, 7100, 100))
    specs = []
    for teff in teffs:
        wave, spec, header = phoenix_spectrum(teff, logg, feh, wavelength_range=False)
        specs.append(spec)
        
    
    
    specs = np.array(specs)
    second_derivatives = np.zeros_like(specs)
    for idx in range(specs.shape[1]):
        print(f"{idx}/{specs.shape[1]}")
        second_derivative = calc_2nd_derivs_spline(teffs, specs[:,idx])
        second_derivatives[:,idx] = second_derivative
    
    
    directory = "phoenix_second_derivatives"
    filename = f"logg{logg:.1f}_feh{feh:.1f}.npy"
    np.save(out_root / directory / filename, second_derivatives)
    
def cubic_spline_interpolation(T, 
                               T_low, 
                               T_high, 
                               spec_low,
                               spec_high,
                               second_deriv_low,
                               second_deriv_high):
    """ Cubic spline interpolation following Press 2007 
    
        (and Husser+2016 but they have slightly weird parameters,
        Husser+2012 has the same params as Press
    """
    A = (T_high - T) / (T_high - T_low)
    B = 1 - A
    C = 1 / 6 * (A**3 - A) * (T_high - T_low)**2
    D = 1 / 6 * (B**3 - B) * (T_high - T_low)**2 
    
    spec_interpol = (A * spec_low +
                     B * spec_high +
                     C * second_deriv_low +
                     D * second_deriv_high)
    
    return spec_interpol


def interpolate_on_temperature(T, ref_wave, ref_spectra, logg, feh, mu=1.0):
    """ Interpolate on a temperature grid"""
    print("Run cubic spline interpolation")
    # First load the second derivatives
    global_dict = parse_global_ini()
    root = global_dict["datapath"]
    directory = "phoenix_second_derivatives"
    filename = f"logg{logg:.1f}_feh{feh}.npy"
    second_derivatives = np.load(root / directory / filename)
    
    if T in ref_spectra.keys():
        return ref_spectra[T][mu]
    
    # determine the adjacent temperatures in the phoenix grid
    T_low = int(np.floor(T/100)*100)
    T_high = int(np.ceil(T/100)*100)
    
    # In case you are very close to a full 100
    if T_high == T_low:
        T_high += 100
    
    # get the spectra
    spec_low = ref_spectra[T_low][mu]
    spec_high = ref_spectra[T_high][mu]
    
    # Now get the second derivatives
    idx_low = np.argmax(np.array(range(2300, 7100, 100)) == T_low)
    second_derivative_low = second_derivatives[idx_low, :] 
    idx_high = np.argmax(np.array(range(2300, 7100, 100)) == T_high)
    second_derivative_high = second_derivatives[idx_high, :]
    
    # Now we still need to cut the second_derivate_array to the same wave
    # regime as the waves given by the ref wave
    # For that first load another PHOENIX spectrum only for the wave
    ph_wave = phoenix_wave()
    wave_min = ref_wave[0]
    wave_max = ref_wave[-1]
    mask = np.logical_and(ph_wave >= wave_min, ph_wave <= wave_max)
    ph_wave = ph_wave[mask]
    # The two wave arrays should npw be identical
    assert (ph_wave == ref_wave).all()
    
    second_derivative_low = second_derivative_low[mask]
    second_derivative_high = second_derivative_high[mask]
    
    # Now we can run the actual interpolation
    spec_interpol = cubic_spline_interpolation(T, 
                                               T_low, 
                                               T_high,
                                               spec_low,
                                               spec_high,
                                               second_derivative_low,
                                               second_derivative_high)
    return spec_interpol
    
if __name__ == "__main__":
    # from utils import get_ref_spectra
    # import matplotlib.pyplot as plt
    # wave, ref_spectra, ref_headers = get_ref_spectra(np.array([4550]), 2.0, 0.0)
    # spec_interpol = interpolate_on_temperature(4550, wave, ref_spectra, 2.0, 0.0, 1.0)
    
    # fig, ax = plt.subplots(1, figsize=(6.35, 3.5))
    # markersize = 1
    # ax.plot(wave, ref_spectra[4500][1.0], marker="^", markersize=markersize, label="T=4500 K (PHOENIX)")
    # ax.plot(wave, ref_spectra[4600][1.0], alpha=0.7, marker="v", markersize=markersize, label="T=4600 K (PHOENIX)")
    # ax.plot(wave, spec_interpol, alpha=0.7, marker="o", markersize=markersize, label="T=4550 K (interpolated)")
    # ax.set_xlim(5499, 5515)
    # ax.set_xlabel(r"Wavelength [$\AA$]")
    # ax.set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
    # ax.set_ylim(0, ax.get_ylim()[1])
    # ax.legend()
    # fig.set_tight_layout(True)
    # plt.savefig("temp_interpolation.png", dpi=300)
    precompute_second_derivative_grid(4.5, 0.0)