import numpy as np
from pathlib import Path
from cfg import parse_global_ini
from dataloader import phoenix_spectrum


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
    filename = f"logg{logg}_feh{feh}.npy"
    np.save(out_root / directory / filename, second_derivatives)


        