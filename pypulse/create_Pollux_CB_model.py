import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def simple_Pollux_CB_model():
    """ Fit a simple Pollux model. There is no deconvolution applied yet."""
    # df = pd.read_csv("/home/dspaeth/pypulse/data/CB_Pollux/Gray14.csv")
    
    # Take data from Gray05 table 4
    flux = np.arange(0.92, 0.24, -0.02)
    bis = np.array([28, 26, 20, 12, 6, 
                    0, -5, -9, -12, -14,
                    -15,-16, -16, -15, -14,
                    -12,-10, -8, -5, -1,
                    4,9, 14, 21, 29,
                    37,47, 59, 73, 90,
                    111,136, 169, 211])
    
    bise = np.array([4, 3, 3, 2, 2,
                     2, 2, 2, 1, 1, 
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1,
                     1, 2, 2, 2, 2, 
                     3, 3, 4, 5])
    assert len(flux) == len(bise)
    assert len(flux) == len(bis)
    # print(flux)
    # exit()
    

    pfit = np.polynomial.Polynomial.fit(flux, bis, w=1/bise, deg=4) * 3


    lin_flux = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(1, figsize=(7.16, 4.0275))

    ax.errorbar(bis, flux, xerr=bise, linestyle="None", marker=".", color="tab:blue")
    ax.plot(pfit(lin_flux), lin_flux, color="tab:red")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Relative Flux")
    fig.set_tight_layout(True)
    plt.savefig("CB_Pollux_fit.png", dpi=600)
    
    
    mu_dict = {1.0:pfit}
    return mu_dict

def simple_alpha_boo_CB_model():
    flux = np.arange(0.9, 0.18, -0.02)
    bis = np.array([201, 196, 185, 171, 146,
                    140, 124, 109, 95, 81,
                    69, 57, 46, 37, 28,
                    19, 12, 5, -1, -7,
                    -11, -15, -18, -21, -23,
                    -23, -23, -22, -19, -14,
                    -7, 3, 16, 37, 48,
                    32])
    
    bise = np.array([3, 2, 2, 2, 2,
                     1, 1, 1, 1, 1,
                     1, 1, 1, 1, 0.5,
                     0.5, 0.5, 0.5, 0.5, 0.5,
                     0.5, 0.5, 0.5, 0.5, 0.5,
                     0.5, 1, 1, 1, 1,
                     1, 1, 1, 1, 2, 
                     23])
    assert len(flux) == len(bise)
    assert len(flux) == len(bis)
    # print(flux)
    # exit()
    

    pfit = np.polynomial.Polynomial.fit(flux, bis, w=1/bise, deg=4)


    lin_flux = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(1, figsize=(7.16, 4.0275))

    ax.errorbar(bis, flux, xerr=bise, linestyle="None", marker=".", color="tab:blue")
    ax.plot(pfit(lin_flux), lin_flux, color="tab:red")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Relative Flux")
    fig.set_tight_layout(True)
    plt.savefig("CB_alpha_boo_fit.png", dpi=600)
    
    
    mu_dict = {1.0:pfit}
    return mu_dict

def deconvolved_Pollux_CB_model():
    """ Fit a simple Pollux model. There is no deconvolution applied yet."""
    df = pd.read_csv("/home/dspaeth/pypulse/data/CB_Pollux/Fe6253.csv")
    
    wv = np.array(df["wave"])
    sp = np.array(df[" flux"])
    
    
    line = 6253.0
    expected = (line, 0.05, 0.9, 1.0)
    
    params, cov = curve_fit(_gauss_continuum, wv, sp, expected)
    center = params[0]
    width = params[1]
    continuum = params[-1]

    bis_wave, bis, left_wv, left_sp, right_wv, right_sp = bisector_on_line(wv, 
                                                                        sp, 
                                                                        center,
                                                                        width=width,
                                                                        outlier_clip=0.5,
                                                                        continuum=1.0)

    # Convert to velocities
    fig, ax = plt.subplots(1,2, figsize=(7.16, 4.0275))
    bis_v = (bis_wave - line) / bis_wave * 3e8
    bis_v -= bis_v[~np.isnan(bis_v)][0]
    
    df = pd.read_csv("/home/dspaeth/pypulse/data/CB_Pollux/Gray14.csv")

    Gray14_flux = np.array(df[" y"])
    Gray14_v = np.array(df["x"])
    Gray14_v -= Gray14_v[-1]
    
    ax[0].plot(wv, sp, marker="o")
    ax[0].plot(bis_wave, bis)
    
    ax[1].plot(bis_v, bis, marker=".")
    ax[1].plot(Gray14_v, Gray14_flux, marker=".")
    # plt.savefig("dbug.png")
    
    # Now try to deconvolve
    # Save original wavelength grid and units
    w_grid = wv
    w_sample = 1
    R = 100000

    # Generate logarithmic wavelength grid for smoothing
    w_logmin = np.log10(np.nanmin(w_grid))
    w_logmax = np.log10(np.nanmax(w_grid))

    n_w = np.size(w_grid) * w_sample
    w_log = np.logspace(w_logmin, w_logmax, num=n_w)

    # Find stddev of Gaussian kernel for smoothing
    R_grid = (w_log[1:-1] + w_log[0:-2]) / (w_log[1:-1] - w_log[0:-2]) / 2
    sigma = np.median(R_grid) / R
    
    gauss = Gaussian1DKernel(stddev=sigma)
    
    
    print(sigma)
    print(R_grid)
    
def simple_ngc4349_CB_model():
    """ Retunrn a simple 2nd order polynomial model for the ngc4349-127 CB"""
    
    # Values were fitted from mean BIS of NGC4349-127
    poly_fit = Polynomial([-2.71849303, -275.15450218, 368.28014225])
    
    mu_dict = {1.0:poly_fit}
    
    return mu_dict

    
    

    
    

if __name__ == "__main__":
    simple_alpha_boo_CB_model()