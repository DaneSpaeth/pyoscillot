import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial
import pandas as pd
from cfg import conf_dict

CB_Gray_dir = conf_dict["datapath"] / "CB_Gray05"

def Gray_CB_model_names():
    """ Return a list of all Gray CB models."""
    all_files = CB_Gray_dir.glob("*.csv")
    models = [f.stem for f in all_files]
    models = sorted(models)
    return models

def Gray_CB_model(model, debug_plot=False):
    """ Fit a simple Pollux model. There is no deconvolution applied yet."""
    
    # Take data from Gray05 table 4
    model_file = CB_Gray_dir / f"{model}.csv"
    if not model_file.is_file():
        print(f"{model_file} does not exist as a CB model.")
        print(f"model must be in {Gray_CB_model_names()}")
        exit()
        
    df = pd.read_csv(model_file)
    flux = np.array(df["flux"])
    bis = np.array(df["bis"])
    bise = np.array(df["bise"])
    
    assert len(flux) == len(bise)
    assert len(flux) == len(bis)
    

    pfit = np.polynomial.Polynomial.fit(flux, bis, w=1/bise, deg=4)


    if debug_plot:
        lin_flux = np.linspace(0, 1, 100)

        fig, ax = plt.subplots(1, figsize=(7.16, 4.0275))
        ax.errorbar(bis, flux, xerr=bise, linestyle="None", marker=".", color="tab:blue")
        ax.plot(pfit(lin_flux), lin_flux, color="tab:red")
        ax.set_xlabel("Velocity [m/s]")
        ax.set_ylabel("Relative Flux")
        fig.set_tight_layout(True)
        plt.savefig(CB_Gray_dir / "plots" / f"CB_{model}_fit.png", dpi=600)
        plt.close()
    
    
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
    """ Returnn a simple 2nd order polynomial model for the ngc4349-127 CB"""
    
    # Values were fitted from mean BIS of NGC4349-127
    poly_fit = Polynomial([-2.71849303, -275.15450218, 368.28014225])
    
    mu_dict = {1.0:poly_fit}
    
    return mu_dict

    
    

    
    

if __name__ == "__main__":
    models = Gray_CB_model_names()
    print(models)
    exit()
    for model in models:
        print(model)
        Gray_CB_model(model, debug_plot=True)