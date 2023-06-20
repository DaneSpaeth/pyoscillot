from dataloader import phoenix_spectrum
from utils import remove_phoenix_bisector, _gauss_continuum, bisector_on_line
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

def plot_bis_corr_oneline():
    Teff = 4500
    logg = 2.0
    FeH = 0.0
    wave, spec, header = phoenix_spectrum(Teff, logg, FeH, wavelength_range=(3000, 7000))
    spec_corr, spec_corr_norm, spec_norm, poly_fit, delta_v, delta_wave = remove_phoenix_bisector(wave, spec, Teff, logg, FeH)
    
    line = 5705.15
    interval = 0.25
    mask = np.logical_and(wave >= line - interval, wave <= line + interval)

    wv = wave[mask]
    sp = spec_norm[mask]
    dv = delta_v[mask]
    # sp /= np.max(sp)
    
    sp_corr = spec_corr_norm[mask]

    # Fit the width and center for an inital guess
    expected = (line, 0.05, 0.9, 1.0)
    try:
        params, cov = curve_fit(_gauss_continuum, wv, sp, expected)
        width = params[1]
        continuum = params[-1]
    except:
        width = 0.05
        continuum = 1.0

    bis_wave, bis, left_wv, left_sp, right_wv, right_sp = bisector_on_line(wv, 
                                                                        sp, 
                                                                        line,
                                                                        width=width,
                                                                        outlier_clip=0.05,
                                                                        continuum=continuum)
    
    expected = (line, 0.05, 0.9, 1.0)
    try:
        params, cov = curve_fit(_gauss_continuum, wv, sp, expected)
        width = params[1]
        continuum = params[-1]
    except:
        width = 0.05
        continuum = 1.0

    (bis_wave_corr,
     bis_corr,
     left_wv_corr,
     left_sp_corr,
     right_wv_corr,
     right_sp_corr) = bisector_on_line(wv, 
                                       sp_corr, 
                                       line,
                                       width=width,
                                       outlier_clip=0.05,
                                       continuum=continuum)

    
    bis_wave = bis_wave[bis<0.8]
    bis = bis[bis<0.8]
    bis_wave_corr = bis_wave_corr[bis_corr<0.8]
    bis_corr = bis_corr[bis_corr<0.8]
    
    # Convert to velocities
    bis_v = (bis_wave - line) / bis_wave * 3e8
    bis_v_corr = (bis_wave_corr - line) / bis_wave_corr * 3e8
    
    
    
    fig, ax = plt.subplots(1,2, figsize=(30,9))
    ax[0].plot(wv, sp, marker="o", markersize=8, color="black", label="Original PHOENIX")
    ax[0].plot(bis_wave, bis, color="black")
    ax[0].plot(wv, sp_corr, color="tab:red", marker="o", markersize=6, label="Adjusted PHOENIX")
    ax[0].plot(bis_wave_corr, bis_corr, color="tab:red")
    ax[0].legend()
    ax[0].set_xlabel(r"Wavelength [$\AA$]")
    ax[0].set_ylabel("Flux")
    
    mean_v = np.nanmean(bis_v)
    bis_v -= mean_v
    bis_v_corr -= mean_v
    
    ax[1].plot(bis_v, bis, color="black", marker="o", markersize=8, label="Original Bisector")
    ax[1].plot(bis_v_corr, bis_corr, color="tab:red", marker="o", markersize=6, label="Adjusted Bisector")
    ax[1].plot(poly_fit(bis_corr), bis_corr, linewidth=5, label="Fitted Mean Bisector")
    ax[1].plot(dv, sp, linewidth=5, label="Used Delta V", color="green", linestyle="None", marker="o")
    ax[1].set_xlim(-75, 75)
    ax[1].vlines(0, 0, 1, linestyle="dashed", color="black")
    ax[1].set_ylim(0,1)
    ax[1].set_xlabel(r"v [m/s]")
    ax[1].legend()
    
    fig.set_tight_layout(True)
    plt.savefig(f"data/plots/phoenix_bisectors/test_bis_corr{line}.png", dpi=500)
    return wave,spec

if __name__ == "__main__":
    plot_bis_corr_oneline()