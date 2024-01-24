from dataloader import phoenix_spectrum, Zhao_bis_polynomials
from utils import remove_phoenix_bisector, _gauss_continuum, bisector_on_line,add_bisector
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import copy
from star import GridSpectrumSimulator


def plot_bis_corr_star():
    import matplotlib.pyplot as plt
    from utils import bisector_on_line, adjust_resolution
    from dataloader import harps_template
    from pathlib import Path
    def plot_line_and_bis(wave, spec, ax, color="tab:blue", label="", line=5728.642,):
        
        interval = 0.25
        mask = np.logical_and(wave >= line - interval, wave <= line + interval)
        wave_line = wave[mask] 
        spec_line = spec[mask] / np.max(spec[mask])
        
        # Fit the width and center for an inital guess
        expected = (line, 0.05, 0.9, 1.0)
        try:
            params, cov = curve_fit(_gauss_continuum, wave_line, spec_line, expected)
            width = params[1]
            continuum = params[-1]
        except:
            width = 0.05
            continuum = 1.0

        bis_wave, bis, left_wv, left_sp, right_wv, right_sp = bisector_on_line(wave_line ,
                                                                                spec_line, 
                                                                                line,
                                                                                width=width,
                                                                                outlier_clip=0.05,
                                                                                continuum=continuum)
        bis_v  = (bis_wave - line) / bis_wave * 3e8
        
        
        ax[0].plot(wave_line, spec_line, color=color, label=label)
        ax[0].plot(bis_wave, bis, color=color,)
        ax[1].plot(bis_v, bis, color=color, label=label)
    
    N_star = 150
    v_rot = 3900
    inclination = 60
    line = 6253.57
    star = GridSpectrumSimulator(N_star=N_star, N_border=1, Teff=4500, logg=3.0, v_rot=v_rot, limb_darkening=False, inclination=inclination, convective_blueshift=False)
    star_cb = GridSpectrumSimulator(N_star=N_star, N_border=1, Teff=4500, logg=3.0, v_rot=v_rot, limb_darkening=False, inclination=inclination, convective_blueshift=True)
    wave, spec, v_total = star.calc_spectrum(min_wave=3550, max_wave=7150)
    wave_cb, spec_cb, v_total = star_cb.calc_spectrum(min_wave=3550, max_wave=7150)
    fig, ax = plt.subplots(1, 2, figsize=(16,9))
    # # plt.plot(wave[mask], spec[mask])
    plot_line_and_bis(wave, spec, ax, color="tab:blue", line=line, label="Normal BIS")
    plot_line_and_bis(wave_cb, spec_cb, ax, color="tab:red", line=line, label="Changed BIS")
    
    # Simply to get the wavelength regime right
    spec_harps, wave_harps, blaze = harps_template(spec_filename="/home/dspaeth/pypulse/data/fake_spectra/NGC4349_Test88/HARPS/fits/ADP.2005-02-03T06:48:39.967_e2ds_A.fits")
    order = 62
    # interval = 0.25
    wave_harps = wave_harps[order]
    wavelength_regime = (np.min(wave_harps), np.max(wave_harps))
    spec_harps = spec_harps[order]
    plot_line_and_bis(wave_harps, spec_harps, ax, color="tab:orange", line=line, label="Normal BIS Harps")
    
    mask_harps_order = np.logical_and(wave >= wavelength_regime[0],
                                      wave <= wavelength_regime[1])
    spec_R = adjust_resolution(wave[mask_harps_order], spec[mask_harps_order],
                                        R=115000, w_sample=20)
    wave_R = wave[mask_harps_order]
    plot_line_and_bis(wave_R, spec_R, ax, color="tab:green", line=line, label="Normal BIS, R=115000")
    
    
    mask_harps_order = np.logical_and(wave_cb >= wavelength_regime[0],
                                      wave_cb <= wavelength_regime[1])
    spec_R_cb = adjust_resolution(wave_cb[mask_harps_order], spec_cb[mask_harps_order],
                                        R=115000, w_sample=20)
    wave_R_cb = wave_cb[mask_harps_order]
    plot_line_and_bis(wave_R_cb, spec_R_cb, ax, color="purple", line=line, label="Changed BIS, R=115000")
    
    spec, wave, blaze = harps_template(spec_filename="/home/dspaeth/pypulse/data/fake_spectra/NGC4349_Test89/HARPS/fits/ADP.2005-02-03T06:48:39.967_e2ds_A.fits")
    order = 62
    interval = 0.25
    wave = wave[order]
    spec = spec[order]
    plot_line_and_bis(wave, spec, ax, color="cyan", line=line, label="Changed BIS HARPS")
    
    ax[0].legend()
    ax[0].set_xlabel(r"Wavelength [\AA]")
    ax[0].set_ylabel("Flux")
    ax[1].set_xlabel("Doppler Velocity [m/s]")
    fig.set_tight_layout(True)
    plt.savefig("/home/dspaeth/pypulse/data/plots/BIS_tests/BIS_N=150_vrot3900_inclination60.png")

def plot_bis_corr_oneline():
    Teff = 4500
    logg = 3.0
    FeH = 0.0
    wave, spec, header = phoenix_spectrum(Teff, logg, FeH, wavelength_range=(3500, 7000))
    
    
    # Create a BIS removed version
    (spec_corr, 
     spec_corr_norm, 
     spec_norm, 
     poly_fit, 
     delta_v, 
     delta_wave) = remove_phoenix_bisector(wave, spec, Teff, logg, FeH)
    
    # fig, ax = plt.subplots(1, figsize=(30,9))
    # ax.plot(wave, spec_norm)
    # ax.set_xlim(5728, 5730)
    # fig.set_tight_layout(True)
    
    # plt.savefig("dbug.png", dpi=300)
    # exit()
    
    # Now let's try and add a Zhao bisector.
    mu_bis_dict = Zhao_bis_polynomials()
    mu_dict = {}
    # Now let's add in the bisectors
    available_mus = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    bis_polynomial_dict = Zhao_bis_polynomials()
    for mu in available_mus:
        spec_add, _, _, _, _ = add_bisector(wave, 
                                           copy.deepcopy(spec_corr), 
                                           bis_polynomial_dict[mu],
                                           Teff, 
                                           logg, 
                                           FeH, 
                                           debug_plot=True)
        mu_dict[mu] = spec_add
        # exit()
    # (spec_add, 
    #  spec_add_norm,
    #  _, 
    #  delta_v_add,
    #  delta_wave_add) = add_bisector(wave, spec, mu_bis_dict[1.0], Teff, logg, FeH )
    
    line = 5728.642
    interval = 0.25
    mask = np.logical_and(wave >= line - interval, wave <= line + interval)
    wv = wave[mask]
    sp = spec_norm[mask]
    dv = delta_v[mask]
    sp_corr = spec_corr[mask] / np.max(spec_corr[mask])
    for mu in mu_dict.keys():

        sp_add = mu_dict[mu]
        sp_add = sp_add[mask] / np.max(sp_add[mask])

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
            params, cov = curve_fit(_gauss_continuum, wv, sp_corr, expected)
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
        
        try:
            params, cov = curve_fit(_gauss_continuum, wv, sp_add, expected)
            width = params[1]
            continuum = params[-1]
        except:
            width = 0.05
            continuum = 1.0

        (bis_wave_add,
        bis_add,
        left_wv_add,
        left_sp_add,
        right_wv_add,
        right_sp_add) = bisector_on_line(wv, 
                                        sp_add, 
                                        line,
                                        width=width,
                                        outlier_clip=0.05,
                                        continuum=continuum)

        
        # bis_wave = bis_wave[bis<0.8]
        # bis = bis[bis<0.8]
        # bis_wave_corr = bis_wave_corr[bis_corr<0.8]
        # bis_corr = bis_corr[bis_corr<0.8]
        # bis_wave_add = bis_wave_add[bis_add<0.8]
        # bis_add = bis_add[bis_add<0.8]
        
        # Convert to velocities
        bis_v = (bis_wave - line) / bis_wave * 3e8
        bis_v_corr = (bis_wave_corr - line) / bis_wave_corr * 3e8
        bis_v_add = (bis_wave_add - line) / bis_wave_add * 3e8
        
        
        
        fig, ax = plt.subplots(1,2, figsize=(30,9))
        ax[0].plot(wv, sp, marker="o", markersize=8, color="black", label="Original PHOENIX")
        ax[0].plot(bis_wave, bis, color="black")
        ax[0].plot(wv, sp_corr, color="tab:red", marker="o", markersize=6, label="Removed PHOENIX Bisector")
        ax[0].plot(bis_wave_corr, bis_corr, color="tab:red")
        ax[0].plot(wv, sp_add, color="tab:green", marker="o", markersize=6, label="Added Zhao Bisector")
        ax[0].plot(bis_wave_add, bis_add, color="tab:green")
        ax[0].legend()
        ax[0].set_xlabel(r"Wavelength [$\AA$]")
        ax[0].set_ylabel("Flux")
        
        # mean_v = np.nanmean(bis_v)
        # bis_v -= mean_v
        # bis_v_corr -= mean_v
        # bis_v_add -= mean_v
        
        ax[1].plot(bis_v, bis, color="black", marker="o", markersize=8, label="Original Bisector")
        ax[1].plot(bis_v_corr, bis_corr, color="tab:red", marker="o", markersize=6, label="Removed PHOENIX Bisector")
        ax[1].plot(poly_fit(bis_corr), bis_corr, linewidth=5, color="tab:blue", label="Fitted Mean Bisector")
        ax[1].plot(dv, sp, linewidth=5, label="Used Delta V", color="tab:blue", linestyle="None", marker="o")
        # ax[1].set_xlim(-75, 75)
        ax[1].vlines(0, 0, 1, linestyle="dashed", color="black")
        ax[1].set_ylim(0,1)
        ax[1].set_xlabel(r"v [m/s]")
        
        lin_depth = np.linspace(0, 1, 100)
        ax[1].plot(mu_bis_dict[mu](lin_depth), lin_depth, label=f"Zhao Polyomial mu={mu}", color="tab:green")
        ax[1].plot(bis_v_add, bis_add, color="tab:green", marker="o", markersize=6, label="Added Zhao Bisector")
        ax[1].legend()
        
        fig.set_tight_layout(True)
        plt.savefig(f"data/plots/phoenix_bisectors/test_bis_corr{line}_mu={mu}.png", dpi=500)
    return wave,spec

if __name__ == "__main__":
    plot_bis_corr_star()