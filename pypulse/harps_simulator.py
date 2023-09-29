import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import deconvolve
import matplotlib.pyplot as plt
from astropy.time import Time
from dataloader import harps_template
from utils import adjust_resolution, adjust_resolution_dane, _gauss_continuum, bisector_on_line, rebin
import cfg



def interpolate(spectrum, wavelength):
    """ Interpolate to the HARPS spectrum.


        :param np.array spectrum: The calculated spectrum to interpolate to HARPS
        :param np.array wavelength: The calculated wavelength to interpolate to HARPS
    """
    # Load the template spectra and wavelength
    (tmpl_spec, tmpl_wave, tmpl_blaze) = harps_template(spec_filename="HARPS_template_ngc4349_127_e2ds_A.fits")

    interpol_spec = []
    R_real = 115000
    # R_test = 130000
    print("Adjusting Resolution")
    for order in range(len(tmpl_wave)):

        # Cut the calculated wavelengths and spectra to the order
        local_wave_mask = np.logical_and(wavelength > tmpl_wave[order][0] - 100,
                                         wavelength < tmpl_wave[order][-1] + 100)

        local_wavelength = wavelength[local_wave_mask]
        local_spectrum = spectrum[local_wave_mask]
        

        # Adjust the resolution per order
        # This will use one kernel per order
        local_spectrum_HARPS = adjust_resolution_dane(local_wavelength, local_spectrum, R=R_real)
        # DEBUG PLOT
        debug_line = 5728.65
        if debug_line > local_wavelength[0] and debug_line < local_wavelength[-1]:
            debug_plot(local_wavelength.copy(), 
                       local_spectrum.copy(), 
                       local_spectrum_HARPS.copy(), 
                       debug_line)

        # Interpolate the calculated spectra onto the tmpl_wave grid
        # func = interp1d(local_wavelength, local_spectrum_HARPS, kind="linear")
        # order_spec = func(tmpl_wave[order])
        order_spec = rebin(local_wavelength, local_spectrum_HARPS, tmpl_wave[order])

        # Adjust for the blaze
        order_spec *= tmpl_blaze[order]

        # Reduce the level to something similar to HARPS
        order_spec = order_spec * \
                     np.nanmean(tmpl_spec[order]) / np.nanmean(order_spec)

        interpol_spec.append(order_spec)
    interpol_spec = np.array(interpol_spec)

    return interpol_spec, tmpl_wave

def debug_plot(wave, spec, spec_HARPS, line):
    """ Create a debug plot."""
    interval = 0.25
    mask = np.logical_and(wave >= line - interval, wave <= line + interval)
    
    wave = wave[mask]
    spec = spec[mask]
    spec_HARPS = spec_HARPS[mask]
    
    # Rough normalization
    spec_norm = spec / np.max(spec)
    spec_HARPS_norm = spec_HARPS / np.max(spec_HARPS)
    
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 4.0275))
    ax[0].plot(wave, spec_norm, color="tab:blue",marker="o", label="Combined PHOENIX Spectrum")
    ax[0].plot(wave, spec_HARPS_norm, color="tab:red",marker="o", label="HARPS Resolution")
    ax[0].legend()
    ax[0].set_xlabel(r"Wavelength [$\AA$]")
    ax[0].set_ylabel("Normalized Flux")
    ax[0].set_title(rf"{line}$\AA$")
    ax[0].ticklabel_format(useOffset=False)
    
    # Fit the bisectors for both lines
    mean_bis_v = None
    for sp, color in zip((spec_norm, spec_HARPS_norm), ("tab:blue", "tab:red")):
        # Fit the width and center for an inital guess
        expected = (line, 0.05, 0.9, 1.0)
        try:
            params, cov = curve_fit(_gauss_continuum, wave, sp, expected)
            width = params[1]
            continuum = params[-1]

        except:
            width = 0.05
            continuum = 1.0
            
    
    
        try:
            bis_wave, bis, left_wv, left_sp, right_wv, right_sp = bisector_on_line(wave, 
                                                                                   sp, 
                                                                                   line,
                                                                                   width=width,
                                                                                   outlier_clip=0.1,
                                                                                   continuum=continuum)
        
            # Convert to velocities
            bis_v = (bis_wave - line) / bis_wave * 3e8
            
            if mean_bis_v is None:
                mean_bis_v = np.nanmean(bis_v)
            ax[0].plot(bis_wave, bis, color=color, marker="o")
            ax[1].plot(bis_v - mean_bis_v, bis, color=color, marker="o")
        except Exception as e:
            pass
    
    
    
    # ax[1].plot(wave[mask], delta_v[mask])
    ax[0].legend(loc="lower left")
    
    # Add the BIS polynomial
    lin_spec = np.linspace(0, 1, 100)
    # ax[1].plot(poly_fit(lin_spec), lin_spec, color="black", alpha=0.7, label=f"Fitted Mean Bisector")
    ax[1].legend(loc="lower left")
    
    
    
    # ax[1].plot(wave[mask], delta_v[mask])
    if cfg.debug_dir is not None:
        out_root = cfg.debug_dir
    else:
        out_root = Path("/home/dspaeth/pypulse/data/plots/phoenix_bisectors/")
    savename = f"bis_HARPS.png"
    fig.set_tight_layout(True)
    plt.savefig(f"{out_root}/{savename}", dpi=600)
    plt.close()



def get_new_header(time, bc=None, bjd=None):
    """ Create the new header for the fake Carmenes spectrum.

        :param time: Time of observation
        :param bc: Barycentric Correction to write into DRS
        :param bjd: Barycentric Julian Date to write into DRS

        Add only keys that should be new.
    """
    time = Time(time, scale="utc")
    header_dict = {"DATE": time.isot,
                   "DATE-OBS": time.isot,
                   "MJD-OBS": time.mjd + 2400000,
                   #"JD": time.jd - 2400000,
                   #"HJD": time.jd - 2400000,
                   "PI-COI": "SpaethSim",
                   "OBJECT": "Sim",
                   "TEXPTIME": 100.0}

    # serval uses the comment of MJD-OBS to retrieve a timeid
    comment_dict = {"MJD-OBS": f"MJD start ({time.isot})"}
    # HJD is wrong but not so important at the moment
    if bc is not None:
        header_dict["HIERARCH ESO DRS BERV"] = bc / 1000
    if bjd is not None:
        header_dict["HIERARCH ESO DRS BJD"] = bjd

    # others: RA, DEC, UTC, LST, MJD-END

    return header_dict, comment_dict

if __name__ == "__main__":
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    num = 1000000
    wave = np.linspace(3000, 9000, num)
    spec = np.ones_like(wave)

    interval = 100
    peak_idx = np.arange(int(0 + interval / 2), num, interval)
    spec[peak_idx] -= 0.5

    interpol_spec, tmpl_wave = interpolate(spec, wave)

    order = 30
    plt.figure(dpi=300)
    plt.plot(tmpl_wave[order], interpol_spec[order])

    w = tmpl_wave[order]
    s = interpol_spec[order]

    mins_idx = s < 0.973

    plt.plot(w[mins_idx], s[mins_idx])
    plt.show()


