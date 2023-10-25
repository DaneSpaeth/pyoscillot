from dataloader import phoenix_spectrum
from utils import normalize_phoenix_spectrum_Rassine
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from continuum_normalization import plot_normalization
from spline_interpolation import interpolate_on_temperature
from utils import get_ref_spectra

    
out_root = Path("/data/dspaeth/pypulse_data/continuum_fits")
WAVE_START = 3550
WAVE_STOP = 17550
cutoff_px = 200
poly_order = 3

# Define stellar parameters
logg = 2.0
feh = 0.0

min_T = 4390
max_T = 4410
step = 0.1

wave, ref_spectra, ref_headers = get_ref_spectra(np.array([min_T, max_T]), logg=logg, feh=feh, wavelength_range=(WAVE_START, WAVE_STOP))
Teffs = np.arange(min_T, max_T+step, step)

for Teff in Teffs:
    # To get the right sign for PHOENIX
    if feh == 0.0:
        feh_str = -0.0000001
    else:
        feh_str = feh
    # Now take care of the folder
    T_round = int(np.floor(Teff/100)*100)
    out_dir = out_root / f"{T_round:05d}K_{logg:.2f}_{feh_str:+.1f}"
    
    if not out_dir.is_dir():
        out_dir.mkdir() 
    # Make a debug plot
    print("\n\n\n")
    print("==================================================")
    print(f"START RUN FOR TEFF={Teff}")
    print("==================================================")
    fig, ax = plt.subplots(1, figsize=(30,9))
    
    
    out_str = f"{Teff:05.1f}K-{logg:.2f}{feh_str:+.1f}"
    
    # wave, spec,_, file = phoenix_spectrum(Teff=Teff, logg=logg, feh=feh, return_filepath=True, wavelength_range=(WAVE_START,WAVE_STOP))
    spec = interpolate_on_temperature(Teff, wave, ref_spectra, logg, feh)
    # Calculate the simple Wien wavelength
    wien_wave = 2.898e-3 / Teff * 1e10
    
    # if Teff < 5300:
    #     lower_split = wien_wave - 500
    # else:
    #     lower_split = wien_wave + 500
    
    max_wave = wave[np.argmax(spec)]
    lower_split = max_wave + 500
    # lower_split = wave[np.argmax(spec)] 
    split_waves = [lower_split, lower_split + 3000, 15000, WAVE_STOP]
    overlap = 500
    
    start_wave = WAVE_START
    
    waves = []
    specs = []
    conts = []
    for idx, split_wave in enumerate(split_waves):
        end_wave = split_wave
        if not end_wave == WAVE_STOP:
            end_wave += overlap
        # wave, spec, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(start_wave, end_wave))
        mask = np.logical_and(wave >= start_wave, wave <= end_wave)
        wave_local = wave[mask]
        spec_local = spec[mask]
        _, spec_norm, continuum_interp = normalize_phoenix_spectrum_Rassine(wave_local, spec_local, Teff, logg, feh, run=True)
        
        # Plot the raw spectrum and the Rassine Continuum
        ax.plot(wave_local, spec_local, color="tab:blue")
        ax.plot(wave_local, continuum_interp, color="tab:green")
        
        # Now cut the array
        wave_local = wave_local[cutoff_px:-cutoff_px]
        continuum_interp = continuum_interp[cutoff_px:-cutoff_px]
        
        # Small adjustment for the first part
        # if idx == 0:
        #     border = max_wave
        #     continuum_interp[wave < border] += continuum_interp[wave < border] * (np.abs(wave[wave < border] - border)/(border - wave[0]) * 0.05) 
        if idx == 0:
            poly_order_local = 5
            weights = continuum_interp
            HARPS_mask = wave_local > 4000
            poly_fit = Polynomial.fit(wave_local[HARPS_mask], continuum_interp[HARPS_mask], poly_order_local, w=weights[HARPS_mask])
        elif idx == (len(split_waves) - 1):
            poly_order_local = poly_order + 2
            weights = np.ones_like(wave_local)
            poly_fit = Polynomial.fit(wave_local, continuum_interp, poly_order_local, w=weights)
        else:
            poly_order_local = poly_order
            weights = np.ones_like(wave_local)
            poly_fit = Polynomial.fit(wave_local, continuum_interp, poly_order_local, w=weights)
        
        ax.plot(wave_local, poly_fit(wave_local), color="tab:orange")
        
        waves.append(wave_local)
        specs.append(spec_local)
        conts.append(poly_fit(wave_local))
    
        start_wave = split_wave - overlap 
        
    # Now calculate the transition regions
    wave_combined = np.array([])
    cont_combined = np.array([])
    
    LAST_MASK = None
    for idx in range(len(waves)-1):
        # For both the current arrays and the next calculate a mask
        # that defines where the arrays overlap
        transition_mask_low = waves[idx] >= waves[idx + 1][0]
        transition_mask_high = waves[idx + 1] <= waves[idx][-1]
        
        transition_wave_low = waves[idx][transition_mask_low]
        transition_cont_low = conts[idx][transition_mask_low]
        transition_wave_high = waves[idx + 1][transition_mask_high]
        transition_cont_high = conts[idx + 1][transition_mask_high]
        
        assert (transition_wave_low == transition_wave_high).all()
        N_over = len(transition_wave_low)
        indices = np.linspace(0, N_over-1, N_over, dtype=float)
        transition_continuum = (transition_cont_low * ((N_over - indices)/N_over) +
                                transition_cont_high * (indices/N_over))
        assert (((N_over - indices)/N_over + indices/N_over) == 1.0).all(), (N_over - indices)/N_over + ((indices)/N_over)[(N_over - indices)/N_over + ((indices)/N_over) != 1.0]
        
        # Now combine the waves and conts
        # Add in the overlap
        if LAST_MASK is None:
            wave_combined = np.append(wave_combined, waves[idx])
            cont_combined = np.append(cont_combined, conts[idx][~transition_mask_low])
        else:
            # Take care that you do not add the beginning again that was already added in from the last loop
            wave_combined = np.append(wave_combined, waves[idx][~LAST_MASK])
            cont_combined = np.append(cont_combined, conts[idx][np.logical_and(~transition_mask_low, ~LAST_MASK)])
        cont_combined = np.append(cont_combined, transition_continuum)
        # So now we have already covered the full first wave array including the continuum region
        # Keep track of the last mask for the next loop
        # So that you do not add them in again
        LAST_MASK = transition_mask_high
        
    # We finally need to add in the final part
    idx = len(waves) - 1
    wave_combined = np.append(wave_combined, waves[idx][~LAST_MASK])
    cont_combined = np.append(cont_combined, conts[idx][~LAST_MASK])
    
    ax.plot(wave_combined, cont_combined, color="tab:red", lw=2)
    
    ax.set_xlabel(r"Wavelength [$\AA]")
    ax.set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
    ax.set_title(out_str)
    ax.set_xlim(3500, 17500)
    
    ylims = ax.get_ylim()
    ax.vlines(3780, ylims[0], ylims[1], linestyle="dashed", color="black")
    ax.vlines(6910, ylims[0], ylims[1], linestyle="dashed", color="black")
    ax.set_ylim(ylims)
    ax.text((6910+3780)/2, ylims[1]*0.97, "HARPS wavelength range", horizontalalignment="center", color="black")
    fig.set_tight_layout(True)
    folder = out_dir / "plots"
    if not folder.is_dir():
        folder.mkdir()
    plt.savefig(folder / (out_str + ".png"), dpi=500)
    plt.close()
    
    # Now also save the arrays
    wave_out = out_root / "wave.npy"
    if not wave_out.is_file():
        np.save(wave_out, wave_combined)
    np.save(out_dir / f"{out_str}_cont.npy", cont_combined)
    
    # And another debug plot
    folder = out_dir / "norm_plots"
    if not folder.is_dir():
        folder.mkdir()
    spec = spec[cutoff_px:-cutoff_px]
    plot_normalization(wave_combined, spec, cont_combined, Teff, logg, feh, folder)
    plt.close()
    
    
    