from dataloader import phoenix_spectrum
from utils import normalize_phoenix_spectrum
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from physics import planck
from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline

# First get rid of duplicates
all_phoenix = sorted(list(Path("/home/dspaeth/pypulse/data/phoenix_spectra").glob("*.fits")))
for file in all_phoenix:
    if "(1).fits" in file.name:
        new_file = file.parent / "duplicates" / file.name
        print(f"Move {file} to {new_file}")
        file.rename(new_file)
    
# Now again
all_phoenix = sorted(list(Path("/home/dspaeth/pypulse/data/phoenix_spectra").glob("lte*.fits")))



# Create here only the models that you need
all_phoenix = []
# _,_,_, filepath = phoenix_spectrum(Teff=5700, logg=4.5, feh=0.0, return_filepath=True)
# all_phoenix.append(filepath)
# _,_,_, filepath = phoenix_spectrum(Teff=5600, logg=4.5, feh=0.0, return_filepath=True)
# all_phoenix.append(filepath)
# _,_,_, filepath = phoenix_spectrum(Teff=5800, logg=4.5, feh=0.0, return_filepath=True)
# all_phoenix.append(filepath)

Teff=4500
_,_,_, filepath = phoenix_spectrum(Teff=Teff, logg=2.0, feh=0.0, return_filepath=True)
all_phoenix.append(filepath)

logg = 2.0
feh = 0.0
for Teff in range(4600, 4700, 100):
    
    _,_,_, file = phoenix_spectrum(Teff=Teff, logg=logg, feh=feh, return_filepath=True)
    name = file.name
    
    # Get Teff, logg, feh from the filenames
    value_str = name.split("lte")[-1].split(".PHOENIX")[0]
    Teff = int(value_str.split("-")[0])
    logg_feh_str = value_str[6:]
    feh_sign = value_str[-4]
    logg = float(logg_feh_str.split(feh_sign)[0])
    feh = float(logg_feh_str.split(feh_sign)[1]) * int(feh_sign+"1")
    
    if Teff < 4000:
        continue
    
    # Now load the full spectra
    wien_wave = 2.898e-3/Teff *1e10
    
    # print(wien_wave)
    wien_wave -= 1000
    
    
    # print(wien_wave)
    # exit()
    # wave, spec, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(6000, 15000))
    wave_low, spec_low, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(3400, wien_wave-500))
    wave_mid, spec_mid, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(wien_wave-1000, wien_wave+5000))
    wave_high, spec_high, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(wien_wave+4500, 17600))
    #wave, spec, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(4500, 5500))
    
    _, spec_norm_low, continuum_interp_low = normalize_phoenix_spectrum(wave_low, spec_low, Teff, logg, feh, run=True)
    # file_out = Path("/home/dspaeth/pypulse/pypulse/RASSINE_phoenix_spec_rassine.p")
    out_root = Path("/home/dspaeth/pypulse/data/continuum_fits")
    # new_file = out_root / (file.stem + ".p") 
    
    
    # file_out.rename(new_file)
    _, spec_norm_mid, continuum_interp_mid = normalize_phoenix_spectrum(wave_mid, spec_mid, Teff, logg, feh, run=True)
    _, spec_norm_high, continuum_interp_high = normalize_phoenix_spectrum(wave_high, spec_high, Teff, logg, feh, run=True)
    
    # Make a debug plot
    fig, ax = plt.subplots(1, figsize=(30,9))
    ax.plot(wave_low, spec_low, color="tab:blue")
    ax.plot(wave_mid, spec_mid, color="tab:blue")
    ax.plot(wave_high, spec_high, color="tab:blue")
    
    ax.plot(wave_low, continuum_interp_low, color="tab:green")
    ax.plot(wave_mid, continuum_interp_mid, color="tab:green")
    ax.plot(wave_high, continuum_interp_high, color="tab:green")
    
    cutoff_px = 2000
    wave_low = wave_low[cutoff_px:-cutoff_px]
    wave_mid = wave_mid[cutoff_px:-cutoff_px]
    wave_high = wave_high[cutoff_px:-cutoff_px]
    continuum_interp_low = continuum_interp_low[cutoff_px:-cutoff_px]
    continuum_interp_mid = continuum_interp_mid[cutoff_px:-cutoff_px]
    continuum_interp_high = continuum_interp_high[cutoff_px:-cutoff_px]
    
    poly_fit_low = Polynomial.fit(wave_low, continuum_interp_low, 8)
    ax.plot(wave_low, poly_fit_low(wave_low), color="tab:orange")
    
    poly_fit_mid = Polynomial.fit(wave_mid, continuum_interp_mid, 8)
    ax.plot(wave_mid, poly_fit_mid(wave_mid), color="tab:orange")
    
    poly_fit_high = Polynomial.fit(wave_high, continuum_interp_high, 8)
    ax.plot(wave_high, poly_fit_high(wave_high), color="tab:orange")
    
    continuum_low = poly_fit_low(wave_low)
    continuum_mid = poly_fit_mid(wave_mid)
    continuum_high = poly_fit_high(wave_high)
    
    transition_mask_wave_low = wave_low >= wave_mid[0]
    transition_mask_wave_mid_low = wave_mid <= wave_low[-1]
    
    transition_wave_low_low = wave_low[transition_mask_wave_low]
    transition_cont_low_low = continuum_low[transition_mask_wave_low]
    transition_wave_low_mid = wave_mid[transition_mask_wave_mid_low]
    transition_cont_low_mid = continuum_mid[transition_mask_wave_mid_low]
    
    
    assert (transition_wave_low_low == transition_wave_low_mid).all()
    N_over = len(transition_wave_low_low)
    indices = np.linspace(0, N_over-1, N_over, dtype=float)
    assert len(indices) == N_over, f"{len(indices)}!={N_over}"
    transition_continuum = (transition_cont_low_low * ((N_over - indices)/N_over) +
                            transition_cont_low_mid * (indices/N_over))
    print(indices)
    # indices = indices.astype(float)
    
    assert (((N_over - indices)/N_over + indices/N_over) == 1.0).all(), (N_over - indices)/N_over + ((indices)/N_over)[(N_over - indices)/N_over + ((indices)/N_over) != 1.0]
    
    wave_combined = wave_low
    continuum_combined = np.append(continuum_low[np.logical_not(transition_mask_wave_low)], transition_continuum)
    
    
    # Do the same for the high range
    transition_mask_wave_mid_high = wave_mid >= wave_high[0]
    transition_mask_wave_high = wave_high <= wave_mid[-1]
    
    transition_wave_high_low = wave_mid[transition_mask_wave_mid_high]
    transition_cont_high_low = continuum_mid[transition_mask_wave_mid_high]
    transition_wave_high_mid = wave_high[transition_mask_wave_high]
    transition_cont_high_mid = continuum_high[transition_mask_wave_high]
    
    
    assert (transition_wave_high_low == transition_wave_high_mid).all()
    N_over = len(transition_wave_high_low)
    indices = np.linspace(0, N_over+1, N_over, dtype=int)
    assert len(indices) == N_over, f"{len(indices)}!={N_over}"
    transition_continuum = (transition_cont_high_low * ((N_over - indices)/N_over) +
                            transition_cont_high_mid * (indices/N_over))
    
    assert ((((N_over - indices)/N_over) + indices/N_over) == 1).all(), (N_over - indices)/N_over + ((indices)/N_over)
    
    
    normal_mid_mask = np.logical_and(np.logical_not(transition_mask_wave_mid_low), np.logical_not(transition_mask_wave_mid_high))
    continuum_combined = np.append(continuum_combined, continuum_mid[normal_mid_mask])
    
    
    continuum_combined = np.append(continuum_combined, transition_continuum)
    continuum_combined = np.append(continuum_combined, continuum_high[np.logical_not(transition_mask_wave_high)])
    
    
    wave_combined = np.append(wave_combined, wave_mid[normal_mid_mask])
    wave_combined = np.append(wave_combined, wave_high)
    
    
    # poly_fit_comb =  Polynomial.fit(wave_combined, continuum_combined, 16)
    ax.plot(wave_combined, continuum_combined, color="tab:red", lw=2)
    ax.set_xlabel(r"Wavelength [$\AA]")
    ax.set_ylabel("Flux")
    # ax.plot(wave, planck(wave*1e-10, T=4500)*10*np.pi)
    ax.set_title(file.stem)
    ax.set_xlim(3500, 17500)
    fig.set_tight_layout(True)
    plt.savefig(out_root  / "plots" / (file.stem + ".png"), dpi=500)
    plt.close()
    
    # Now also save the arrays
    np.save(out_root / f"{file.stem}_wave.npy", wave_combined)
    np.save(out_root / f"{file.stem}_cont.npy", continuum_combined)
    