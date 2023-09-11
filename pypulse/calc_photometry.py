from dataloader import V_band_filter
from scipy.interpolate import interp1d
import numpy as np

def V_band_flux(wave, spec):
    """ Calculate the flux in the Bessel V band"""
    filter_wave, filter_curve = V_band_filter()
    assert wave[0] < filter_wave[0], "Your wavelength array starts before the V band range"
    assert wave[-1] > filter_wave[-1], "Your wavelength array ends after the V band range"
    
    # Interpolate the filter range
    interp_filter = interp1d(filter_wave, filter_curve, kind="linear", fill_value=0.0)
    
    # Cut the wave array to lie within the V band range
    mask = np.logical_and(wave >= filter_wave[0], wave <= filter_wave[-1])
    wave_V = wave[mask]
    spec_V = spec[mask]
    
    filter_flux = np.sum(spec_V * interp_filter(wave_V))
    
    return filter_flux

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    root = Path("/data/dspaeth/pypulse_fake_spectra/NGC4349_Test188/arrays")
    
    wave_files = sorted(list((root / "wavelength").glob("*npy")))
    spec_files =  sorted(list((root / "spectrum").glob("*npy")))
    
    V_fluxes = []
    bjds = []
    total_fluxes = []
    for wave_file, spec_file in zip(wave_files, spec_files):
        bjds.append(float(spec_file.stem))
        assert wave_file.name == spec_file.name
        
        wave = np.load(wave_file)
        spec = np.load(spec_file)
        V_flux = V_band_flux(wave, spec)
        V_fluxes.append(V_flux)
        total_fluxes.append(np.sum(spec))
        
    V_fluxes = np.array(V_fluxes)
    V_fluxes /= np.mean(V_fluxes)
    total_fluxes = np.array(total_fluxes)
    total_fluxes /= np.mean(total_fluxes)
    
    plt.scatter(bjds, V_fluxes, label="V flux")
    plt.scatter(bjds, total_fluxes, label="Total flux")
    plt.savefig("dbug.png", dpi=300)
    
    
    