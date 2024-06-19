import numpy as np
import matplotlib.pyplot as plt
from dataloader import continuum, phoenix_spectrum
from scipy.interpolate import CubicSpline
from pathlib import Path
import plot_settings


def plot_normalization(wave, spec, cont, Teff, logg, feh, out_root):
    # wave, spec, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(3600, 17500))
    # wave_cont, cont = continuum(Teff, logg, feh, wavelength_range=(3600, 17500))
    
    # assert (wave == wave_cont).all()
    
    fig, ax = plt.subplots(2, 1, figsize=(plot_settings.THESIS_WIDTH, 3.5), sharex=True)
    
    print(spec)
    ax[0].plot(wave, spec, lw=0.5)
    ax[0].plot(wave, cont, lw=2, color="tab:red")
    
    ax[1].plot(wave, spec/cont, lw=0.5)
    ax[1].set_xlabel(r"Wavelength [\AA]")
    ax[0].set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\,cm\,cm^2}} \right]$")
    ax[1].set_ylabel(r"Normalized Flux")
    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0, top=1.10)
    fig.subplots_adjust(hspace=0, left=0.12, right=.99, top=0.95, bottom=0.13)
    ax[1].set_xlim(3600, 17500)
    ax[1].yaxis.set_label_coords(-0.08, 0.5)
    ax[0].yaxis.set_label_coords(-0.075, 0.5)
    
    plt.savefig(out_root / f"{Teff:.1f}K_{logg}_{feh}_norm.pdf", dpi=500)
    
    
def interpolate_continuum(T_local, logg, feh, wavelength_range=None):
    """ Interpolate a Continuum"""
    # For the moment only within [4400, 4600]
    Ts = np.array([4400, 4500, 4600])
    cont_dict = {}
    for T in Ts:
        wave, cont = continuum(T, logg, feh, wavelength_range=wavelength_range)
        cont_dict[T] = cont
        
        
    cont_interpol = np.zeros_like(wave)
    for idx, wv in enumerate(wave):
        print(idx, len(wave))
        cont_at_wave = []
        for T, cont in cont_dict.items():
            cont_at_wave.append(cont[idx])
        sp = CubicSpline(Ts, cont_at_wave)
        cont_interpol[idx] = sp(T)
        
    return wave, cont_interpol

if __name__ == "__main__":
    out_root = Path("/home/dspaeth/pyoscillot/PhD_plots")
    Teff = 4500
    logg = 2.0
    feh = 0.0
    wave, spec, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(3600, 17500))
    
    
    wave_cont, cont = continuum(Teff, logg, feh, wavelength_range=(3600, 17500))
    plot_normalization(wave, spec, cont, 4500, 2.0, 0.0, out_root)

