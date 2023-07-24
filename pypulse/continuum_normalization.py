import numpy as np
import matplotlib.pyplot as plt
from dataloader import phoenix_spectrum, continuum
from scipy.interpolate import CubicSpline


def plot_normalization(wave, spec, cont, Teff, logg, feh, out_root):
    # wave, spec, header = phoenix_spectrum(Teff, logg, feh, wavelength_range=(3600, 17500))
    # wave_cont, cont = continuum(Teff, logg, feh, wavelength_range=(3600, 17500))
    
    # assert (wave == wave_cont).all()
    
    fig, ax = plt.subplots(2, 1, figsize=(6.35, 3.5), sharex=True)
    ax[0].plot(wave, spec, lw=0.5)
    ax[0].plot(wave, cont, lw=2, color="tab:red")
    
    ax[1].plot(wave, spec/cont, lw=0.5)
    ax[1].set_xlabel(r"Wavelength [$\AA$]")
    ax[0].set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
    ax[1].set_ylabel("Normalued Flux")
    fig.subplots_adjust(hspace=0, left=0.1, right=.99, top=0.99, bottom=0.12)
    ax[1].set_xlim(3500, 17500)
    
    plt.savefig(out_root / f"{Teff}K_{logg}_{feh}_norm.png", dpi=500)
    
    
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
    wave, cont_interpol = plot_normalization(4500, 2.0, 0.0)
    print(cont_interpol)

