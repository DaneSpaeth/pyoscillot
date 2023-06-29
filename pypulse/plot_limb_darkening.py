import numpy as np
import matplotlib.pyplot as plt
from three_dim_star import ThreeDimStar, TwoDimProjector
from utils import add_limb_darkening
from dataloader import phoenix_spectrum

def plot_stellar_disk_comparison():
    star = ThreeDimStar()
    proj = TwoDimProjector(star, N=1000, border=3, limb_darkening=False)

    mu = proj.mu()

    wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(4200, 7700))

    wavelengths = [4500, 5500, 6500, 7500]
    wave_idxs = []

    for wavelength in wavelengths:
        wave_idxs.append(np.argmin(np.abs(wave - wavelength)))
        
    mask = np.zeros_like(wave, dtype=bool)
    for idx in wave_idxs:
        mask[idx] = True
        
    wave = wave[mask]
    spec = spec[mask]
        

    intensities_at_waves = np.zeros((mu.shape[0], mu.shape[1], len(wave_idxs)))
    for (row, col), m in np.ndenumerate(mu):
        int_mu = add_limb_darkening(wave, spec, m)[0]
        print(row, col)
        for idx, wv_idx in enumerate(wave_idxs):
            # print(idx)
            intensities_at_waves[row, col, idx] = int_mu[idx]
        

    colors = ["tab:blue", "tab:green", "yellow", "tab:red"]

    fig, ax = plt.subplots(2,2, figsize=(7.16, 7.16))

    # print(len(wavelengths))
    # print(len(colors))
    # print(len(ax.flatten()))

    for idx, (wavelength, color, a) in enumerate(zip(wavelengths, colors, ax.flatten())):
        intensities_at_waves[:,:,idx][np.isnan(intensities_at_waves[:,:,idx])] = 0
        wv_idx = np.argmin(np.abs(wave - wavelength))
        # twina = a.twiny()
        a.imshow(intensities_at_waves[:,:,idx], vmin=0.0, vmax=1.0, cmap="inferno")
        
        y_offset = int(intensities_at_waves.shape[1]/2)
        a.plot(y_offset - 100*intensities_at_waves[:, y_offset, idx], color=color, label=f"{int(wavelength/10)}nm", lw=5)
        a.set_title(f"{int(wavelength/10)}nm")
        a.set_xticks([])
        a.set_yticks([])
        

    # ax.legend()
    # ax.set_ylabel("Normalized Intensity")
    fig.set_tight_layout(True)
    plt.savefig("limb_dark_3D.png", dpi=600)
    plt.close()
    
def plot_mu_comparison():
    wavelengths = np.array([4500, 5500, 6500, 7500], dtype=float)
    
    fig, ax = plt.subplots(1,1,figsize=(7.16, 4.0275))
    mu = np.linspace(0,1,1000)
    colors = ["tab:blue", "tab:green", "yellow", "tab:red"]
    
    for idx, (wavelength, color) in enumerate(zip(wavelengths, colors )):
        
        ax.plot(mu, [add_limb_darkening(wavelengths, None, m)[0][idx] for m in mu], color=color, label=f"{int(wavelength/10)}nm", lw=5)
    ax.set_xlabel("µ")
    ax.set_ylabel("Relative Intensity")
    ax.legend()
    ax.set_xlim(0,1)
    fig.set_tight_layout(True)
    plt.savefig("limb_dark_comparison.png", dpi=600)
    
def plot_spectral_change():
    wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(4200, 7700))
    
    norm = np.nanmax(spec)
    
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7.16, 4.0275))
    lw = 0.25
    ax[0].plot(wave, spec/norm, color="blue", label="Original PHOENIX", lw=lw)
    mu = 1
    _, spec_limb = add_limb_darkening(wave, spec, mu)
    ax[0].plot(wave, spec_limb/norm, color="red", alpha=0.7, label=f"Adjusted for Limb Darkening at µ={mu}", lw=lw)
    mu = 0.2
    _, spec_limb = add_limb_darkening(wave, spec, mu)
    ax[1].plot(wave, spec/norm, color="blue", label="Original PHOENIX", lw=lw)
    ax[1].plot(wave, spec_limb/norm, color="red", alpha=0.7, label=f"Adjusted for Limb Darkening at µ={mu}", lw=lw)
    
    ax[1].set_xlabel(r"Wavelength [$\AA$]")
    ax[0].set_ylabel("Flux [arb. units]")
    ax[1].set_ylabel("Flux [arb. units]")
    ax[0].legend()
    ax[1].legend()
    
    ax[0].set_ylim(0, 1.05)
    ax[1].set_ylim(0, 1.05)
    
    fig.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.12, hspace=0)
    
    plt.savefig("limb_dark_spec.png", dpi=600)
    
if __name__ == "__main__":
    plot_spectral_change()