import numpy as np
import matplotlib.pyplot as plt
from three_dim_star import ThreeDimStar, TwoDimProjector
from utils import calc_mean_limb_dark, add_limb_darkening, calc_limb_dark_intensity
from dataloader import phoenix_spectrum
from star import GridSpectrumSimulator
from cfg import parse_global_ini
conf_dict = parse_global_ini()
import plot_settings

def plot_mean_limb_dark():
    star = ThreeDimStar()
    N = 1000
    proj = TwoDimProjector(star, N=N, border=3, limb_darkening=False)

    mu = proj.mu()
    wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(3500, 17500))
    
    wave_start = np.min(wave)
    wave_stop = np.max(wave)
    mean_limb_dark = calc_mean_limb_dark(wave, mu, N=N)
    
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 4.0275))
    ax[0].imshow(mu, vmin=0, vmax=1, cmap="viridis")
    ax[1].plot(wave, mean_limb_dark)
    plt.savefig("mean_limb_dark.png", dpi=600)
    
    






####### Probably all wrong from here! ########

def plot_stellar_disk_comparison():
    star = ThreeDimStar()
    proj = TwoDimProjector(star, N=150, border=3)

    mu = proj.mu()

    wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(3600, 7150))

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

    fig, ax = plt.subplots(1,4, figsize=(plot_settings.THESIS_WIDTH, 4.0275/2))

    # print(len(wavelengths))
    # print(len(colors))
    # print(len(ax.flatten()))

    for idx, (wavelength, color, a) in enumerate(zip(wavelengths, colors, ax.flatten())):
        intensities_at_waves[:,:,idx][np.isnan(intensities_at_waves[:,:,idx])] = 0
        wv_idx = np.argmin(np.abs(wave - wavelength))
        # twina = a.twiny()
        a.imshow(intensities_at_waves[:,:,idx], vmin=0.0, vmax=1.0, cmap="inferno")
        
        y_offset = int(intensities_at_waves.shape[1]/2)
        # a.plot(y_offset - 100*intensities_at_waves[:, y_offset, idx], color=color, label=f"{int(wavelength/10)}nm", lw=5)
        a.set_title(f"{int(wavelength/10)} nm")
        a.set_xticks([])
        a.set_yticks([])
        

    # ax.legend()
    # ax.set_ylabel("Normalized Intensity")
    fig.set_tight_layout(True)
    
    plt.savefig("/home/dspaeth/pyoscillot/PhD_plots/limb_dark_3D_1x4.pdf", dpi=600)
    plt.close()
    
def plot_mu_comparison():
    wavelengths = np.array([5345, 7400, 13350], dtype=float)
    
    fig, ax = plt.subplots(1, 1, figsize=(plot_settings.THESIS_WIDTH, 3.0))
    mu = np.linspace(0, 1, 1000)
    colors = ["tab:blue", "tab:green", "tab:red"]
    linestyles = ["solid", "dashdot", "dashed"]
    
    for idx, (wavelength, color, l) in enumerate(zip(wavelengths, colors, linestyles )):
        
        intensity = calc_limb_dark_intensity(wavelength, mu)
        print(intensity)
        
        # ax.plot(mu, [add_limb_darkening(wavelengths, None, m)[0][idx] for m in mu], color=color, label=f"{int(wavelength/10)}nm", lw=5)
        ax.plot(mu, intensity, label=f"{int(wavelength/10)} nm", lw=2.5, color=color, linestyle=l)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel("Relative Intensity")
    ax.legend(handlelength=4)
    ax.set_xlim(0,1)
    fig.set_tight_layout(True)
    plt.savefig("limb_dark_comparison.pdf", dpi=600)
    
def plot_spectral_change():
    wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(4200, 10000))
    
    norm = np.nanmax(spec)
    
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7.16, 4.0275))
    lw = 0.25
    ax[0].plot(wave, spec/norm, color="tab:blue", label="Original PHOENIX", lw=lw)
    mu = 1
    _, spec_limb = add_limb_darkening(wave, spec, mu)
    ax[0].plot(wave, spec_limb/norm, color="tab:red", alpha=0.7, label=f"Adjusted for Limb Darkening at µ={mu}", lw=lw)
    mu = 0.2
    _, spec_limb = add_limb_darkening(wave, spec, mu)
    ax[1].plot(wave, spec/norm, color="tab:blue", label="Original PHOENIX", lw=lw)
    ax[1].plot(wave, spec_limb/norm, color="tab:red", alpha=0.7, label=f"Adjusted for Limb Darkening at µ={mu}", lw=lw)
    
    ax[1].set_xlabel(r"Wavelength [$\AA$]")
    ax[0].set_ylabel("Normalized Flux")
    ax[1].set_ylabel("Normalized Flux")
    ax[0].legend()
    ax[1].legend()
    
    ax[0].set_ylim(0, 1.05)
    ax[1].set_ylim(0, 1.05)
    
    fig.subplots_adjust(left=0.07, right=0.99, top=0.99, bottom=0.12, hspace=0)
    
    plt.savefig("limb_dark_spec.png", dpi=600)
    
def plot_summed_spectral_change():
    fig, ax = plt.subplots(1, figsize=(7.16, 4.0275))
    star = GridSpectrumSimulator(N_star=100, Teff=4500, logg=2, limb_darkening=False)
    min_wave = 4200
    max_wave = 10000
    wave, spec, v = star.calc_spectrum(min_wave=min_wave, max_wave=max_wave)
    
    norm = np.nanmax(spec)
    
    ax.plot(wave, spec/norm, color="tab:blue", label="No Limb Darkening Correction", lw=0.25)
    
    star = GridSpectrumSimulator(N_star=100, Teff=4500, logg=2, limb_darkening=True)
    wave, spec, v = star.calc_spectrum(min_wave=min_wave, max_wave=max_wave)
    ax.plot(wave, spec/norm, color="tab:red", label="Limb Darkening Correction", lw=0.25, alpha=0.7)
    
    ax.set_xlabel(r"Wavelength [$\AA$]")
    ax.set_ylabel("Normalized Flux")
    fig.set_tight_layout(True)
    plt.savefig("limb_dark_summed_spec.png", dpi=600)
    
    
if __name__ == "__main__":
    # plot_spectral_change()
    # plot_summed_spectral_change()
    # plot_stellar_disk_comparison()
    # plot_mean_limb_dark()
    
    plot_mu_comparison()