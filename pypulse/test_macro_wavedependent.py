import matplotlib.pyplot as plt
import numpy as np
from dataloader import phoenix_spectrum
from astropy.convolution import Gaussian1DKernel
from physics import delta_relativistic_doppler
from astropy.convolution import convolve_fft
import cfg
import time

def add_isotropic_convective_broadening(wave, spec, v_macro, wave_dependent=True, debug_plot=False, wave_step=1.0):
    """ Add the effect of macroturbulence, i.e. convective broadening, via convolution.
    
        This function assumes an isotropic broadening term, i.e. a constant
        convolution kernel across the stellar disk.
        
        :param np.array wave: Wavelength array in Angstrom
        :param np.array spec: Spectrum array 
        :param float v_macro: Macroturbulent velocity (eta) in m/s
    """
    print("Add isotropic macroturbulence")
    if not wave_dependent:
        center_idx = int(len(wave) / 2)
        delta_wave = delta_relativistic_doppler(wave[center_idx], v_macro)
        # this corresponds to the FWHM of the Gaussian kernel, so we need the conversion factor
        delta_wave /= 2*np.sqrt(2*np.log(2))
        pixel_scale = wave[center_idx] - wave[center_idx - 1]
        
        #TODO: check if pixel scale is constant
        
        
        
        sigma_px = delta_wave / pixel_scale
        kernel = Gaussian1DKernel(stddev=sigma_px)
        spec_conv = convolve_fft(spec, kernel)
    else:
        center_idx = int(len(wave) / 2)
        delta_wave = delta_relativistic_doppler(wave, v_macro)
        delta_wave /= 2*np.sqrt(2*np.log(2))
        pixel_scale = wave[1:] - wave[:-1]
        pixel_scale = np.append(pixel_scale, pixel_scale[-1])
        
        # The pixel scale is contant bus has jumps at 5000, 10000 and 15000 A
        sigma_px = delta_wave / pixel_scale
        
        spec_conv = np.zeros_like(spec)
        
        # mask = np.logical_and(wave>4999, wave<5001)
        
        
        # px_step = 100
        max_sigma = np.max(sigma_px)
        # Should be around 
        px_over = int(round(10*max_sigma, -1))
        for i in range((px_step+px_over), len(wave) - (px_step+px_over), px_step):
            print(i, len(spec_conv))
            spec_local = spec[i - (px_step+px_over):i + (px_step+px_over) + 1]
            print(len(spec_local))
            kernel = Gaussian1DKernel(stddev=sigma_px[i])
            spec_conv_local = convolve_fft(spec_local, kernel)
            spec_conv[i-px_step:i+px_step] = spec_conv_local[px_over:-(px_over+1)]    
            
            
    
    if debug_plot:
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = Path("/home/dspaeth/pypulse/data/plots/macroturbulence/")
        savename = f"macroturbulence.png"
        outfile = out_root / savename
        # Only save one debug plot (otherwise you would have that for every cell)
        if not outfile.is_file():    
            fig, ax = plt.subplots(1, figsize=(6.35, 3.5))
            ax.plot(wave, spec, label="Simulated Spectrum")
            ax.plot(wave, spec_conv, label=f"Broadend Spectrum by v_macro={v_macro}m/s")
            ax.set_xlim(wave[center_idx]-5, wave[center_idx]+5)
            ax.set_xlabel(r"Wavelength $[\AA]$")
            ax.set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
            ax.legend()
            fig.set_tight_layout(True)
            plt.savefig(f"{out_root}/{savename}", dpi=600)
        
        
        plt.close()
    
    return spec_conv

if __name__ == "__main__":
    wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(3500, 17500))
    
    # spec_conv_no_wave = add_isotropic_convective_broadening(wave, spec, 5000, wave_dependent=False)
    # np.save("spec_conv_no_wave.npy", spec_conv_no_wave)
    spec_conv_wave = add_isotropic_convective_broadening(wave, spec, 5000, wave_dependent=True)
    np.save("spec_conv_wave.npy", spec_conv_wave)
    # spec_conv_wave_px = add_isotropic_convective_broadening(wave, spec, 5000, wave_dependent=True, px_step=1)
    # np.save("spec_conv_wave_px.npy", spec_conv_wave_px)
    
    spec_conv_no_wave = np.load("spec_conv_no_wave.npy")
    spec_conv_wave = np.load("spec_conv_wave.npy")
    spec_conv_wave_px = np.load("spec_conv_wave_px.npy")
    

    fig, ax = plt.subplots(2, 1, figsize=(6.35, 3.5), sharex=True)
    ax[0].plot(wave, spec, color="tab:grey", label="PHOENIX Spectrum")
    # ax[0].plot(wave, spec_conv_no_wave, color="tab:orange")
    ax[0].plot(wave, spec_conv_wave, marker="*", color="tab:blue", label="100 pixel bins")
    ax[0].plot(wave, spec_conv_wave_px, marker=".", color="tab:red", alpha=0.7, linestyle="--", label="1 pixel bins")
    ax[0].legend(loc="lower left")
    
    
    mask = np.logical_and(wave >= 6000, wave < 6010)
    max_in_range = np.max(spec_conv_wave[mask])
    ax[1].plot(wave,(spec_conv_wave-spec_conv_wave_px)/max_in_range, color="tab:red")
    ax[1].set_ylim(-0.0001, 0.0001)
    ax[1].set_xlim(4990, 5010)
    ax[0].set_ylim(bottom=0)
    
    ax[1].set_xlabel(r"Wavelength $[\AA]$")
    ax[0].set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
    ax[1].set_ylabel("Relative Difference")
    ax[1].set_yticks([-0.0001, -0.00005, 0.0, 0.00005])
    figsize=(6.35, 3.5)
    # fig.set_tight_layout(True)
    fig.subplots_adjust(left=0.15, top=0.95, right=0.99, bottom=0.15, hspace=0)
    fig.align_ylabels()
    plt.savefig("dbug.png",dpi=500)