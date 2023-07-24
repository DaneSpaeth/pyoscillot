import matplotlib.pyplot as plt
import numpy as np
from dataloader import phoenix_spectrum
from astropy.convolution import Gaussian1DKernel
from physics import delta_relativistic_doppler
from astropy.convolution import convolve_fft
import cfg
import time

def add_isotropic_convective_broadening(wave, spec, v_macro, wave_dependent=True, debug_plot=False):
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
        sigma_px = delta_wave / pixel_scale
        
        print(sigma_px)
        
        
        dpx = 30
        spec_conv =np.zeros_like(spec)
        
        # print(sigma_px[1000:1100])
        # exit()
        for i in range(300, len(wave) - 300, 100):
            print(i, len(spec_conv))
            spec_local = spec[i - 130:i + 130 + 1]
            print(len(spec_local))
            kernel = Gaussian1DKernel(stddev=sigma_px[i])
            spec_conv_local = convolve_fft(spec_local, kernel)
            spec_conv[i-100:i+100] = spec_conv_local[30:-31]    
            
            
    
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
            ax.set_ylabel("Flux [erg/s/cm^2/cm]")
            ax.legend()
            fig.set_tight_layout(True)
            plt.savefig(f"{out_root}/{savename}", dpi=600)
        
        
        plt.close()
    
    return spec_conv

if __name__ == "__main__":
    wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(3550, 7500))
    start = time.time()
    spec_conv = add_isotropic_convective_broadening(wave, spec, 5000, wave_dependent=False)
    stop = time.time()
    
    print(round(stop- start, 3))
    fig, ax = plt.subplots(1, figsize=(30, 9))
    ax.plot(wave, spec, lw=0.5)
    ax.plot(wave, spec_conv, lw=0.5)
    fig.set_tight_layout(True)
    plt.savefig("dbug.png")