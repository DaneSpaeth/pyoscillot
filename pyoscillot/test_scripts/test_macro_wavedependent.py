import matplotlib.pyplot as plt
import numpy as np
from dataloader import phoenix_spectrum
from astropy.convolution import Gaussian1DKernel
from physics import delta_relativistic_doppler
from astropy.convolution import convolve_fft, convolve
import cfg
import time
from pathlib import Path

def gaussian(x, mu, sig):
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-1/2 * np.square((x - mu)/sig))

def add_isotropic_convective_broadening(wave, spec, v_macro, wave_dependent=True, debug_plot=False, wave_step=0.5, per_pixel=False, convolution=True, old=False):
    """ Add the effect of macroturbulence, i.e. convective broadening, via convolution.
    
        This function assumes an isotropic broadening term, i.e. a constant
        convolution kernel across the stellar disk.
        
        :param np.array wave: Wavelength array in Angstrom
        :param np.array spec: Spectrum array 
        :param float v_macro: Macroturbulent velocity (eta) in m/s
    """
    print(f"Add isotropic macroturbulence, wave_dependent={wave_dependent}, per_pixel={per_pixel}, convolution={convolution}")
    if not wave_dependent:
        center_idx = int(len(wave) / 2)
        delta_wave = delta_relativistic_doppler(wave[center_idx], v_macro)
        # this corresponds to the FWHM of the Gaussian kernel, so we need the conversion factor
        delta_wave /= 2*np.sqrt(2*np.log(2))
        pixel_scale = wave[center_idx] - wave[center_idx - 1]
        
        sigma_px = delta_wave / pixel_scale
        kernel = Gaussian1DKernel(stddev=sigma_px)
        spec_conv = convolve(spec, kernel)
    else:
        center_idx = int(len(wave) / 2)
        delta_wave = delta_relativistic_doppler(wave, v_macro)
        delta_wave /= 2*np.sqrt(2*np.log(2))
        
        # The pixel scale is constant but has jumps at 5000, 10000 and 15000 A
        scale_jumps = [0, 5000, 10000, 15000, 20000]
        pixel_scales_dict = {0: None,
                             5000: 0.006,
                             10000: 0.01,
                             15000: 0.03,
                             20000: None}
        scale_jumps = [sj for sj in scale_jumps if sj < wave[-1] + 5000 and sj > wave[0] - 5000]
        
        scale_jump_px = [(np.abs(wave-sj)).argmin() for sj in scale_jumps]
        
        last_idx = 0
        spec_conv = np.zeros_like(wave)
        for jump_interval, idx in enumerate(scale_jump_px):
            # Make arrays that run exactly to the jump but do not include it
            if jump_interval == 0:
                continue
            wave_local = wave[last_idx:idx]
            spec_local = spec[last_idx:idx]
            wave_start=wave_local[0]
            wave_stop=wave_local[-1]
            pixel_scale_local = pixel_scales_dict[scale_jumps[jump_interval]]
        
            
            delta_wave_local = delta_wave[last_idx:idx]
            sigma_px_local = delta_wave_local / pixel_scale_local
            
            
            # Let's first calculate the largest width in the current segment
            max_dw = np.max(delta_wave_local)
            # Convert it to pixel
            max_dpx = max_dw / pixel_scale_local
            # And define 50 times as a overhead
            if not per_pixel:
                px_step = int(wave_step / pixel_scale_local / 2) 
                px_over = int(np.ceil(max_dpx*50))
            else:
                px_step = 0
                px_over = int(np.ceil(max_dpx*50))
            
            spec_conv_local = np.zeros_like(wave_local)
            
            if not old:
                root = cfg.conf_dict["datapath"] / "macroturbulence_kernels"
                kernels_file = root / Path(f"kernels_v{v_macro:.1f}_w{wave_start}_{wave_stop}.npy")
                if kernels_file.is_file():
                    kernels = np.load(kernels_file)
                else:
                    # Lets try to precompute the kernels in an array
                    lin_px = np.linspace(-px_over, px_over, 2*px_over+1)
                    # kernels = np.array([gaussian(lin_px, 0., sigma) for sigma in sigma_px_local])
                    lin_px = np.array([lin_px for i in range(len(wave_local))])
                    sigma_px_local = np.array([sigma_px_local for i in range(2*px_over+1)]).T
                    kernels = gaussian(lin_px, 0., sigma_px_local)
                    np.save(kernels_file, kernels)
            
                spec_loops = []
                rowmask = np.zeros(kernels.shape[0], dtype=bool)
            
            for i in range(px_step, len(wave_local), max(px_step,1)):
                # print(f"\r{i}, {len(wave_local)}", end="")
                di_high = 0
                if (i - px_step - px_over) < 0:
                    if jump_interval == 1:
                        # Cannot interpolate
                        continue
                    di = i - px_step - px_over
                    # We have to interpolate into the last range
                    prev_interval_wave = wave[scale_jump_px[jump_interval-2]:scale_jump_px[jump_interval-1]]
                    prev_interval_spec = spec[scale_jump_px[jump_interval-2]:scale_jump_px[jump_interval-1]]
                    
                    lin_wave = np.linspace(wave_local[0] - np.abs(di)*pixel_scale_local,
                                            wave_local[0],
                                            np.abs(di))
                    interp_spec = np.interp(lin_wave, prev_interval_wave, prev_interval_spec)
                    # Now you have the interpolated spectrum in the new sampling range
                    # Now stitch together
                    spec_loop = interp_spec
                    spec_loop = np.append(spec_loop, spec_local[:i+px_step+px_over+1])
                elif (i + px_step + px_over) >= len(wave_local):
                    if not len(scale_jump_px) > jump_interval + 1:
                        # Cannot interpolate
                        continue
                    
                    di = i + px_step + px_over - len(wave_local) + 1
                    # We have to interpolate into the last range
                    next_interval_wave = wave[scale_jump_px[jump_interval]:scale_jump_px[jump_interval+1]]
                    next_interval_spec = spec[scale_jump_px[jump_interval]:scale_jump_px[jump_interval+1]]
                    
                    lin_wave = np.linspace(wave_local[-1] + pixel_scale_local,
                                            wave_local[-1] + np.abs(di)*pixel_scale_local, np.abs(di))
                    interp_spec = np.interp(lin_wave, next_interval_wave, next_interval_spec)
                    # Now you have the interpolated spectrum in the new sampling range
                    # Now stitch together
                    spec_loop = spec_local[i - px_step - px_over:]
                    spec_loop = np.append(spec_loop, interp_spec)
                    di_high = di - px_over
                    
                else:
                    spec_loop = spec_local[i - (px_step+px_over):i + (px_step+px_over) + 1]
                    
                
                if convolution:
                    kernel = Gaussian1DKernel(stddev=sigma_px_local[i])
                    spec_conv_loop = convolve(spec_loop, kernel)
                    
                    
                    if di_high > 0:
                        spec_conv_local[i-px_step:i+px_step+1] = spec_conv_loop[px_over:px_over+2*px_step+1-di_high]
                    else:
                        spec_conv_local[i-px_step:i+px_step+1] = spec_conv_loop[px_over:px_over+2*px_step+1]
                    
                else:
                    if old:
                        kernel = gaussian(np.linspace(-(px_step+px_over), (px_step+px_over), len(spec_loop)), 0., sigma_px_local[i])
                        spec_conv_loop = np.sum(spec_loop * kernel)
                        spec_conv_local[i] = spec_conv_loop
                    if not old:
                        rowmask[i] = True
                        spec_loops.append(spec_loop)
                    
            if not old:
                spec_loops = np.array(spec_loops)
                kernels = kernels[rowmask,:]
        
                spec_conv_local[rowmask] = np.sum(spec_loops * kernels, axis=1)
            spec_conv[last_idx:idx] = spec_conv_local
            last_idx = idx      
    
    if debug_plot:
        if cfg.debug_dir is not None:
            out_root = cfg.debug_dir
        else:
            out_root = Path("/home/dspaeth/pyoscillot/data/plots/macroturbulence/")
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
    wv_range_start = 3600
    wv_range_stop = 7150
    wave, spec, header = phoenix_spectrum(4500, 2.0, 0.0, wavelength_range=(wv_range_start, wv_range_stop))
    import copy
    
    # spec_conv_no_wave = add_isotropic_convective_broadening(wave, spec, 5000, wave_dependent=False)
    # np.save("spec_conv_no_wave.npy", spec_conv_no_wave)
    import time
    start = time.time()
    # spec_conv_wave = add_isotropic_convective_broadening(copy.deepcopy(wave),copy.deepcopy(spec), 5000, wave_dependent=True, per_pixel=False)
    # spec_conv_wave_old, wave_loop_old = add_isotropic_convective_broadening_old(copy.deepcopy(wave), copy.deepcopy(spec), 5000, wave_dependent=True, per_pixel=False)
    

    # np.save("spec_conv_wave_new.npy", spec_conv_wave)
    #spec_noconv_wave_px = add_isotropic_convective_broadening(wave, spec, 5000, wave_dependent=True, per_pixel=True, convolution=False, old=True)
    #stop = time.time()
    spec_conv_wave_px = add_isotropic_convective_broadening(wave, spec, 5000, wave_dependent=True, per_pixel=True, convolution=False)
    # spec_conv_wave_conv = add_isotropic_convective_broadening(wave, spec, 5000, wave_dependent=True, per_pixel=True, convolution=True)
    # print(len(spec_conv_wave_px))
    # np.save("spec_noconv_new.npy", spec_conv_wave_px)
    stop = time.time()
    print()
    print(round(stop-start, 2))
    # exit()
    # exit()
    # np.save("spec_conv_wave_px.npy", spec_conv_wave_px)
    
    # spec_conv_no_wave = np.load("spec_conv_no_wave.npy")
    # spec_conv_wave = np.load("spec_conv_wave_new.npy")
    # spec_conv_wave_px = np.load("spec_conv_wave_px.npy")
    
    # spec_noconv_wave_px = np.load("spec_noconv_before.npy")
    
    

    fig, ax = plt.subplots(1, figsize=(6.35, 3.5), sharex=True)
    ax.plot(wave, spec, color="tab:grey", label="PHOENIX Spectrum", marker=".", markersize=3)
    
    # print(len(wave))
    # print(len(spec_noconv_wave_px))
    # ax[0].plot(wave, spec_conv_no_wave, color="tab:orange")
    ax.plot(wave, spec_conv_wave_px, color="tab:blue", label=r"$\eta$=5000m/s", marker="*")
    # ax[0].plot(wave, spec_noconv_wave_px, marker=".", color="tab:red", alpha=0.7, linestyle="--", label="noconv_before")
    ax.legend(loc="lower left")
    
    xlim_low = wv_range_start + (wv_range_stop - wv_range_start) / 2 - 0.5
    xlim_high = xlim_low + 1
    
    xlim_low = 4999
    xlim_high = 5001
    mask = np.logical_and(wave >= xlim_low, wave < xlim_high)
    
    
    # max_in_range = np.max(spec_conv_wave_px[mask])
    # ax[1].plot(wave,(spec_conv_wave_px-spec_noconv_wave_px), color="tab:red", marker=".")
    # print(((spec_conv_wave_px-spec_noconv_wave_px)[mask]))
    # ax[1].set_ylim(-0.02, 0.00002)
    # ax[1].set_ylim(-0.00005, 0.00005)
    
    ax.set_xlim(xlim_low, xlim_high)
    ax.set_ylim(bottom=0)
    # ax[0].vlines(5000.0, ax[0].get_ylim()[0], ax[0].get_ylim()[1])
    
    ax.set_xlabel(r"Wavelength $[\AA]$")
    ax.set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
    # ax[1].set_ylabel("Relative Difference")
    # ax[1].set_yticks([-0.000015, -0.00001, -0.000005, 0.0, 0.000005, 0.00001, 0.000015])
    figsize=(6.35, 3.5)
    # fig.set_tight_layout(True)
    fig.subplots_adjust(left=0.11, top=0.95, right=0.96, bottom=0.13, hspace=0)
    fig.align_ylabels()
    # plt.rcParams['agg.path.chunksize'] = 10000
    plt.savefig("v_macro_new_around5000.png",dpi=500)