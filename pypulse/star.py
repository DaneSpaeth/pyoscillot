import numpy as np
import matplotlib.pyplot as plt
from utils import gaussian, get_ref_spectra, add_limb_darkening, add_isotropic_convective_broadening
from plapy.constants import C
from three_dim_star import ThreeDimStar, TwoDimProjector
from physics import get_interpolated_spectrum, delta_relativistic_doppler
from dataloader import Zhao_bis_polynomials
from utils import remove_phoenix_bisector, add_bisector, calc_mean_limb_dark, oversampled_wave_interpol
from CB_models import simple_Pollux_CB_model, simple_alpha_boo_CB_model, simple_ngc4349_CB_model
import copy
import cfg
import photometry
# Try to solve the memory leak but not working?
import matplotlib
matplotlib.use('Agg')

class GridSpectrumSimulator():
    """ Simulate a spectrum of a star with a grid."""

    def __init__(self, N_star=500, N_border=5, Teff=4800, logg=3.0, feh=0.0,
                 v_rot=3000, inclination=90, limb_darkening=True,
                 convective_blueshift=False, convective_blueshift_model="alpha_boo", 
                 v_macro=0):
        """ Initialize grid.

            :param int N_star: number of grid cells on the star in x and y
                               (will automatically be adjusted to be odd
                                to have a center coordinate)
            :param int N_border: number of pixels at the border (left and right
                                 each)
            :param int Teff: Effective Temperature [K] (must be available)
            :param int v_rot: V*sin(i) [m/s]
            :param bool convective_blueshift: Add in the conv blueshift Bisectors
            :param str convective_blueshift_model: Which convective blueshift model to use. At the moment [alpha_boo, sun]
            :param int v_macro: (Isotropic) macroturbulent velocity [m/s]
        """
        self.three_dim_star = ThreeDimStar(Teff=Teff, v_rot=v_rot)
        self.three_dim_star.create_rotation(v_rot)
        # self.three_dim_star.add_granulation()
        self.projector = TwoDimProjector(self.three_dim_star,
                                         N=N_star,
                                         border=N_border,
                                         inclination=inclination,
                                         line_of_sight=True)
        self.logg = logg
        self.feh = feh
        self.spectrum = None
        self.flux = None
        self.V_band_flux = None
        self.conv_blue = convective_blueshift
        self.conv_blue_model = convective_blueshift_model
        self.limb_dark = limb_darkening
        self.v_macro = v_macro
        
        self.mean_limb_dark = None

    def add_spot(self, phase=0.25, theta_pos=90, radius=25, T_spot=4300):
        """ Add a circular starspot at position x,y.

            :param phase: Phase ranging from 0 to 1 (0 being left edge,
                                                     1 being one time around)
            :param radius: Radius in degree
            :param theta_pos: Altitude on star in degree (90 is on equator)
            :param T_spot: Spot temperature in K
        """
        phi = phase * 360.
        self.three_dim_star.add_spot(radius, phi_pos=phi, theta_pos=theta_pos, T_spot=T_spot)

    def add_granulation(self, dT=500, dv=1000, granule_size=2):
        """ Add a random granulation pattern to the star.
        
            :param float dT: Temperature variation in K
            :param float dv: Velocity variation in m/s
            :param float granule_size: Spatial granule size in degree
        """
        self.three_dim_star.add_granulation()

    def _get_ref_spectra_for_mode(self, mode, wavelength_range):
        """ Get a wavelength grid and dictionarys of the available
        Phoenix spectra that are close to the temperature
        """
        if mode == "phoenix" or mode == "oneline":
            (rest_wavelength,
             ref_spectra,
             ref_headers) = get_ref_spectra(self.temperature,
                                            logg=self.logg,
                                            feh=self.feh,
                                            wavelength_range=wavelength_range,
                                            spec_intensity=False,
                                            change_bis=self.conv_blue)
            ref_mu = None
        elif mode == "spec_intensity":
            (rest_wavelength,
             ref_spectra,
             ref_headers,
             ref_mu) = get_ref_spectra(self.temperature,
                                       logg=self.logg,
                                       feh=self.feh,
                                       wavelength_range=wavelength_range,
                                       spec_intensity=True)
        elif mode == "gaussian":
            print("Gaussian mode")
            # Does not work with Temperature variations at the moment
            center = 7000  # A

            rest_wavelength = np.linspace(
                center - 0.7, center + 0.7, 10000)
            spec = 1 - gaussian(rest_wavelength, center, 0.2)
            ref_spectra = {self.Teff: spec}
            ref_headers = {self.Teff: []}
            ref_mu = None

        return rest_wavelength, ref_spectra, ref_headers, ref_mu

    def calc_spectrum(self, min_wave=5000, max_wave=12000, mode="phoenix"):
        """ Return Spectrum (potentially Doppler broadened) from min to max.

            :param float min_wave: Minimum Wavelength (Angstrom)
            :param float max_wave: Maximum Wavelength (Angstrom)

            :returns: Array of wavelengths, array of flux value
        """

        wavelength_range = (min_wave - 0.25, max_wave + 0.25)

        self.temperature = self.projector.temperature()

        rest_wavelength, ref_spectra, ref_headers, ref_mu = self._get_ref_spectra_for_mode(mode, wavelength_range)

        self.rotation = self.projector.rotation()
        self.pulsation = self.projector.pulsation()
        self.mu = self.projector.mu()
        # self.granulation = self.projector.granulation_velocity()
        self.granulation = np.zeros(self.pulsation.shape)
        
        if self.limb_dark:
            # Calculate (or preaload if existing) an averaged limb darkening array
            self.mean_limb_dark = calc_mean_limb_dark(rest_wavelength, self.mu)
            
            debug_plot = False
            if debug_plot:
                fig, ax = plt.subplots(1, figsize=cfg.figsize)
                ax.plot(rest_wavelength, self.mean_limb_dark, label="Averaged Limb Darkening Profile")
                ax.set_xlabel(r"Wavelength [$\AA$]")
                ax.set_ylabel("Normalized Flux")
                ax.legend()
                fig.set_tight_layout(True)
                # Add saving to fake_spectra
                plt.savefig("LD_profile.png", dpi=600)
            

        T_precision_decimals = 1
        weights = self.projector.weights
        rest_wavelength, total_spectrum, v_total = _compute_spectrum(self.temperature,
                                                                     self.rotation,
                                                                     self.pulsation,
                                                                     self.granulation,
                                                                     self.mu, 
                                                                     rest_wavelength,
                                                                     ref_spectra,
                                                                     ref_headers,
                                                                     T_precision_decimals,
                                                                     self.logg, 
                                                                     self.feh,
                                                                     change_bis=self.conv_blue,
                                                                     convective_blueshift_model=self.conv_blue_model, 
                                                                     limb_dark=self.limb_dark,
                                                                     v_macro=self.v_macro,
                                                                     mean_limb_dark=self.mean_limb_dark,
                                                                     weights=weights)
        self.spectrum = total_spectrum
        self.wavelength = rest_wavelength

        # Also calculate the fluxes
        self.calc_flux()
        self.calc_V_flux()

        return rest_wavelength, total_spectrum, v_total

    def get_arrays(self):
        """ Get all arrays (e.g. pulsation, temp) of the simulation.

            Return all arrays as a dictionary.

            Useful for creating movies afterwards.
        """
        array_dict = {
            "pulsation_rad": self.projector.pulsation_rad(),
            "pulsation_phi": self.projector.pulsation_phi(),
            "pulsation_theta": self.projector.pulsation_theta(),
            "pulsation": self.projector.pulsation(),
            "temperature": self.projector.temperature(),
            "rotation": self.projector.rotation(),
            "intensity_stefan_boltzmann": self.projector.intensity_stefan_boltzmann(),
            "spectrum":self.spectrum,
            "wavelength":self.wavelength}
            #"granulation_velocity":self.projector.granulation_velocity()}

        return array_dict

    def add_pulsation(self, t=0, l=1, m=1, nu=1 / 600, v_p=1, k=100,
                      T_var=0, T_phase=0, refbjd=0):
        """ Add a pulsation to the star.
        
            Also adjust the phase here.
        """
        # TODO make these values adjustable
        # t = phase / self.three_dim_star.nu
        # Adjust the phase, i.e. reftime corresponds to phase 0
        t -= refbjd
        self.three_dim_star.add_pulsation(t=t, l=l, m=m, nu=nu, v_p=v_p, k=k,
                                          T_var=T_var, T_phase=T_phase)

    def calc_flux(self):
        """ Calculate the local flux of the star.

            Sum up the total flux of the spectrum that you calculated.
        """
        if self.spectrum is None:
            return 0

        self.flux = np.sum(self.spectrum)
        return self.flux
    
    def calc_V_flux(self):
        """ Calculate the flux in the V band filter
        """
        self.V_band_flux = photometry.V_band_flux(self.wavelength, self.spectrum)
        

def _compute_spectrum(temperature, rotation, pulsation, granulation, mu, 
                      rest_wavelength, ref_spectra, ref_headers, T_precision_decimals,
                      logg, feh, 
                      change_bis=False, convective_blueshift_model="alpha_boo", 
                      limb_dark=False, v_macro=0, mean_limb_dark=None, weights=None):
    """ Compute the spectrum.

        Does all the heavy lifting
    """
    if weights is None:
        print("USE UNIFORM WEIGHTS")
        weights = np.ones_like(temperature)

    # Calc v over c
    v_c_rot = rotation / C
    v_c_pulse = pulsation / C
    v_c_gran = granulation / C
    total_spectrum = np.zeros(len(rest_wavelength))

    # We have a new approach, instead of precalculating all fine ref spectra which leads to a lot of memory overhead
    # we sort the temperature array and round it to the T_precision. Next compute the temperature adjusted
    # spectrum for all cells at this temperature and compute the velocity adjustment
    
    # Unfortunately we now have to change that a bit to allow the BIS mu dependence
    # We have the mu values given in the steps
    # [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]

    # First flatten all arrays
    temperature = temperature.flatten()
    v_c_rot = v_c_rot.flatten()
    v_c_pulse = v_c_pulse.flatten()
    v_c_gran = v_c_gran.flatten()
    mu = mu.flatten()
    weights = weights.flatten()

    sorted_temp_indices = np.argsort(temperature)
    sorted_temperature = temperature[sorted_temp_indices]
    sorted_v_c_rot = v_c_rot[sorted_temp_indices]
    sorted_v_c_pulse = v_c_pulse[sorted_temp_indices]
    sorted_v_c_gran = v_c_gran[sorted_temp_indices]
    sorted_mus = mu[sorted_temp_indices]
    sorted_weights = weights[sorted_temp_indices]

    sorted_temperature = np.round(sorted_temperature, decimals=T_precision_decimals)
    
    # The mu values that are available for the BIS calculations of convective_blueshift
    if change_bis:
        # available_mus = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00])
        if convective_blueshift_model == "alpha_boo":
            available_mus = np.array(list(ref_spectra[list(ref_spectra.keys())[0]].keys()))
        elif convective_blueshift_model == "sun":
            available_mus = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00])
        print(f"Available mus = {available_mus}")
        rounded_mus = [available_mus[np.argmin(np.abs(m - available_mus))] for m in sorted_mus]
        rounded_mus = np.array(rounded_mus)
    else:
        rounded_mus = np.ones_like(sorted_temperature)
        
    v_total = np.nanmean(pulsation)
    fine_ref_spectra_dict = None
    fine_ref_temperature = None
    
    num_skipped_nans_pulsation = (np.count_nonzero(~np.isnan(sorted_temperature)) -
                                  np.count_nonzero(~np.isnan(sorted_v_c_pulse)))
    print(f"{num_skipped_nans_pulsation} will be skipped due to NaNs in the pulsation! Probably at the pole!")
    weight_sum = 0

    for temp, v_c_r, v_c_p, v_c_g, rounded_mu, mu, weight in zip(sorted_temperature, 
                                            sorted_v_c_rot, 
                                            sorted_v_c_pulse,
                                            sorted_v_c_gran,
                                            rounded_mus,
                                            sorted_mus,
                                            sorted_weights):
        # print(f"Calc cell {done}/{total_cells}")
        # done += 1
        if np.isnan(temp):
            continue
        if np.isnan(v_c_r):
            print(f"Skip a cell since rotation is NaN")
            continue
        if np.isnan(v_c_p):
            print(f"Skip a cell since pulsation is NaN")
            continue
        if np.isnan(v_c_g):
            print(f"Skip a cell since granulation is NaN")
            continue
        if np.isnan(weight):
            print("Skip a cell since the weight is NaN")
            continue
        
        # Let's make the fine_ref_spectrum a dictionary of all needed mu angles
        
        
        
        # first check if you have a current ref spectrum
        if fine_ref_spectra_dict is None or temp != fine_ref_temperature:
            
            # Calculate all mu angles that are needed
            # If not compute a current ref spectrum
            fine_ref_temperature = temp
            needed_mus = np.unique(rounded_mus[sorted_temperature==fine_ref_temperature])
            
            
            print(f"Needed mu angles for temperature {fine_ref_temperature}={needed_mus}")
            # print(f"Compute new fine_ref_spectrum for Temp={temp}K")
            # We get a dictionary of {mu:spec} for each temperature
            _, fine_ref_spectra_dict, _ = get_interpolated_spectrum(temp,
                                                                    ref_wave=rest_wavelength,
                                                                    ref_spectra=ref_spectra,
                                                                    ref_headers=ref_headers,
                                                                    mu_angles=needed_mus,
                                                                    logg=logg,
                                                                    feh=feh,
                                                                    interpolation_mode="cubic_spline")
            
            # We now have T interpolated PHOENIX spectra
            # We choose to apply the LD correction after the T interpolation to allow
            # an easily precomputed spline interpolation
            # Now we remove the mean LD from the interpolated spectra
            if limb_dark:
                # Now correct the fine_ref_spectra
                print(f"Apply mean limb darkening correction for fine_ref_spectra at T {temp}")
                for mu, spectrum in fine_ref_spectra_dict.items():
                    spectrum_LD_removed = spectrum / mean_limb_dark
                    fine_ref_spectra_dict[mu] = spectrum_LD_removed
                    
                debug_plot = False
                if debug_plot:
                    fig, ax = plt.subplots(1, figsize=cfg.figsize)
                    ax.plot(rest_wavelength, spectrum, label=f"Original PHOENIX Spectrum (T={temp}K)")
                    ax.plot(rest_wavelength, fine_ref_spectra_dict[1.0], label="Mean Limb Darkening Removed Spectrum", alpha=0.7)
                    ax.set_xlabel(r"Wavelength [$\AA$]")
                    ax.set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
                    ax.set_ylim(0, ax.get_ylim()[1]*1.15)
                    ax.legend()
                    fig.set_tight_layout(True)
                    plt.savefig("LD_removed_spectrum.png", dpi=600)
                    plt.close(fig)
            
            # First add in the macroturbulence
            if v_macro:
                for mu, spec in fine_ref_spectra_dict.items():
                    fine_ref_spectra_dict[mu] = add_isotropic_convective_broadening(rest_wavelength, 
                                                                                    spec,
                                                                                    v_macro=v_macro, 
                                                                                    debug_plot=False, 
                                                                                    per_pixel=True, 
                                                                                    convolution=False,
                                                                                    old=False)
                    
            # NOTE: We assume that the macroturbulence correction leaves the continuum largely
            # unchanged so that we don't need to adjust the continuum
                    
            # If the limb darkening correction was applied this will have affected the continuum
            # Since we load precomputed continuum corrections that were calculated for raw
            # PHOENIX spectra we now need to keep track of the continuum correction of the LD
            
            if change_bis:
                # First remove the PHOENIX bisector
                spec_corr, _, _, _, _, _ = remove_phoenix_bisector(rest_wavelength,
                                                                   fine_ref_spectra_dict[rounded_mu],
                                                                   fine_ref_temperature,
                                                                   logg,
                                                                   feh,
                                                                   limb_dark_continuum=mean_limb_dark)
                if convective_blueshift_model == "alpha_boo":
                    bis_polynomial_dict = simple_alpha_boo_CB_model()
                elif convective_blueshift_model == "sun":
                    bis_polynomial_dict = Zhao_bis_polynomials()
                # spec_corr = fine_ref_spectra_dict[rounded_mu]
                
                for mu in fine_ref_spectra_dict.keys():
                    spec_add, _, _, _, _ = add_bisector(rest_wavelength, 
                                                        copy.deepcopy(spec_corr), 
                                                        bis_polynomial_dict[mu],
                                                        fine_ref_temperature, 
                                                        logg, 
                                                        feh, 
                                                        debug_plot=True,
                                                        mu=mu)
                    
                    fine_ref_spectra_dict[mu] = spec_add
                    
            

        local_spectrum = fine_ref_spectra_dict[rounded_mu].copy()
        

        if not v_c_r and not v_c_p and not v_c_g:
            if limb_dark:
                _, local_spectrum = add_limb_darkening(rest_wavelength, local_spectrum, mu)
            # print(f"Skip Star Element {row, col}")
            # Also add in the weight (if cell is only partially on the star)
            local_spectrum *= weight
            if not np.isnan(local_spectrum).any():
                total_spectrum += local_spectrum
            else:
                print("Skip Cell since there is a NaN in the local spectrum")
                print(np.isnan(local_spectrum).all())
                print(f"Weight is nan: {np.isnan(weight)}")
        else:
            # print(f"Calculate Star Element {row, col}")
            # local_wavelength = rest_wavelength + \
            #                    v_c_r * rest_wavelength + \
            #                    v_c_g * rest_wavelength + \
            #                    v_c_p * rest_wavelength
            v_c_tot = v_c_r + v_c_g + v_c_p
            local_wavelength = rest_wavelength + delta_relativistic_doppler(rest_wavelength,
                                                                            v_c=v_c_tot)

            # Interpolate the spectrum to the same rest wavelength grid
            # interpol_spectrum = np.interp(rest_wavelength, local_wavelength, local_spectrum)
            interpol_spectrum = oversampled_wave_interpol(rest_wavelength, local_wavelength, local_spectrum)
            if limb_dark:
                _, interpol_spectrum = add_limb_darkening(rest_wavelength, interpol_spectrum, mu)
            # Also add in the weight (if cell is only partially on the star)
            interpol_spectrum *= weight
            if not np.isnan(interpol_spectrum).any():
                total_spectrum += interpol_spectrum
                weight_sum += weight
            else:
                print("Skip Cell since there is a NaN in the local spectrum")
                print(np.isnan(interpol_spectrum).all())
                print(f"Weight is nan: {weight}")
                print(f"v_c_tot: {v_c_tot}")
            
    
    print(f"DIVIDE BY TOTAL WEIGHT {weight_sum}")
    total_spectrum /= weight_sum

    return rest_wavelength, total_spectrum, v_total


if __name__ == "__main__":
    print("Test")
    Teff = 4500
    logg = 2
    feh = 0.0
    star = GridSpectrumSimulator(N_star=150, Teff=Teff, logg=logg, feh=feh, limb_darkening=False, convective_blueshift=True, v_macro=0)
    star.add_pulsation()
    wave, spec, _ = star.calc_spectrum(min_wave=3600, max_wave=7150)
    
    print(spec)
    
    
    #### To TEST the Limb Darkening ####
    # Teff = 4500
    # logg = 2
    # feh = 0.0
    # star = GridSpectrumSimulator(N_star=150, Teff=Teff, logg=logg, feh=feh, limb_darkening=True, convective_blueshift=True, v_macro=0)
    # wave, spec_LD, _ = star.calc_spectrum(min_wave=3600, max_wave=7150)
    
    # star = GridSpectrumSimulator(N_star=150, Teff=Teff, logg=logg, feh=feh, limb_darkening=False, convective_blueshift=True, v_macro=0)
    # wave, spec, _ = star.calc_spectrum(min_wave=3600, max_wave=7150)
    
    # fig, ax = plt.subplots(2, 1, figsize=cfg.figsize, sharex=True)
    # ax[0].plot(wave, spec_LD, alpha=0.7, label="Limb Darkening Correction")
    # ax[0].plot(wave, spec, label="No Limb Darkening Correction")
    # ax[0].legend()
    # ax[1].plot(wave, (spec_LD - spec) / spec)
    # ax[1].set_xlabel(r"Wavelength [$\AA$]")
    # ax[0].set_ylabel(r"Flux $\left[ \frac{\mathrm{erg}}{\mathrm{s\ cm\ cm^2}} \right]$")
    # ax[1].set_ylabel(r"$\Delta$ Flux [%]")
    # ax[0].set_ylim(0, ax[0].get_ylim()[1]*1.15)
    # # fig.set_tight_layout(True)
    # fig.subplots_adjust(hspace=0, wspace=0, left=0.15, right=0.99, bottom=0.15, top=0.99)
    # plt.savefig("LD_comparison.png", dpi=600)
    
    
    
    
    
    
    ###### To test the macroturbulence #####
    # from utils import measure_bisector_on_line, normalize_phoenix_spectrum_precomputed
    # fig, ax = plt.subplots(1, 2)
    # PER_PIXEL = False
    # for v_macro, color, label, marker in zip([0, 5000], ("tab:blue", "tab:orange"), ("v_macro=0", "v_macro=5000"), ("x", "o")):
    #     # PER_PIXEL = b
    #     Teff = 4500
    #     logg = 2
    #     feh = 0.0
    #     line = 5088.84
    #     star = GridSpectrumSimulator(N_star=100, Teff=Teff, logg=logg, feh=feh, limb_darkening=False, convective_blueshift=True, v_macro=v_macro)
    #     # star.add_pulsation(l=1, m=1, v_p=0.7, k=2439, T_var=34.0)
    #     wave, spec, v = star.calc_spectrum(min_wave=4900, max_wave=5300)

    #     mask = np.logical_and(wave>5088.7, wave<5089)
    #     wave = wave[mask]
    #     spec = spec[mask]
    #     spec_norm = spec / np.max(spec)
    #     bis_wave, bis_v, bis = measure_bisector_on_line(wave, spec_norm, line)
        
    #     ax[0].plot(wave, spec_norm, marker=marker, color=color, label=label)
    #     ax[0].plot(bis_wave, bis, color=color)
        
    #     ax[1].plot(bis_v, bis, color=color)
    # ax[0].legend()
    # plt.savefig("dbug.png", dpi=500)
    
    