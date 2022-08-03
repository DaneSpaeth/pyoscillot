import numpy as np
from utils import gaussian
from plapy.constants import C
from three_dim_star import ThreeDimStar, TwoDimProjector
from physics import get_ref_spectra, get_interpolated_spectrum

class GridSpectrumSimulator():
    """ Simulate a spectrum of a star with a grid."""

    def __init__(self, N_star=500, N_border=5, Teff=4800, logg=3.0, feh=0.0,
                 v_rot=3000, inclination=90, limb_darkening=True):
        """ Initialize grid.

            :param int N_star: number of grid cells on the star in x and y
                               (will automatically be adjusted to be odd
                                to have a center coordinate)
            :param int N_border: number of pixels at the border (left and right
                                 each)
            :param int Teff: Effective Temperature [K] (must be available)
            :param int v_rot: V*sin(i) [m/s]
        """
        self.three_dim_star = ThreeDimStar(Teff=Teff, v_rot=v_rot)
        self.three_dim_star.create_rotation(v_rot)
        # self.three_dim_star.add_granulation()
        self.projector = TwoDimProjector(self.three_dim_star,
                                         N=N_star,
                                         border=N_border,
                                         inclination=inclination,
                                         line_of_sight=True,
                                         limb_darkening=limb_darkening)
        self.logg = logg
        self.feh = feh
        self.spectrum = None
        self.flux = None

    def add_spot(self, phase=0.25, theta_pos=90, radius=25, T_spot=4300):
        """ Add a circular starspot at position x,y.

            :param phase: Phase ranging from 0 to 1 (0 being left edge,
                                                     1 being one time around)
            :param radius: Radius in degree
            :param altitude: Altitude on star in degree (90 is on equator)
            :param deltaT: Temp difference to Teff in K
            :param bool reset: If True, reset the temp before adding another spot
        """
        phi = phase * 360.
        self.three_dim_star.add_spot(radius, phi_pos=phi, theta_pos=theta_pos, T_spot=T_spot)

    def add_granulation(self, dT=500, dv=1000, granule_size=2):
        """ Add a random granulation pattern to the star.
        
            :param float dT: Temperature variation in K
            :param float dv: Velocity variation in m/s
            :param float granule_size: Spatial granule size in degree
        """
        self.three_dim_star.add_granulation(dT=dT, dv=dv, granule_size=granule_size)

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
                                            spec_intensity=False)
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
        # self.granulation = self.projector.granulation_velocity()
        self.granulation = np.zeros(self.pulsation.shape)


        T_precision_decimals = 0
        rest_wavelength, total_spectrum, v_total = _compute_spectrum(self.temperature,
                                                                     self.rotation,
                                                                     self.pulsation,
                                                                     self.granulation,
                                                                     rest_wavelength,
                                                                     ref_spectra,
                                                                     ref_headers,
                                                                     T_precision_decimals)
        self.spectrum = total_spectrum

        # Also calculate the flux
        self.calc_flux()

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
            "intensity_stefan_boltzmann": self.projector.intensity_stefan_boltzmann()}
            #"granulation_velocity":self.projector.granulation_velocity()}

        return array_dict

    def add_pulsation(self, t=0, l=1, m=1, nu=1 / 600, v_p=1, k=100,
                      T_var=0, T_phase=0):
        """ Add a pulsation to the star."""
        # TODO make these values adjustable
        # t = phase / self.three_dim_star.nu
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

# @jit(nopython=True)
def _compute_spectrum(temperature, rotation, pulsation, granulation,
                      rest_wavelength, ref_spectra, ref_headers, T_precision_decimals):
    """ Compute the spectrum.

        Does all the heavy lifting
    """

    # Calc v over c
    v_c_rot = rotation / C
    v_c_pulse = pulsation / C
    v_c_gran = granulation / C
    total_spectrum = np.zeros(len(rest_wavelength))


    # We have a new approach, instead of precalculating all fine ref spectra which leads to a lot of memory overhead
    # we sort the temperature array and round it to the T_precision. Next compute the temperature adjusted
    # spectrum for all cells at this temperature and compute the velocity adjustment

    # First flatten all arrays
    temperature = temperature.flatten()
    v_c_rot = v_c_rot.flatten()
    v_c_pulse = v_c_pulse.flatten()
    v_c_gran = v_c_gran.flatten()

    sorted_temp_indices = np.argsort(temperature)
    sorted_temperature = temperature[sorted_temp_indices]
    sorted_v_c_rot = v_c_rot[sorted_temp_indices]
    sorted_v_c_pulse = v_c_pulse[sorted_temp_indices]
    sorted_v_c_gran = v_c_gran[sorted_temp_indices]

    sorted_temperature = np.round(sorted_temperature, decimals=T_precision_decimals)

    v_total = np.nanmean(pulsation)
    fine_ref_spectrum = None
    fine_ref_temperature = None

    # total_cells = len(sorted_temperature)
    # done = 0

    for temp, v_c_r, v_c_p, v_c_g in zip(sorted_temperature, sorted_v_c_rot, sorted_v_c_pulse, sorted_v_c_gran):
        # print(f"Calc cell {done}/{total_cells}")
        # done += 1
        if np.isnan(temp):
            continue
        # first check if you have a current ref spectrum
        if fine_ref_spectrum is None or temp != fine_ref_temperature:
            # If not compute a current ref spectrum
            fine_ref_temperature = temp
            print(f"Compute new fine_ref_spectrum for Temp={temp}K")
            # This one will automatically be kept in memory until all cells with this temperature are completed
            _, fine_ref_spectrum, _ = get_interpolated_spectrum(temp,
                                                          ref_wave=rest_wavelength,
                                                          ref_spectra=ref_spectra,
                                                          ref_headers=ref_headers)

        local_spectrum = fine_ref_spectrum.copy()

        if not v_c_r and not v_c_p and not v_c_g:
            # print(f"Skip Star Element {row, col}")
            total_spectrum += local_spectrum
        else:
            # print(f"Calculate Star Element {row, col}")
            local_wavelength = rest_wavelength + \
                               v_c_r * rest_wavelength + \
                               v_c_g * rest_wavelength + \
                               v_c_p * rest_wavelength

            # Interpolate the spectrum to the same rest wavelength grid
            interpol_spectrum = np.interp(rest_wavelength, local_wavelength, local_spectrum)

            total_spectrum += interpol_spectrum

    return rest_wavelength, total_spectrum, v_total


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    star = GridSpectrumSimulator(N_star=300, N_border=1, v_rot=3000, limb_darkening=False, inclination=60)
    star.add_spot(radius=5, theta_pos=180)
    star.add_spot(radius=5, theta_pos=150)
    star.add_spot(radius=5, theta_pos=120)
    star.add_spot(radius=5, theta_pos=90)
    star.add_spot(radius=5, theta_pos=60)
    star.add_spot(radius=5, theta_pos=30)
    star.add_spot(radius=5, theta_pos=0)


    fig, ax = plt.subplots(1)
    ax.imshow(star.projector.temperature(), origin="lower")
    plt.show()

