import numpy as np
import matplotlib.pyplot as plt
import constants as const
from dataloader import phoenix_spectrum
from spline_interpolation import interpolate_on_temperature

DIVIDING_TEMP = 5100

def energy_flux_to_photon_flux(wavelength, spectrum):
    """ Convert a sepctrum given in energy flux (such as a PHOENIX spectrum) to a photon flux.
    
        Give wavenlength in units of m
    """
    energy_photon = const.H * const.C / (wavelength)
    spectrum_photon = spectrum / energy_photon
    return spectrum_photon

def planck(wav, T):
    """ Return planck's law at wavelength and Temperature.

        :param float/np.array wav: Wavelength in m
        :param T: Temperature in K

        :returns intensity: in J/s sr-1 m⁻2 m⁻1
        
        erg/s/cm^2/cm'
    """
    a = 2.0 * const.H * const.C**2
    b = const.H * const.C / (wav * const.K_b * T)
    intensity = a / ((wav**5) * (np.exp(b) - 1.0))
    return intensity


def planck_ratio(wav, T_1, T_2):
    """ Return the ratio of the planck's law between T_1, T_2 per wavelength.

        R = S(T_1) / S(T_2)


        :param float/np.array wav: Wavelength in m
        :param T_1: Temp 1 in K
        :param T_2: Temp 2 in K
        :returns: int or np.array of ratio
    """
    int_1 = planck(wav, T_1)
    int_2 = planck(wav, T_2)
    return int_1 / int_2

def get_interpolated_spectrum(T_local,
                              ref_wave=None,
                              ref_spectra=None,
                              ref_headers=None,
                              mu_angles=None,
                              spec_intensity=False,
                              mu_local=1,
                              ref_mu=None,
                              logg=None,
                              feh=None,
                              interpolation_mode="cubic_spline"):
    """ Return a potentially interpolated spectrum. Returns the same format as
        the phoenix spectrum.
        
        :param float T_local: local Temperature
        :param np.array ref_wave: The wavelength array
        :param dict ref_spectra: A nested dictionary of {T:{mu:spec}}, 
                                 with spec being a np.array containing the spectral fluxes
        :param dict ref_headers: A dictionary of {T:header}
        :param np.array mu_angles: A np.array containing all mu_angles for which to compute
                                   the interpolation
        :param bool spec_intensity: Run in Spec intensity mode (Not Implemented anymore)
        :param float mu_local: The local mu_angle (Needed in old implementation of the spec intensity)
        :param list ref_mu: A list of mu angles available for the spec intensity calculation 
        
        
        :returns: wave, mu_dict (dictionary of {mu:specs}), header 
    """
    assert ref_spectra is not None, "Please add a Reference Wavelength using the ref_spectra param"
    assert ref_wave is not None, "Please add a Reference Wavelength using the ref_wave param"
    assert ref_headers is not None, "Please add the Reference headers using the ref_headers param"
    if spec_intensity:
        raise NotImplementedError
        assert ref_mu is not None, "You have to provide mu if spec_intensity is True"

    T_close = int(round(T_local, -2))

    # Get closest spectrum
    wave = ref_wave
    header = ref_headers[T_close]
    mu_dict = {}
    for mu in mu_angles:
        if not spec_intensity:
            spec = ref_spectra[T_close][1.0]
        else:
            # The specific intensities are saved as a datacube
            spec_int_cube = ref_spectra[T_close]
            # Get the closest index of the mu
            idx = np.abs(ref_mu - mu_local).argmin()
            print(f"Closest mu to {mu_local} at idx={idx}")
            spec = ref_spectra[T_close][idx]
        
        assert wave.shape == spec.shape

        # Now interpolate with the contrast given by the Planck curves
        if interpolation_mode == "planck_ratio":
            if int(T_local) != T_close:
                spec = spec * planck_ratio(wave * 1e-10, T_local, T_close)
        elif interpolation_mode == "cubic_spline":
            spec = interpolate_on_temperature(T_local, wave, ref_spectra, logg=logg, feh=feh, mu=1.0)
        else:
            raise NotImplementedError(f"interpolation_mode={interpolation_mode} is not implemented")
        mu_dict[mu] = spec
    return wave, mu_dict, header

    
def radiance_to_temperature(radiance):
    """ Convert an array of radiance to temperature.
    
    Following the formula from:
    https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law#Integration_of_intensity_derivation
    , i.e.: radiance = sigma / pi * T^4
    The factor pi in the end stems from the integration over the hemisphere onto which the energy is emitted
    but decreased by the integral over the cos(z) as the emission is Lambertian
    See: https://en.wikipedia.org/wiki/Radiance  under Description

    :param array_like or float spectral_radiance: Spectral radiance in erg/cm^2/s/sr
    """
    # stefan-boltzmann constant in erg/cm^2/s/K^4
    sigma = 5.670374e-5
    temperature = np.power(radiance / sigma * np.pi, 1 / 4)
    return temperature


def calc_granulation_velocity_rad(granulation_temp_local):
    """ Calculate the radial velocity of the granulation for ony map.

        :param np.array granulation_temp_local: Temperature map in K
    """
    granulation_rad_local = np.zeros_like(granulation_temp_local)
    # determine the velocity values
    # First the radial velocity component
    # Crude way to find the granular lanes and the granules
    granule_mask = granulation_temp_local >= DIVIDING_TEMP
    granular_lane_mask = granulation_temp_local < DIVIDING_TEMP
    v_lane = 4500
    v_granule = -1500
    granulation_rad_local[granular_lane_mask] = v_lane * (
                DIVIDING_TEMP - granulation_temp_local[granular_lane_mask]) / (
                                                        DIVIDING_TEMP - np.min(granulation_temp_local))
    granulation_rad_local[granule_mask] = v_granule * (DIVIDING_TEMP - granulation_temp_local[granule_mask]) / (
            DIVIDING_TEMP - np.max(granulation_temp_local))
    i = 0
    while np.abs(np.mean(granulation_rad_local)) > 0.001:
        i += 1
        print(i, np.abs(np.mean(granulation_rad_local)), v_lane, v_granule)
        if i > 1e6:
            break
        dv = 0.01
        if np.mean(granulation_rad_local) > 0:
            v_lane -= dv
            v_granule -= dv
        else:
            v_lane += dv
            v_granule += dv
        granulation_rad_local[granular_lane_mask] = v_lane * (
                    DIVIDING_TEMP - granulation_temp_local[granular_lane_mask]) / \
                                                    (DIVIDING_TEMP - np.min(granulation_temp_local))
        granulation_rad_local[granule_mask] = v_granule * (DIVIDING_TEMP - granulation_temp_local[granule_mask]) / \
                                              (DIVIDING_TEMP - np.max(granulation_temp_local))
    return granulation_rad_local

def calc_granulation_velocity_phi_theta(granulation_temp_local, vel_rad=None):
    """ Calculate the phi and theta components of the granulation."""
    # Define the areas which are granules
    granule_mask = granulation_temp_local >= DIVIDING_TEMP
    if vel_rad is None:
        vel_rad = calc_granulation_velocity_rad(granulation_temp_local)

    # Determine the size of a cell (will be useful)
    size = granule_mask.shape[0]

    # To fix border effects we want to compute all that several cells and only use the values for the center cell
    # Start out with a 3x3 cell grid
    vel_rad3x3 = np.zeros((size * 3, size * 3))
    granule_mask3x3 = np.zeros((size * 3, size * 3), dtype=bool)
    for i in range(3):
        for j in range(3):
            vel_rad3x3[i * size:(i + 1) * size, j * size:(j + 1) * size] = vel_rad
            granule_mask3x3[i * size:(i + 1) * size, j * size:(j + 1) * size] = granule_mask

    # To reduce computation time we now only take the 2x2 cells centered on the main cell
    # i.e. we cut the outer cells in half or into quarter
    vel_rad2x2 = vel_rad3x3[int(0.5 * size):int(2.5 * size), int(0.5 * size):int(2.5 * size)]
    granule_mask2x2 = granule_mask3x3[int(0.5 * size):int(2.5 * size), int(0.5 * size):int(2.5 * size)]

    # Define an array that will contain the vectors
    vec_field = np.zeros((size * 2, size * 2, 2))
    for row in range(int(0.5 * size), int(1.5 * size)):
        for col in range(int(0.5 * size), int(1.5 * size)):
            if not granule_mask2x2[row, col]:
                # For the moment assume now horizontal velocity within the inter granular lanes
                vec = np.array([0, 0])
            else:
                # if row != size or col != size:
                #     vec = np.array([0, 0])
                # else:
                current_pos = np.array((row, col))
                # Calculate the distance of the complete map from the current pixel
                # HERE we could still optimize, since that can clearly be computed more efficiently
                dist = distance_from_px(granule_mask2x2, current_pos[0], current_pos[1])
                # Set the distance of all stuff in a granule to infinite
                dist[granule_mask2x2] = np.inf
                # Get the closest non infinite distance
                min_dist_coords = np.array(np.unravel_index(dist.argmin(), dist.shape))
                # Compute in vector form and normalize using the radial velocity
                vec = min_dist_coords - current_pos
                normalization = np.abs(vel_rad2x2[row, col]) / np.linalg.norm(vec)
                # The minus sign gets the direction right in ThreeDimStar
                vec = - vec * normalization
            vec_field[row, col] = vec

    vel_phi = vec_field[:, :, 1][int(0.5 * size):int(1.5 * size), int(0.5 * size):int(1.5 * size)]
    vel_theta = vec_field[:, :, 0][int(0.5 * size):int(1.5 * size), int(0.5 * size):int(1.5 * size)]
    vec_field = vec_field[int(0.5 * size):int(1.5 * size),int(0.5 * size):int(1.5 * size), :]
    granule_mask2x2 = granule_mask2x2[int(0.5 * size):int(1.5 * size), int(0.5 * size):int(1.5 * size)]
    return vel_phi, vel_theta, vec_field, granule_mask2x2

def distance_from_px(img, row, col):
    """ Compute the distance from a pixel"""
    coords = np.linspace(0, img.shape[0], img.shape[0], dtype=int)
    cols, rows = np.meshgrid(coords, coords)
    dist = np.sqrt(np.square(rows - row) +
                   np.square(cols - col))
    return dist

def delta_relativistic_doppler(wave, v=None, v_c=None):
    if v_c is None:
        v_c = v / const.C
    
    rel_wave_shift = np.sqrt((1 + v_c) / (1 - v_c))
    wave_shifted = wave * rel_wave_shift
    
    delta_wave = wave_shifted - wave
    
    return delta_wave

if __name__ == "__main__":
    wave, spec, header = phoenix_spectrum(wavelength_range=(5010, 5030))
    
    v = 10000
    wave_pos = wave + delta_relativistic_doppler(wave, v)
    wave_neg = wave + delta_relativistic_doppler(wave, -v)

    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(wave, spec, color="tab:green")
    ax.plot(wave_pos, spec, color="tab:red")
    ax.plot(wave_neg, spec,color="tab:blue")
    plt.savefig("dbug.png")
