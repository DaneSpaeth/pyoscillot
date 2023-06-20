import numpy as np
import plapy.constants as const
from dataloader import phoenix_spectrum, phoenix_spec_intensity
import matplotlib.pyplot as plt
from plapy.constants import C


DIVIDING_TEMP = 5100

def energy_flux_to_photon_flux(wavelength, spectrum):
    """ Convert a sepctrum giben in energy flux (such as a PHOENIX spectrum) to a photon flux"""
    energy_photon = const.H * const.C / wavelength
    spectrum_photon = spectrum / energy_photon
    return spectrum_photon

def planck(wav, T):
    """ Return planck's law at wavelength and Temperature.

        :param float/np.array wav: Wavelength in m
        :param T: Temperature in K

        :returns intensity: in J/s sr-1 m⁻2 m⁻1
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


# def get_interpolated_spectrum(T_local,
#                               ref_wave,
#                               ref_spectra,
#                               ref_headers=None):
#     """ Return a potentially interpolated spectrum. Returns the same format as
#         the phoenix spectrum.
#     """

#     T_low = int(np.floor(T_local / 100) * 100)
#     T_high = int(np.ceil(T_local / 100) * 100)

#     if T_low == T_high:
#         # print("No Temperature Interpolation")
#         spec = ref_spectra[T_low]
#         header = ref_headers[T_low]
#         wave = ref_wave
#         return wave, spec, header
#     else:

#         # print(f"Use the given Reference Spectra at T={T_close}")
#         assert ref_wave is not None, "Please add a Reference Wavelength using the ref_wave param"
#         assert ref_headers is not None, "Please add the Reference headers using the ref_headers param"
#         wave = ref_wave
#         spec_low = ref_spectra[T_low]
#         spec_high = ref_spectra[T_high]
#         header = ref_headers[T_low]

#         # Now interpolate with the contrast given by the Planck curves

#         ratio_high = 1 - (np.abs(T_high - T_local)) / 100
#         ratio_low = 1 - (np.abs(T_low - T_local)) / 100
#         spec_low_interpol = spec_low * \
#             planck_ratio(wave * 1e-10, T_local, T_low)
#         spec_high_interpol = spec_high * \
#             planck_ratio(wave * 1e-10, T_local, T_low)
#         spec = (spec_low_interpol * ratio_low +
#                 spec_high_interpol * ratio_high)

#     return wave, spec, header

def get_interpolated_spectrum(T_local,
                              ref_wave=None,
                              ref_spectra=None,
                              ref_headers=None,
                              spec_intensity=False,
                              mu_local=1,
                              ref_mu=None):
    """ Return a potentially interpolated spectrum. Returns the same format as
        the phoenix spectrum.

        At the moment:
        logg=3.0, feh=0.0
    """
    assert ref_spectra is not None, "Please add a Reference Wavelength using the ref_spectra param"
    assert ref_wave is not None, "Please add a Reference Wavelength using the ref_wave param"
    assert ref_headers is not None, "Please add the Reference headers using the ref_headers param"
    if spec_intensity:
        assert ref_mu is not None, "You have to provide mu if spec_intensity is True"

    T_close = int(round(T_local, -2))

    # Get closest spectrum
    wave = ref_wave
    header = ref_headers[T_close]
    if not spec_intensity:
        spec = ref_spectra[T_close]
    else:
        # The specific intensities are saved as a datacube
        spec_int_cube = ref_spectra[T_close]
        # Get the closest index of the mu
        idx = np.abs(ref_mu - mu_local).argmin()
        print(f"Closest mu to {mu_local} at idx={idx}")
        spec = ref_spectra[T_close][idx]

    assert wave.shape == spec.shape

    # Now interpolate with the contrast given by the Planck curves
    if int(T_local) != T_close:
        spec = spec * planck_ratio(wave * 1e-10, T_local, T_close)
    return wave, spec, header


def get_ref_spectra(T_grid, logg, feh, wavelength_range=(3000, 7000),
                    spec_intensity=False, fit_and_remove_bis=False):
    """ Return a wavelength grid and a dict of phoenix spectra and a dict of
        pheonix headers.

        The dict has the form {Temp1:spectrum1, Temp2:spectrum2}

        This dict and wavelength grid can be used to interpolate the spectra
        for local T in a later step. Loading them beforhand as a dictionary
        reduces the amount of disk reading at later stages (i.e. if you
        read everytime you want to compute a local T spectrum)

        :param np.array T_grid: Temperature grid in K. The function
                                automatically determines the necessary ref spec
        :param float logg: log(g) for PHOENIX spectrum
        :param float feh: [Fe/H] for PHOENIX spectrum
        :param tuple wavelength_range: Wavelength range fro spectrum in A

        :param bool spec_intensity: If True it will return the spec intensity
                                    spectra cubes and the mu angle as the final
                                    parameter

        :return: tuple of (wavelength grid, T:spec dict, T:header dict, [mu])

    """
    T_grid = T_grid[~np.isnan(T_grid)]
    T_grid = T_grid[T_grid > 0]
    T_grid = np.round(T_grid, -2)
    T_unique = np.unique(T_grid)
    T_unique = T_unique.astype(int)

    # Append the next lowest and next highest values as well
    T_unique = np.insert(T_unique, 0, T_unique[0] - 100)
    T_unique = np.append(T_unique, T_unique[-1] + 100)

    # And now define a grid from the lowest to the highest value with all full 100s

    T_unique = np.linspace(np.min(T_unique), np.max(T_unique), int(
        (np.max(T_unique) - np.min(T_unique)) / 100) + 1, dtype=int)

    ref_spectra = {}
    ref_headers = {}
    if not spec_intensity:
        for T in T_unique:
            wave, ref_spectra[T], ref_headers[T] = phoenix_spectrum(
                Teff=float(T), logg=logg, feh=feh,
                wavelength_range=wavelength_range)
            # All waves are the same, so just return the last one
            if fit_and_remove_bis:
                raise NotImplementedError
                

        return wave, ref_spectra, ref_headers
    else:
        for T in T_unique:
            wave, ref_spectra[T], mu, ref_headers[T] = phoenix_spec_intensity(
                Teff=float(T), logg=logg, feh=feh,
                wavelength_range=wavelength_range)
            # All waves are the same, also all mu should be the same
            # So just return the last one
        return wave, ref_spectra, ref_headers, mu
    
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
        v_c = v / C
    
    wave_shifted = wave * np.sqrt((1 + v_c) / (1-v_c))
    
    delta_wave = wave_shifted - wave
    
    return delta_wave

if __name__ == "__main__":
    wave, spec, header = phoenix_spectrum()

    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(wave, spec)
    ax.plot(wave, planck(wave*1e-10, 4800)*4*np.pi, color="red")
    plt.show()
