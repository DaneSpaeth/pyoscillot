import numpy as np
import matplotlib.pyplot as plt
import dataloader as load


def create_circular_mask(h, w, center=None, radius=None):
    """ Create a circular mask.

        From https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array
    """

    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask


def gaussian(x, mu=0, sigma=0.001):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-np.square(x - mu) / (2 * sigma**2))


def stellar_line():
    pass


class GridStar():
    """ Simulate a star with a grid."""

    def __init__(self, N=500, Teff=4500, vsini=1000, add_spot=False):
        """ Initialize grid."""
        if not N % 2:
            N += 1
        self.N = N
        self.Teff = Teff
        self.grid = np.zeros((N, N))
        self.rotation = self.grid
        self.pulsation = self.grid
        self.center = (int(N / 2), int(N / 2))

        self.star = create_circular_mask(N, N)
        self.add_rotation(vsini=vsini)
        self.temperature = Teff * self.star
        if add_spot:
            print("Add spot")
            self.add_star_spot(400, 250, radius=200, deltaT=1500)
        self.sim_lines()

    def add_rotation(self, vsini):
        """ Add Rotation, vsini in m/s"""
        pos_map = self.star
        rel_hor_dist = self.grid
        one_line = np.arange(0, self.N, dtype=int) - self.center[0]
        print(one_line)

        for i in range(rel_hor_dist.shape[0]):
            rel_hor_dist[i, :] = one_line

        self.rotation = (rel_hor_dist * self.star) / \
            np.max(rel_hor_dist) * vsini

        print(np.max(self.rotation))

    def sim_lines(self):
        num_lin = 1000
        wave_center = 700

        self.doppler_shift = self.rotation / 3e8 * wave_center

        # For the moment 1d approximation
        # doppler_shift = self.doppler_shift[self.center[1], :, ]

        doppler_shift = self.doppler_shift[self.star]
        min_wave = 6432.45
        max_wave = 6432.8
        wavelength_range = (min_wave - 1, max_wave + 1)
        rest_spectrum, wavelength, _ = load.load_spectrum(
            Teff=self.Teff, wavelength_range=wavelength_range)
        # TODO make spot temp adjustable
        spot_spectrum, _, _ = load.load_spectrum(
            Teff=3000, wavelength_range=wavelength_range)
        self.spectrum = np.zeros(len(wavelength))

        for row in range(self.N):
            for col in range(self.N):
                if self.star[row, col]:
                    if self.temperature[row, col] == 3000:
                        local_spectrum = spot_spectrum
                    else:
                        local_spectrum = rest_spectrum
                    print(self.doppler_shift[row, col])
                    shift_spectrum = add_doppler_shift(
                        local_spectrum, wavelength, self.doppler_shift[row, col])
                    self.spectrum = self.spectrum + shift_spectrum
                    # print(f"Finished Row {row}, Column {col}")

        self.spectrum = (self.spectrum / np.max(self.spectrum))
        self.rest_spectrum = rest_spectrum / np.max(rest_spectrum)

        self.spectrum, self.wavelength = cut_to_maxshift(self.spectrum, wavelength,
                                                         min_wave, max_wave)
        self.rest_spectrum, _ = cut_to_maxshift(self.rest_spectrum, wavelength,
                                                min_wave, max_wave)
        # exit()

        # lines = np.zeros((self.N, self.N, num_lin))
        # for i in range(num_lin):
        #     lines[:, :, i] = line[i]

        # self.spectrum = np.sum(lines, axis=(0, 1))

    def add_star_spot(self, x, y, radius=50, deltaT=1000):
        """ Add a circular starspot at position x,y."""
        self.star_spot_mask = create_circular_mask(
            self.N, self.N, center=(x, y), radius=radius)
        self.temperature[self.star_spot_mask] -= deltaT


def bisector(wavelength, spectrum):
    bisector_waves = []
    bisector_flux = []
    search_for = np.linspace(np.min(spectrum), 0.8, 25)
    max_idx = np.argmin(spectrum)
    left_spectrum = spectrum[0:max_idx]
    print(wavelength[max_idx])
    left_wavelength = wavelength[0:max_idx]
    right_spectrum = spectrum[max_idx:]
    right_wavelength = wavelength[max_idx:]
    for s in search_for:
        diff_left = np.abs(left_spectrum - s)
        diff_right = np.abs(right_spectrum - s)
        left_idx = np.argmin(diff_left)
        right_idx = np.argmin(diff_right)
        left_wave = left_wavelength[np.argmin(diff_left)]
        right_wave = right_wavelength[np.argmin(diff_right)]

        bisector_wave = (right_wave + left_wave) / 2
        bisector_flux.append(s)
        bisector_waves.append(bisector_wave)

    return np.array(bisector_waves), np.array(bisector_flux)


def add_doppler_shift(rest_spectrum, rest_wavelength, doppler_shift):
    """ Return shifted spectrum

        Wavelengths and shift in Angstrom.

        Assume that the spectrum outside the shifted area must not be correct.
    """
    if doppler_shift == 0:
        return rest_spectrum
    elif doppler_shift > 0:
        shift_idx = np.argmin(
            np.abs(rest_wavelength - (rest_wavelength[0] + doppler_shift)))
        if not shift_idx:
            return rest_spectrum
        shift_spectrum = np.zeros(len(rest_spectrum) + shift_idx)
        shift_spectrum[shift_idx:] = rest_spectrum
        shift_spectrum = shift_spectrum[:-shift_idx]
        return shift_spectrum
    elif doppler_shift < 0:
        shift_idx = np.argmin(
            np.abs(rest_wavelength - (rest_wavelength[-1] + doppler_shift)))
        if shift_idx == len(rest_spectrum) - 1:
            return rest_spectrum
        shift_spectrum = np.zeros(
            len(rest_spectrum) + len(rest_spectrum) - shift_idx)
        shift_spectrum[:shift_idx] = rest_spectrum[len(
            rest_spectrum) - shift_idx:]
        shift_spectrum = shift_spectrum[:len(rest_spectrum)]
        return shift_spectrum


def cut_to_maxshift(spectrum, wavelength, min_wave, max_wave):
    """ Cut the spectrum such that the maximal shift is masked."""
    min_idx = np.argmin(np.abs(wavelength - min_wave))
    max_idx = np.argmin(np.abs(wavelength - max_wave))
    spectrum = spectrum[min_idx:max_idx]
    wavelength = wavelength[min_idx:max_idx]

    return spectrum, wavelength


if __name__ == "__main__":

    # max_shift = 0.0000001

    # wavelength_range = (6432 - np.abs(max_shift), 6434 + np.abs(max_shift))
    # rest_spectrum, rest_wavelength, _ = load.load_spectrum(
    #     Teff=4500, wavelength_range=wavelength_range)

    # shift_spectrum = add_doppler_shift(
    #     rest_spectrum, rest_wavelength, max_shift)

    # plt.plot(rest_wavelength, rest_spectrum)
    # plt.plot(rest_wavelength, shift_spectrum)
    # plt.show()
    # exit()

    s = GridStar(vsini=30000)
    spotstar = GridStar(vsini=0, add_spot=False)
    # s2 = GridStar(vsini=10000)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(s.wavelength, s.spectrum, label="Rotation")
    ax[0].plot(spotstar.wavelength, spotstar.spectrum, label="No Rotation")
    ax[0].legend()
    # plt.plot(s2.wavelength, s2.spectrum, color="Red")
    bisector_waves, bisector_flux = bisector(s.wavelength, s.spectrum)
    spot_bisector_waves, spot_bisector_flux = bisector(
        spotstar.wavelength, spotstar.spectrum)
    # ax[0].scatter(bisector_waves, bisector_flux)
    ax[1].plot(bisector_waves, bisector_flux)
    ax[1].plot(spot_bisector_waves, spot_bisector_flux)
    # plt.imshow(s.spectrum)
    # plt.imshow(s.star_spot_mask)
    plt.show()
