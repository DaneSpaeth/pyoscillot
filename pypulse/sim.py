import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, N=500, Teff=5000, vsini=1000):
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
        self.add_star_spot(400, 250, radius=100, deltaT=5000)
        self.sim_lines()

    def add_rotation(self, vsini):
        """ Add Rotation, vsini in m/s"""
        pos_map = self.star
        rel_hor_dist = self.grid
        one_line = np.arange(0, self.N, dtype=int) - self.center[0]

        for i in range(rel_hor_dist.shape[0]):
            rel_hor_dist[i, :] = one_line

        self.rotation = (rel_hor_dist * self.star) / \
            np.max(rel_hor_dist) * vsini

    def sim_lines(self):
        num_lin = 1000
        wave_center = 700
        self.wavelength = np.linspace(
            wave_center - 0.01, wave_center + 0.01, num_lin)

        self.doppler_shift = self.rotation / 3e8 * wave_center

        # For the moment 1d approximation
        # doppler_shift = self.doppler_shift[self.center[1], :, ]

        # doppler_shift = self.doppler_shift[self.star]
        self.spectrum = np.zeros(num_lin)
        for row in range(self.N):
            for col in range(self.N):
                if self.star[row, col]:
                    contrast = self.temperature[row, col] / self.Teff

                    self.spectrum += contrast * gaussian(self.wavelength,
                                                         wave_center + self.doppler_shift[row, col])

        self.spectrum = - (self.spectrum / np.max(self.spectrum))

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
    search_for = np.linspace(-0.01, -1, 1000)
    max_idx = np.argmin(spectrum)
    left_spectrum = spectrum[0:max_idx]
    left_wavelength = wavelength[0:max_idx]
    right_spectrum = spectrum[max_idx:]
    right_wavelength = wavelength[max_idx:]
    for s in search_for:
        print(s)
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


if __name__ == "__main__":
    s = GridStar(vsini=3000)
    # s2 = GridStar(vsini=10000)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(s.wavelength, s.spectrum)
    # plt.plot(s2.wavelength, s2.spectrum, color="Red")
    bisector_waves, bisector_flux = bisector(s.wavelength, s.spectrum)
    ax[0].scatter(bisector_waves, bisector_flux)
    ax[1].scatter(bisector_waves, bisector_flux)
    # plt.imshow(s.spectrum)
    # plt.imshow(s.star_spot_mask)
    plt.show()
