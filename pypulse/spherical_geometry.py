import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm


def get_circular_phi_theta_x_z():
    """ Sample x and z for a circular star."""
    phi = np.linspace(0, np.pi, 250)
    theta = np.linspace(0, np.pi, 250)
    phi, theta = np.meshgrid(phi, theta)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(phi) * np.cos(theta)
    z = np.cos(phi)

    return phi, theta, x, z


def create_starmask(N=100):
    """ Return a starmask."""
    _, _, x, z = get_circular_phi_theta_x_z()

    # Remember that both x and z span from -1 to 1
    # So factor two comes in and also the -1 factor in x_diff and z_diff
    dN = 2 / N
    grid = np.zeros((N, N))
    for row in range(N):
        for col in range(N):
            x_grid = dN * row - 1
            z_grid = dN * col - 1
            x_diff = np.abs(x - x_grid)
            z_diff = np.abs(z - z_grid)
            x_mask = np.where(x_diff < dN, 1, 0)
            z_mask = np.where(z_diff < dN, 1, 0)
            mask = np.logical_and(x_mask, z_mask)
            x_value = x[mask]
            z_value = z[mask]
            if not np.sum(mask):
                star = 0
            else:
                star = 1
            grid[col, row] = star

    return grid


def get_spherical_harmonics(l=1, m=1, N=100):
    phi, theta, x, z = get_circular_phi_theta_x_z()
    # Calculate the spherical harmonic Y(l,m)
    harmonic = sph_harm(m, l, theta, phi).real

    # TODO maybe normalize?

    # Remember that both x and z span from -1 to 1
    # So factor two comes in and also the -1 factor in x_diff and z_diff
    dN = 2 / N
    grid = np.zeros((N, N))
    for row in range(N):
        for col in range(N):
            x_grid = dN * row - 1
            z_grid = dN * col - 1
            x_diff = np.abs(x - x_grid)
            z_diff = np.abs(z - z_grid)
            x_mask = np.where(x_diff < dN, 1, 0)
            z_mask = np.where(z_diff < dN, 1, 0)
            mask = np.logical_and(x_mask, z_mask)
            x_value = x[mask]
            z_value = z[mask]
            sph_value = harmonic[mask]
            if not len(sph_value):
                sph_value = np.nan
            else:
                sph_value = np.mean(sph_value)
            grid[col, row] = sph_value

    return grid


if __name__ == "__main__":
    harm = get_spherical_harmonics()
    plt.imshow(harm)
    plt.show()
