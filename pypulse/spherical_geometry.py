import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D


def get_circular_phi_theta_x_y_z(inclination=90):
    """ Sample x and z for a circular star."""
    N = 100
    phi = np.linspace(0, np.pi, N)
    factor = 2
    theta = np.linspace(0, factor * np.pi, factor * N)
    phi, theta = np.meshgrid(phi, theta)

    # The Cartesian coordinates of the unit sphere
    # phi_incl = np.mod(phi + np.radians(inclination - 90), np.pi)
    # print(phi_incl)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return phi, theta, x, y, z


def create_starmask(N=100):
    """ Return a starmask."""
    _, _, x, y, z = get_circular_phi_theta_x_y_z()

    star = np.ones(x.shape)
    star_mask = project_2d(x, y, z, star)
    star_mask[np.isnan(star_mask)] = 0
    return star_mask


def pulsation_rad(l=1, m=1, N=100, project=True, inclination=90):
    """ Get radial component of pulsation.


        :param int l: Number of surface lines of nodes
        :param int m: Number of polar lines of nodes (-l<=m<=l)
        :param int N: Number of cells to sample
        :param bool project: If True: Project the component onto the line of
                                      sight
    """
    phi, theta, x, y, z = get_circular_phi_theta_x_y_z(inclination)
    # Calculate the spherical harmonic Y(l,m)
    harmonic = sph_harm(m, l, theta, phi)

    if project:
        # Add in line of sight
        harmonic = harmonic * np.sin(theta)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z)
    # # ax.aspect_ratio("equal")
    # plt.show()
    # exit()

    # TODO maybe normalize?
    grid = project_2d(x, y, z, harmonic, N)

    return grid


def project_2d(x, y, z, values, N=100):
    """ Project the 2d array value on the unitsphere.

        At the moment no inclination
    """
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
            value = values[mask]
            if not len(value):
                value = np.nan
            else:
                value = np.mean(value)
            grid[col, row] = value

    return grid


if __name__ == "__main__":
    rad = pulsation_rad(l=3, m=3, N=100, project=False)
    rad_incl = create_starmask()
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(rad.real, cmap="seismic", )
    # rad = rad * np.exp(1j * 2 * np.pi * nu * t)
    ax[1].imshow(rad_incl.real, cmap="seismic", )

    plt.show()
