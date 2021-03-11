import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import griddata


def get_circular_phi_theta_x_y_z(inclination=90):
    """ Sample x and z for a spherical star."""
    N = 250
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


def pulsation_rad(l=1, m=1, N=1000, project=True, inclination=90):
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

    # plot_3d(x, y, z)

    if project:
        # Add in line of sight
        harmonic = harmonic * np.sin(theta)

    # TODO maybe normalize?
    grid = project_2d_new(x, y, z, harmonic, N)

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


def project_2d_new(x, y, z, values, N=1000, inclination=30):
    """ https://math.stackexchange.com/questions/2305792/3d-projection-on-a-2d-plane-weak-maths-ressources/2306853"""

    y = y
    large_number = 1e20
    # keep origin fixed
    origin = (0, large_number, 0)
    d = 1

    rot = Rot.from_euler('x', 90 - inclination, degrees=True)

    x_rot = np.zeros(x.shape)
    y_rot = np.zeros(y.shape)
    z_rot = np.zeros(z.shape)
    for row in range(x.shape[0]):
        for col in range(x.shape[1]):
            vec = (x[row, col], y[row, col], z[row, col])
            vec_rot = rot.apply(vec)
            x_rot[row, col] = vec_rot[0]
            y_rot[row, col] = vec_rot[1]
            z_rot[row, col] = vec_rot[2]

    x_rot[y_rot < 0] = np.nan
    z_rot[y_rot < 0] = np.nan

    x_proj = (x_rot - origin[0]) * d / (y_rot - origin[1])
    z_proj = (z_rot - origin[2]) * d / (y_rot - origin[1])

    x_proj = x_proj / np.nanmax(x_proj)
    z_proj = z_proj / np.nanmax(z_proj)

    dN = 2 / N
    x_grid = np.arange(-1 - dN, 1 + 2 * dN, dN)
    z_grid = np.arange(-1 - dN, 1 + 2 * dN, dN)
    xx, zz = np.meshgrid(x_grid, z_grid, sparse=False)

    coords = []
    for x_val, z_val in zip(x_proj.flatten(), z_proj.flatten()):
        coords.append((x_val, z_val))
    coords = np.array(coords)
    nan_mask = np.logical_not(np.isnan(x_proj.flatten()))
    coords = coords[nan_mask]
    values = values.flatten()[nan_mask]
    grid = griddata(coords, values.real, (xx, zz),
                    method='linear', fill_value=np.nan)
    return grid


def plot_3d(x, y, z):
    fig = plt.figure(figsize=plt.figaspect(1.))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    # ax.aspect_ratio("equal")
    plt.show()
    exit()


if __name__ == "__main__":
    rad = pulsation_rad(l=3, m=3, N=1000, project=False)
    # rad_incl = create_starmask()
    fig, ax = plt.subplots(1)

    ax.imshow(rad.real, cmap="seismic", origin='lower')
    # rad = rad * np.exp(1j * 2 * np.pi * nu * t)
    # ax[1].imshow(rad_incl.real, cmap="seismic", )

    plt.show()
