import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import griddata


def get_circular_phi_theta_x_y_z():
    """ Sample x and z for a spherical star."""
    N = 250
    # Theta: Polar Angle
    theta = np.linspace(0, np.pi, N)

    # Phi: Azimuthal angle
    phi = np.linspace(0, 2 * np.pi, 2 * N)
    phi, theta = np.meshgrid(phi, theta)

    # The Cartesian coordinates of the unit sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return phi, theta, x, y, z


def create_starmask(N=1000):
    """ Return a starmask."""
    _, _, x, y, z = get_circular_phi_theta_x_y_z()

    star = np.ones(x.shape)
    star_mask = project_2d(x, y, z, star, N)
    star_mask[np.isnan(star_mask)] = 0
    return star_mask


def pulsation_rad(l=1, m=1, N=1000, line_of_sight=True, inclination=90):
    """ Get radial component of pulsation.


        :param int l: Number of surface lines of nodes
        :param int m: Number of polar lines of nodes (-l<=m<=l)
        :param int N: Number of cells to sample
        :param bool project: If True: Project the component onto the line of
                                      sight
    """
    phi, theta, x, y, z = get_circular_phi_theta_x_y_z()
    # Calculate the spherical harmonic Y(l,m)
    harmonic = sph_harm(m, l, phi, theta)

    # TODO maybe normalize?
    harmonic = harmonic / np.nanmax(harmonic)

    # plot_3d(x, y, z)

    grid = project_2d(x, y, z, phi, theta, harmonic, N, component="rad",
                      inclination=inclination,
                      line_of_sight=line_of_sight)

    return grid


def pulsation_phi(l=1, m=1, N=1000, line_of_sight=True, inclination=90):
    """ Get radial component of pulsation.


        :param int l: Number of surface lines of nodes
        :param int m: Number of polar lines of nodes (-l<=m<=l)
        :param int N: Number of cells to sample
        :param bool project: If True: Project the component onto the line of
                                      sight
    """
    phi, theta, x, y, z = get_circular_phi_theta_x_y_z()
    # Calculate the spherical harmonic Y(l,m)
    harmonic = sph_harm(m, l, phi, theta)
    # You need the partial derivative wrt to phi
    harmonic = 1 / np.sin(theta) * 1j * m * harmonic

    # TODO maybe normalize?
    harmonic = harmonic / np.nanmax(harmonic)

    # plot_3d(x, y, z)

    grid = project_2d(x, y, z, phi, theta, harmonic, N, component="phi",
                      inclination=inclination,
                      line_of_sight=line_of_sight)

    return grid


def project_2d(x, y, z, phi, theta, values, N, component="rad", inclination=90, line_of_sight=True):
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
    border = 10
    x_grid = np.arange(-1 - border * dN, 1 + (border + 1) * dN, dN)
    z_grid = np.arange(-1 - border * dN, 1 + (border + 1) * dN, dN)
    xx, zz = np.meshgrid(x_grid, z_grid, sparse=False)

    coords = []
    for x_val, z_val in zip(x_proj.flatten(), z_proj.flatten()):
        coords.append((x_val, z_val))
    coords = np.array(coords)
    nan_mask = np.logical_not(np.isnan(x_proj.flatten()))
    coords = coords[nan_mask]
    x_proj = x_proj.flatten()[nan_mask]
    z_proj = z_proj.flatten()[nan_mask]
    values = values.flatten()[nan_mask]

    if line_of_sight:
        theta = theta.flatten()[nan_mask]
        phi = phi.flatten()[nan_mask]

        # Line of sight unit vector
        los = np.array(
            (0,
             np.cos(np.radians(90 - inclination)),
             -np.sin(np.radians(90 - inclination))))

        scalar_prods = []
        for p, t in zip(phi, theta):
            if component == "rad":
                # Unit vector of r
                r_unit = np.array((np.sin(t) * np.cos(p),
                                   np.sin(t) * np.sin(p),
                                   np.cos(t)))

                scalar_prods.append(np.dot(r_unit, los))
            elif component == "phi":
                phi_unit = np.array((-np.sin(p),
                                     np.cos(p),
                                     0))
                scalar_prods.append(np.dot(phi_unit, los))

        scalar_prods = np.array(scalar_prods)
        print(np.nanmin(scalar_prods), np.nanmax(scalar_prods))
        values = values * scalar_prods

    grid = griddata(coords, values, (xx, zz),
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
    l = 5
    m = 5
    rad_proj = pulsation_rad(
        l=l, m=m, N=100, line_of_sight=True, inclination=60)
    phi_proj = pulsation_phi(
        l=l, m=m, N=100, line_of_sight=True, inclination=60)

    pulse = rad_proj + phi_proj
    # rad_incl = pulsation_rad(
    # l=5, m=5, N=1000, line_of_sight=True, inclination=60)
    # rad = create_starmask()
    fig, ax = plt.subplots(1, 3)

    ax[0].imshow(rad_proj.real, cmap="seismic",
                 origin='lower', vmin=-1, vmax=1)
    nu = 1
    t = 0.25
    # phi = phi * np.exp(1j * 2 * np.pi * nu * t)
    ax[1].imshow(phi_proj.real, cmap="seismic",
                 origin="lower", vmin=-1, vmax=1)

    ax[2].imshow(pulse.real, cmap="seismic",
                 origin="lower", vmin=-1, vmax=1)

    plt.show()
