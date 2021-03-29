import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import time


def sph_to_x_y_z(phi, theta):
    """ Convert spherical coordinates phi, theta to x,y,z."""
    # The Cartesian coordinates of the unit sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def get_spherical_phi_theta_x_y_z(N=250):
    """ Sample x and z for a spherical star."""
    # Theta: Polar Angle
    theta = np.linspace(0, np.pi, N)

    # Phi: Azimuthal angle
    phi = np.linspace(0, 2 * np.pi, 2 * N)
    # phi = np.linspace(0, np.pi)
    phi, theta = np.meshgrid(phi, theta)

    x, y, z = sph_to_x_y_z(phi, theta)

    return phi, theta, x, y, z


def pulsation_rad(l=1, m=1, N=1000, line_of_sight=True, inclination=90, border=10):
    """ Get radial component of pulsation.


        :param int l: Number of surface lines of nodes
        :param int m: Number of polar lines of nodes (-l<=m<=l)
        :param int N: Number of cells to sample
        :param bool project: If True: Project the component onto the line of
                                      sight
    """
    phi, theta, x, y, z = get_spherical_phi_theta_x_y_z()
    # Calculate the spherical harmonic Y(l,m)
    displ = sph_harm(m, l, phi, theta)

    # TODO maybe normalize?
    # displ = displ / np.nanmax(displ)

    # plot_3d(x, y, z)

    grid = project_2d(x, y, z, phi, theta, displ, N, component="rad",
                      inclination=inclination,
                      line_of_sight=line_of_sight,
                      border=border)

    return grid


def pulsation_phi(l=1, m=1, N=1000, line_of_sight=True, inclination=90, border=10):
    """ Get phi component of pulsation.


        :param int l: Number of surface lines of nodes
        :param int m: Number of polar lines of nodes (-l<=m<=l)
        :param int N: Number of cells to sample
        :param bool project: If True: Project the component onto the line of
                                      sight
    """
    phi, theta, x, y, z = get_spherical_phi_theta_x_y_z()
    # Calculate the spherical harmonic Y(l,m)
    harmonic = sph_harm(m, l, phi, theta)
    # You need the partial derivative wrt to phi
    displ = 1 / np.sin(theta) * 1j * m * harmonic

    # TODO maybe normalize?
    # displ = displ / np.nanmax(displ)

    # plot_3d(x, y, z)

    grid = project_2d(x, y, z, phi, theta, displ, N, component="phi",
                      inclination=inclination,
                      line_of_sight=line_of_sight,
                      border=border)

    return grid


def pulsation_theta(l=1, m=1, N=1000, line_of_sight=True, inclination=90, border=10):
    """ Get theta component of pulsation.


        :param int l: Number of surface lines of nodes
        :param int m: Number of polar lines of nodes (-l<=m<=l)
        :param int N: Number of cells to sample
        :param bool project: If True: Project the component onto the line of
                                      sight
    """
    phi, theta, x, y, z = get_spherical_phi_theta_x_y_z()
    # Calculate the spherical harmonic Y(l,m)
    harmonic = sph_harm(m, l, phi, theta)
    # You need the partial derivative wrt to theta
    # Taken from
    # https://functions.wolfram.com/Polynomials/SphericalHarmonicY/20/ShowAll.html
    if m < l:
        part_deriv = m * 1 / np.tan(theta) * harmonic + \
            np.sqrt((l - m) * (l + m + 1)) * np.exp(-1j * phi) * \
            sph_harm(m + 1, l, phi, theta)
    else:
        part_deriv = m * 1 / np.tan(theta) * harmonic

    displ = part_deriv

    # TODO maybe normalize?
    # displ = displ / np.nanmax(displ)

    # plot_3d(x, y, z)

    grid = project_2d(x, y, z, phi, theta, displ, N, component="theta",
                      inclination=inclination,
                      line_of_sight=line_of_sight,
                      border=border)

    return grid


def project_2d(x, y, z, phi, theta, values, N, border=10, component=None, inclination=90, azimuth=0, line_of_sight=False):
    """ https://math.stackexchange.com/questions/2305792/3d-projection-on-a-2d-plane-weak-maths-ressources/2306853"""

    y = y
    large_number = 1e20
    # keep origin fixed
    origin = (0, large_number, 0)
    d = 1

    rot = Rot.from_euler('xz', [90 - inclination, azimuth], degrees=True)

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
    x_grid = np.linspace(-1 - border * dN, 1 + border * dN, N + 2 * border)
    z_grid = np.linspace(-1 - border * dN, 1 + border * dN, N + 2 * border)
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
            elif component == "theta":
                theta_unit = np.array((np.cos(t) * np.cos(p),
                                       np.cos(t) * np.sin(p),
                                       -np.sin(t)))
                scalar_prods.append(np.dot(theta_unit, los))

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


def calculate_pulsation(l, m, V_p, k, nu, t, inclination=90, N=100, border=10):
    """ Calculate the total pulsation.

        Calculates all values with line of sight projection.


        V_p: Amplitude of pulsation in m/s
        k: ratio of phi and theta amplitude wrt to radial amplitude
           1.2 for g-mode (compare to Hatzes1996)
        nu: frequency of pulsation (1/P) in 1/d
        t: time in days

        :returns: Total velocity grid
    """
    rad = pulsation_rad(l=l, m=m, N=N, line_of_sight=True,
                        inclination=inclination, border=border)
    phi = pulsation_phi(l=l, m=m, N=N, line_of_sight=True,
                        inclination=inclination, border=border)
    theta = pulsation_theta(
        l=l, m=m, N=N, line_of_sight=True, inclination=inclination, border=border)
    # Add a factor of 1j. as the pulsations are yet the radial displacements
    # you need to differentiate the displacements wrt t which introduces
    # a factor 1j * 2 * np.pi * nu
    # but we absorb the  2 * np.pi * nu part in the V_p constant
    # See Kochukhov et al. (2004)
    rad = 1j * V_p * rad * np.exp(1j * 2 * np.pi * nu * t)
    phi = 1j * k * V_p * phi * np.exp(1j * 2 * np.pi * nu * t)
    theta = 1j * k * V_p * theta * np.exp(1j * 2 * np.pi * nu * t)

    pulsation = rad + phi + theta

    return pulsation, rad, phi, theta


def calc_temp_variation(l, m, amplitude, nu, t, phase_shift=0, inclination=90, N=100, border=10):
    """ Calculate the temperature variation."""
    rad = pulsation_rad(l=l, m=m, N=N, line_of_sight=False,
                        inclination=inclination, border=border)

    rad = rad * np.exp(1j * 2 * np.pi * nu * t)
    var = (rad * np.exp(1j * phase_shift)).real
    # Normalize
    var = var / np.nanmax(var)
    var = var * amplitude
    return var, rad


if __name__ == "__main__":
    star_mask = create_starmask()
    temp = star_mask * 4800
    spot_mask = create_spotmask_new(15, theta_pos=45, phi_pos=90)
    temp[spot_mask] = 3000
    plt.imshow(temp)
    plt.show()
