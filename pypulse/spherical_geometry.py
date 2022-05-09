import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import time

""" Through this document we follow the ISO convention often used in physics as described here:
    https://en.wikipedia.org/wiki/Spherical_coordinate_system
    
    i.e.: 
    theta: Polar angle or colatitude or inclination, measured from the pole southward : [0,  pi]
    phi: azimuthal angle: [0, 2pi]
"""

def sph_to_x_y_z(phi, theta):
    """ Convert spherical coordinates phi, theta to x,y,z."""
    # The Cartesian coordinates of the unit sphere
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z

def x_y_z_to_sph(x, y, z):
    """ Convert x, y, z to spherical coordinates phi, theta."""
    # The Cartesian coordinates of the unit sphere
    r = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    assert np.abs(1-r) < 1e-10, "You should be on the Unit sphere!"
    theta = np.arccos(z/r)

    if x > 0:
        phi = np.arctan(y/x)
    elif x < 0 and y >= 0:
        phi = np.arctan(y/x) + np.pi
    elif x < 0 and y < 0:
        phi = np.arctan(y/x) - np.pi
    elif x == 0 and y > 0:
        phi = np.pi / 2
    elif x == 0 and y < 0:
        phi = -np.pi / 2
    elif x == 0 and y == 0:
        if z > 0:
            # Map to the pole
            phi = 0
        elif z < 0:
            phi = np.pi

    return phi, theta



def get_spherical_phi_theta_x_y_z(N=250):
    """ Sample x and z for a spherical star."""
    # Theta: Polar Angle, i.e. colatitude or zenith angle
    theta = np.linspace(0, np.pi, N)

    # Phi: Azimuthal angle
    phi = np.linspace(0, 2 * np.pi, 2 * N)
    # phi = np.linspace(0, np.pi)
    phi, theta = np.meshgrid(phi, theta)

    x, y, z = sph_to_x_y_z(phi, theta)

    return phi, theta, x, y, z


def project_line_of_sight(phi, theta, values, component, inclination):
    """ Project the values onto the line of sight."""

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
    # print(np.nanmin(scalar_prods), np.nanmax(scalar_prods))
    values = values * scalar_prods

    return values


def project_2d(x, y, z, phi, theta, values, N,
               border=10, component=None, inclination=90,
               azimuth=0, line_of_sight=False):
    """ Project the 3d geometry onto a 2d plane.

    https://math.stackexchange.com/questions/2305792/3d-projection-on-a-2d-plane-weak-maths-ressources/2306853"""

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
        phi = phi.flatten()[nan_mask]
        theta = theta.flatten()[nan_mask]
        values = project_line_of_sight(
            phi, theta, values, component, inclination)

    # print(values)

    # original
    method = "linear"

    grid = griddata(coords, values, (xx, zz),
                    method=method, fill_value=np.nan)
    return grid
