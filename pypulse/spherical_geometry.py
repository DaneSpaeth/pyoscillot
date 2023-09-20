import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as Rot
from scipy.interpolate import griddata, bisplrep, bisplev, interp2d
from scipy.spatial.distance import cdist
import time
from numba import jit

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
        phi = np.arctan2(y, x)
    elif x < 0 and y >= 0:
        phi = np.arctan(y/ x) + np.pi
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

    phi = np.mod(phi + np.pi, 2*np.pi)

    return phi, theta



def get_spherical_phi_theta_x_y_z(N=250):
    """ Sample x and z for a spherical star."""
    # Theta: Polar Angle, i.e. colatitude or zenith angle
    theta = np.linspace(0, np.pi, N)

    # Phi: Azimuthal angle
    # previously I had 2 N here
    phi = np.linspace(0, 2 * np.pi, N)
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
        # print(p, t)
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

# @jit(nopython=True)
def project_2d(x, y, z, phi, theta, values, N,
               border=10, component=None, inclination=90,
               azimuth=0, line_of_sight=False, return_grid_points=False,
               edge_extapolation="nearest"):
    """ Project the 3d geometry onto a 2d plane.

    https://math.stackexchange.com/questions/2305792/3d-projection-on-a-2d-plane-weak-maths-ressources/2306853
    """

    y = y
    large_number = -1e20
    # keep origin fixed
    origin = (0, large_number, 0)
    d = 1
    
    # First rotate the x y and z values to put the observer into the origin
    # i.e. account for inclination (and azimuth, although the latter is not used)
    # Rotate by 90 - inclination in the around the x axis
    # And yb azimuth around the z axis
    # The y axis is defined as the line of sight
    rot = Rot.from_euler('xz', [-(90 - inclination), azimuth], degrees=True)
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
    
    # Hide the layers on the backside of the star
    x_rot[y_rot < 0] = np.nan
    z_rot[y_rot < 0] = np.nan
    
    # fig = plt.figure(figsize=(9,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(x_rot, z_rot, c=values, vmin=4400, vmax=4600)
    # ax.set_xlabel("X'")
    # ax.set_ylabel("Z'")
    # ax.set_aspect('equal', 'box')
    # plt.savefig("rotation.png", dpi=300)
    # plt.close()
    
    # Now project the x and z values onto the line of sight
    x_proj = (x_rot - origin[0]) * d / (y_rot - origin[1])
    z_proj = (z_rot - origin[2]) * d / (y_rot - origin[1])
    
    
    # Since you moved so far back for the projection the values are now very small
    # So we divide by the maximum in x and z (since that is the limb of the star)
    # and should be normalized to one
    # But does it do anything now?
    x_proj = x_proj / np.nanmax(x_proj)
    z_proj = z_proj / np.nanmax(z_proj)
    
    # fig = plt.figure(figsize=(9,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(x_proj, z_proj, c=values, vmin=4400, vmax=4600)
    # ax.set_xlabel("X'")
    # ax.set_ylabel("Z'")
    # ax.set_aspect('equal', 'box')
    # plt.savefig("projection.png", dpi=300)
    # plt.close()

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
    
    # fig = plt.figure(figsize=(9,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(x_proj, z_proj, c=values, vmin=0, vmax=np.pi)
    # ax.set_xlabel("X'")
    # ax.set_ylabel("Y'")
    # ax.set_aspect('equal', 'box')
    # plt.savefig("projection.png", dpi=300)
    # plt.close()
    

    if line_of_sight:
        phi = phi.flatten()[nan_mask]
        theta = theta.flatten()[nan_mask]
        values = project_line_of_sight(
            phi, theta, values, component, inclination)

    # print(values)

    # original
    
    # linearly interpolate all the irregular grid points onto the regular grid
    method = "linear"
    grid = griddata(coords, values, (xx, zz),
                    method=method, fill_value=np.nan)
    nanmask_grid = np.isnan(grid)
    
    # Create the edge_mask
    dx = xx[0,1] - xx[0,0]
    distances_from_origin = np.sqrt(xx**2 + zz**2)
    inside_mask = distances_from_origin <= 1 - np.sqrt(2*(dx/2)**2)
    outside_mask = distances_from_origin > 1 + np.sqrt(2*(dx/2)**2)
    edge_mask = np.logical_and(~inside_mask, ~outside_mask)
    # Also take out the values which are not Nan from the previous interpolation
    edge_mask = np.logical_and(edge_mask, nanmask_grid)
    
    # Test to also take the cells, whose center is outside the circle
    # Using nearest neighbor interpolation -> not ideal
    if edge_extapolation == "nearest":
        grid_nearest = griddata(coords, values, (xx, zz),
                                method="nearest", fill_value=np.nan)
        grid[edge_mask] = grid_nearest[edge_mask]
    elif edge_extapolation == "bispline":
        tck = bisplrep(x_proj, z_proj, values, kx=3, ky=3)
        values_bispl = bisplev(x_grid, z_grid, tck).T 
        grid[edge_mask] = values_bispl[edge_mask]
    else:
        raise NotImplementedError
    
    
    if return_grid_points:
        return grid, xx, zz, nanmask_grid
    else:
        return grid
    
def percentage_within_circle(x, y):
    """ Calculate the percentage of a pixel that is within a unit circle.
    
        :param xs, ys: Center coordinates of all pixels
        :param dx: Size of one pixel in 1 direction
    """
    
    dx = x[0,1] - x[0,0]
    distances_from_origin = np.sqrt(x**2 + y**2)
    percentages = np.zeros_like(distances_from_origin)
    
    # You can already set the ones inside and outside to 0
    inside_mask = distances_from_origin <= 1 - np.sqrt(2*(dx/2)**2)
    percentages[inside_mask] = 1
    outside_mask = distances_from_origin > 1 + np.sqrt(2*(dx/2)**2)
    percentages[outside_mask] = 0
    
    edge_mask = np.logical_and(~inside_mask, ~outside_mask)
    x_border = x[edge_mask].flatten()
    y_border = y[edge_mask].flatten()
    pct_border = np.zeros_like(x_border)
    for idx, (_x, _y) in enumerate(zip(x_border, y_border)):
        sample_x = np.linspace(_x - dx/2, _x + dx/2, 100)
        sample_y = np.linspace(_y - dx/2, _y + dx/2, 100)
    
        xx, yy = np.meshgrid(sample_x, sample_y)
        dist = np.sqrt(xx**2 + yy**2)
        dist = dist.flatten()
        pct = (dist <= 1).sum() / len(dist)
        pct_border[idx] = pct
    
    percentages[edge_mask] = pct_border
    return percentages
        

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    
    def plot_test_weighting(phi, theta, xx, yy, zz, savename):
        N = 50
        x = xx.flatten()
        y = yy.flatten()
        z = zz.flatten()
        grid, xx, zz, nanmask_grid = project_2d(xx, yy, zz, phi, theta,
                                                np.ones_like(phi)*4500, return_grid_points=True,
                                                N=N, inclination=90, azimuth=0)
        
        fig = plt.figure(figsize=(18,9))
        ax = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122)
        ax.scatter(x, y, z, marker=".", c=theta, vmin=0, vmax=np.pi)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Project
        # ax2.imshow(grid)
        
        percentages = percentage_within_circle(xx, zz)
        img = ax2.scatter(xx, zz, marker=".", c=percentages, vmin=0, vmax=1, s=15,)
        xlim = (-1.1, 1.1)
        ylim = (-1.1, 1.1)
        ax2.plot(xx[percentages > 0], zz[percentages > 0], 
                 c="tab:red", marker="s", fillstyle='none',markersize=8, linestyle="None",)
        
        # ax2.scatter(xx[nanmask_grid], zz[nanmask_grid], marker=".", c=percentages[nanmask_grid], vmin=0, vmax=1, s=15,)
        ax2.plot(xx[percentages < 0], zz[percentages < 0], 
                 c="gray", marker="s", fillstyle='none',markersize=8, linestyle="None",)
        
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel("X'")
        ax2.set_ylabel("Z'")
        
        circle = Circle((0, 0), 1, fill=False)
        ax2.add_patch(circle)
        
        ax2.hlines(0, xlim[0], xlim[1], linestyle="--", linewidth=1, color="black")
        ax2.vlines(0, ylim[0], ylim[1], linestyle="--", linewidth=1, color="black")
        
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        
        # ax3 = fig.add_subplot(133)
        plt.colorbar(img, label="Cell weighting")
        fig.set_tight_layout(True)
        plt.savefig(savename, dpi=300)
        
        
        plt.close()
        
    def plot_test_extrapolation(phi, theta, xx, yy, zz, savename, edge_extapolation):
        N = 50
        x = xx.flatten()
        y = yy.flatten()
        z = zz.flatten()
        grid, xx, zz, nanmask_grid = project_2d(xx, yy, zz, phi, theta, 4500+100*np.sin(phi), return_grid_points=True, N=N, inclination=90, azimuth=0, edge_extapolation=edge_extapolation)
        
        fig = plt.figure(figsize=(18,18))
        ax = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax.scatter(x, y, z, marker=".", c=4500+100*np.sin(phi), vmin=4400, vmax=4600)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        
        percentages = percentage_within_circle(xx, zz)
        grid[percentages <= 0] = np.nan
        img = ax2.scatter(xx, zz, marker=".", c=grid, vmin=4400, vmax=4600, s=30,)
        xlim = (-1.1, 1.1)
        ylim = (-1.1, 1.1)
        ax2.plot(xx[percentages > 0], zz[percentages > 0], 
                 c="tab:red", marker="s", fillstyle='none',markersize=8, linestyle="None",)
        
        # # ax2.scatter(xx[nanmask_grid], zz[nanmask_grid], marker=".", c=percentages[nanmask_grid], vmin=0, vmax=1, s=15,)
        # ax2.plot(xx[percentages < 0], zz[percentages < 0], 
        #          c="gray", marker="s", fillstyle='none',markersize=8, linestyle="None",)
        
        ax2.set_aspect('equal', 'box')
        ax2.set_xlabel("X'")
        ax2.set_ylabel("Z'")
        
        circle = Circle((0, 0), 1, fill=False)
        ax2.add_patch(circle)
        
        ax2.hlines(0, xlim[0], xlim[1], linestyle="--", linewidth=1, color="black")
        ax2.vlines(0, ylim[0], ylim[1], linestyle="--", linewidth=1, color="black")
        
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        
        from matplotlib import cm
        ax3 = fig.add_subplot(223, projection="3d")
        ax3.plot_surface(xx, zz, grid, cmap=cm.viridis,  vmin=4400, vmax=4600,)
        ax3.azim = 45
        ax3.elev = 45
        ax3.set_xlabel("X'")
        ax3.set_ylabel("Z'")
        ax3.set_zlabel("T [K]")
        
        plt.colorbar(img, label="T [K]")
        fig.set_tight_layout(True)
        plt.savefig(savename, dpi=300)
        
        
        plt.close()
    
    # Test a sphere
    phi, theta, xx, yy, zz = get_spherical_phi_theta_x_y_z(N=151)
    plot_test_extrapolation(phi, theta, xx, yy, zz, "3D_cloud_test_extrapol_nearest.png", "nearest")
    plot_test_extrapolation(phi, theta, xx, yy, zz, "3D_cloud_test_extrapol_bispline.png", "bispline")
    
    
    # Test a cube
    # x = np.linspace(-3, 3, 25)
    # y = np.linspace(-2, 2, 25)
    # z = np.linspace(-1, 1, 25)
    
    # xx, yy = np.meshgrid(x, y)
    
    # # yy ,zz = np.meshgrid(y, z)
    # # yy = np.ones_like(xx)
    
    
    # phi = np.zeros_like(xx)
    # theta = np.zeros_like(xx)
    # plot_3d_and_proj(phi, theta, xx, yy, zz, "3D_cuboid.png")
    
    
    
    
    
    
    





