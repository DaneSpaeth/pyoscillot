import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from scipy.special import sph_harm
import spherical_geometry as geo
from matplotlib import cm
import random


class ThreeDimStar():
    """ Three Dimensional Star.

        Intended to calculate all masks, displacements, spots etc in
        3d spherical geometry and have a project method to project
        in onto a grid in the end.
    """

    def __init__(self, nu=1 / 600, v_p=50, k=1.2, Teff=4800, T_var=0, T_phase=0):
        """ Create a 3d star.

            :param nu: Pulsation frequency (without 2pi factor)
            :param V_p: Pulsation velocity in m/s
            :param k: Ratio between radial component and phi/theta component
                      1.2 for g-mode (compare to Hatzes1996)
            :param int Teff: effective Temperature [K] of star
            :param float T_var: Amplitude[K] of Temp variation due to pulsation
            :parm float T_phase: Phase shift of Temp variation wrt to radial
                                 displacement
        """
        (self.phi,
         self.theta,
         self.x,
         self.y,
         self.z) = geo.get_spherical_phi_theta_x_y_z()

        self.nu = nu
        self.v_p = v_p
        self.k = k
        self.T_var = T_var
        self.T_phase = T_phase
        self.Teff = Teff

        self.default_maps()

    def default_maps(self):
        """ Create default maps."""
        # Create default maps
        self.starmask = np.ones(self.phi.shape)
        self.spotmask = np.zeros(self.phi.shape)
        self.rotation = np.zeros(self.phi.shape)
        # Displacement describes the actual deformation of the star
        self.displacement_rad = np.zeros(self.phi.shape, dtype="complex128")
        self.displacement_phi = np.zeros(self.phi.shape, dtype="complex128")
        self.displacement_theta = np.zeros(self.phi.shape, dtype="complex128")
        # Pulsation describes the change of deformation of the star
        # i.e. the velocity per element
        self.pulsation_rad = np.zeros(self.phi.shape, dtype="complex128")
        self.pulsation_phi = np.zeros(self.phi.shape, dtype="complex128")
        self.pulsation_theta = np.zeros(self.phi.shape, dtype="complex128")

        self.temperature = self.Teff * np.ones(self.phi.shape)
        self.base_temp = self.temperature

    def add_spot(self, rad, theta_pos=90, phi_pos=90, T_spot=4000):
        """ Add a spot to the 3D star.

            Idea: The radius defines the angle in theta that is covered by the spot
                  Calculate the distance from the pole in theta with that radius
                  Define a plane by one point on that circle and the normal vector
                  Rotate these two vectors wrt to theta_pos and phi_pos
                  Calculate a mask with all points on the sphere that are above the
                  plane by using the sign of the scalar product of the normal vec
                  and the difference between the point on the plane and each point


            :param rad: Radius in degree
            :param theta_pos: Theta Position in degree. 0 defined as pole, 90 center
            :param phi_pos: Phi Position. 0 defined as left edge, 90 center
        """
        # Calculate where the star is sliced by the plane
        z_slice = np.cos(np.radians(rad))
        # define plane
        # plane normal
        normal = np.array([0, 0, 1])
        # Fixed point in plane
        p = np.array([0, 0, z_slice])

        # Rotate the two normal vector and the vector to the point in the plane
        rot = Rot.from_euler('xz', [theta_pos, phi_pos + 90], degrees=True)
        normal = rot.apply(normal)
        p = rot.apply(p)

        # above_plane_mask = np.zeros(phi.shape).flatten()
        vecs = np.array((self.x.flatten(),
                         self.y.flatten(),
                         self.z.flatten())).T
        above_plane_mask = np.dot(vecs - p, normal) >= 0
        above_plane_mask = above_plane_mask.reshape(self.phi.shape)

        self.spotmask += above_plane_mask
        self.temperature[self.spotmask.astype(bool)] = T_spot

    def add_granulation(self, planes=3500):
        """ First try to add granulation cells to the star.

           :param int cells: Number of cells (not implemented yet)
        """
        # Define vectors of all points on sphere
        vecs = np.array((self.x.flatten(),
                         self.y.flatten(),
                         self.z.flatten())).T

        # Define center of sphere
        center = np.array([0, 0, 0])
        vecs = vecs - center
        self.border_mask = np.zeros(self.phi.shape)
        for n in range(planes):
            normal = np.array(
                [random.uniform(-1, 1),
                 random.uniform(-1, 1),
                 random.uniform(-1, 1)])
            normal = normal / np.dot(normal, normal)

            close_plane_mask = np.abs(np.dot(vecs, normal)) <= 0.0001
            close_plane_mask = close_plane_mask.reshape(self.phi.shape)

            self.temperature[close_plane_mask] = self.Teff - 400
            self.border_mask[close_plane_mask] = 1
        print(
            f"Border Value: {np.sum(self.border_mask)/np.size(self.border_mask)}")

    def create_rotation(self, v=3000):
        """ Create a 3D rotation map.

            :param v: Rotation velocity at equator in m/s.
        """
        self.rotation = -v * np.sin(self.theta)

    def add_pulsation_rad(self, t=0, l=1, m=1):
        """ Add the radial component of displacement and pulsation.


            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
        """
        # Calculate the spherical harmonic Y(l,m)
        harm = sph_harm(m, l, self.phi, self.theta)
        displ = harm * np.exp(1j * 2 * np.pi * self.nu * t)

        # Add a factor of 1j. as the pulsations are yet the radial displacements
        # you need to differentiate the displacements wrt t which introduces
        # a factor 1j * 2 * np.pi * nu
        # but we absorb the  2 * np.pi * nu part in the V_p constant
        # See Kochukhov et al. (2004)
        pulsation = 1j * self.v_p * displ

        self.displacement_rad = displ
        self.pulsation_rad = pulsation
        # Caution temperature is not reseted
        temp_variation = (displ * np.exp(1j * self.T_phase)).real
        temp_variation = self.T_var * \
            temp_variation / np.nanmax(temp_variation)

        self.temperature = self.base_temp + temp_variation

    def add_pulsation_phi(self, t=0, l=1, m=1):
        """ Get phi component of displacement.


            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
        """

        # Calculate the spherical harmonic Y(l,m)
        harmonic = sph_harm(m, l, self.phi, self.theta)
        # You need the partial derivative wrt to phi
        part_deriv = 1 / np.sin(self.theta) * 1j * m * harmonic
        displ = part_deriv * np.exp(1j * 2 * np.pi * self.nu * t)

        pulsation = 1j * self.k * self.v_p * displ

        print(f"K IS {self.k}")
        print(f"V_P IS {self.v_p}")

        self.displacement_phi = displ
        self.pulsation_phi = pulsation

    def add_pulsation_theta(self, t=0, l=1, m=1):
        """ Get theta component of displacement.

            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
        """
        # Calculate the spherical harmonic Y(l,m)
        harmonic = sph_harm(m, l, self.phi, self.theta)
        # You need the partial derivative wrt to theta
        # Taken from
        # https://functions.wolfram.com/Polynomials/SphericalHarmonicY/20/ShowAll.html
        if m < l:
            part_deriv = m * 1 / np.tan(self.theta) * harmonic + \
                np.sqrt((l - m) * (l + m + 1)) * np.exp(-1j * self.phi) * \
                sph_harm(m + 1, l, self.phi, self.theta)
        else:
            part_deriv = m * 1 / np.tan(self.theta) * harmonic

        displ = part_deriv * np.exp(1j * 2 * np.pi * self.nu * t)

        pulsation = 1j * self.k * self.v_p * displ

        self.displacement_theta = displ
        self.pulsation_theta = pulsation

    def add_pulsation(self, t=0, l=1, m=1):
        """ Convenience function to add all pulsations in one go.

            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
        """
        self.add_pulsation_rad(t, l, m)
        self.add_pulsation_phi(t, l, m)
        self.add_pulsation_theta(t, l, m)


class TwoDimProjector():
    """ Project a 3D star onto a 2D plane."""

    def __init__(self, star, N=1000, border=10, inclination=90, azimuth=0,
                 line_of_sight=True):
        """ Construct Projector object.

            :param star: Instance of 3D star to project
            :param int N: Number of cells in 1d on star
            :param int border: Number of border cells next to the star
            :param inclination: Inclination angle (0:pole on, 90:equator on)
            :param azimuth: Azimuth angle (not Implemented yet)
        """
        self.star = star
        self.N = N
        self.border = border
        self.inclination = inclination
        self.azimuth = azimuth
        self.line_of_sight = line_of_sight

    def _project(self, values, line_of_sight=False, component=None):
        """ Helper function to project stuff.

            :param values: Values to project
            :param line_of_sight: If True, line of sight projection will be
                                  enabled
            :param component: Unit direction in which to project if
                              line of sight is True
        """
        if line_of_sight:
            assert component is not None, "Component missing for projection"

        projection = geo.project_2d(self.star.x,
                                    self.star.y,
                                    self.star.z,
                                    self.star.phi,
                                    self.star.theta,
                                    values,
                                    self.N,
                                    inclination=self.inclination,
                                    border=self.border,
                                    line_of_sight=line_of_sight,
                                    component=component)
        return projection

    def starmask(self):
        """ Return a 2D projected starmask."""
        starmask_2d = self._project(self.star.starmask, line_of_sight=False)
        starmask_2d[np.isnan(starmask_2d)] = 0
        starmask_2d = starmask_2d.astype(np.bool)

        return starmask_2d

    def spotmask(self):
        """ Project the spotmask onto a 2d plane."""
        spotmask_2d = self._project(self.star.spotmask, line_of_sight=False)
        spotmask_2d[spotmask_2d > 0] = 1
        spotmask_2d[np.isnan(spotmask_2d)] = 0
        spotmask_2d = spotmask_2d.astype(np.bool)
        return spotmask_2d

    def temperature(self):
        """ Project the temperature onto a 2d plane."""
        tempmap = self._project(self.star.temperature, line_of_sight=False)
        return tempmap

    def rotation(self):
        """ Project rotation onto a 2d plane."""
        rotation_2d = self._project(self.star.rotation,
                                    line_of_sight=self.line_of_sight,
                                    component="phi")
        return np.rint(rotation_2d).astype(int)

    def displacement_rad(self):
        """ Project the radial displacement of the star onto a 2d plane.


            Caution: Returns only real part
        """
        projection = self._project(self.star.displacement_rad,
                                   line_of_sight=self.line_of_sight,
                                   component="rad")

        return projection.real

    def pulsation_rad(self):
        """ Project the radial pulsation of the star onto a 2d plane.


            Caution: Returns only real part
        """
        projection = self._project(self.star.pulsation_rad,
                                   line_of_sight=self.line_of_sight,
                                   component="rad")

        return projection.real

    def displacement_phi(self):
        """ Project the phi displacement of the star onto a 2d plane.


            Caution: Returns only real part
        """
        projection = self._project(self.star.displacement_phi,
                                   line_of_sight=self.line_of_sight,
                                   component="phi")

        return projection.real

    def pulsation_phi(self):
        """ Project the phi pulsation of the star onto a 2d plane.


            Caution: Returns only real part
        """
        projection = self._project(self.star.pulsation_phi,
                                   line_of_sight=self.line_of_sight,
                                   component="phi")

        return projection.real

    def displacement_theta(self):
        """ Project the theta displacement of the star onto a 2d plane.


            Caution: Returns only real part
        """
        projection = self._project(self.star.displacement_theta,
                                   line_of_sight=self.line_of_sight,
                                   component="theta")

        return projection.real

    def pulsation_theta(self):
        """ Project the theta displacement of the star onto a 2d plane.


            Caution: Returns only real part
        """
        projection = self._project(self.star.pulsation_theta,
                                   line_of_sight=self.line_of_sight,
                                   component="theta")

        return projection.real

    def pulsation(self):
        """ Project the complete pulsation of the star onto a 2d plane.

            Difference to previous versions:
            First we project each individual component onto the line of sight
            Then we add them up and in the end project them onto the 2d plane
            In that way the relatively long 2d projection is done only once

            Caution: Returns only real part
        """
        # los = line of sight
        p = self.star.phi.flatten()
        t = self.star.theta.flatten()
        if self.line_of_sight:
            rad_los = geo.project_line_of_sight(p,
                                                t,
                                                self.star.pulsation_rad.flatten(),
                                                "rad",
                                                inclination=self.inclination)
            rad_los = rad_los.reshape(self.star.phi.shape)

            phi_los = geo.project_line_of_sight(p,
                                                t,
                                                self.star.pulsation_phi.flatten(),
                                                "phi",
                                                inclination=self.inclination)
            phi_los = phi_los.reshape(self.star.phi.shape)

            theta_los = geo.project_line_of_sight(p,
                                                  t,
                                                  self.star.pulsation_theta.flatten(),
                                                  "theta",
                                                  inclination=self.inclination)
            theta_los = theta_los.reshape(self.star.phi.shape)
        else:
            print("CAUTION! ADDING PULSATIONS WITHOUT PROJECTION ONTO THE " +
                  "LINE OF SIGHT DOES NOT MAKE SENSE!")
            rad_los = self.star.pulsation_rad
            phi_los = self.star.pulsation_phi
            theta_los = self.star.pulsation_theta

        # Only valid since we projected onto the line of sight
        pulsation_3d_los = rad_los + phi_los + theta_los
        # Project onto 2d plane
        # Important! Make sure not to project onto the line of sight again!
        pulsation_2d = self._project(pulsation_3d_los, line_of_sight=False)

        return pulsation_2d.real

    def grid(self):
        """ Add a 2d projection of a grid."""
        longitude_lines = np.arange(0, 360, 30)
        grid = np.zeros(self.star.theta.shape)
        for longline in longitude_lines:

            grid += np.where(np.abs(self.star.phi -
                                    np.radians(longline)) < 0.01, 1, 0)

        latitude_lines = np.arange(0, 180, 30)
        for latline in latitude_lines:
            grid += np.where(np.abs(self.star.theta -
                                    np.radians(latline)) < 0.01, 1, 0)

        grid_2d = self._project(grid, line_of_sight=False)
        grid_2d[grid_2d > 0] = 1
        grid_2d[np.isnan(grid_2d)] = 0
        grid_2d = grid_2d.astype(bool)
        return grid_2d


def plot_3d(x, y, z, value, scale_down=1):
    """ Plot the values in 3d."""
    # Calculate the colors from the values
    if value.dtype == "complex128":
        value = value.real
    vmax, vmin = np.nanmax(value), np.nanmin(value)
    print(vmax, vmin)
    if vmax == vmin:
        vmin = 0
    value = (value - vmin) / (vmax - vmin)
    print(value.min())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z,
                    facecolors=cm.seismic(value),
                    rstride=1, cstride=1)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60)

    # Create plots
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(projector.temperature(), origin="lower",
              cmap="hot", vmin=0, vmax=8000)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/temp.pdf")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60)
    star.create_rotation()
    rot_map = projector.rotation()
    rot_map = np.where(np.abs(rot_map) < 1e-5, np.nan, rot_map)
    ax.imshow(rot_map, origin="lower",
              cmap="seismic", vmin=-3000, vmax=3000)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/rotation.pdf")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60, line_of_sight=False)
    star.add_pulsation(l=2, m=2)
    ax.imshow(projector.pulsation_rad(), origin="lower",
              cmap="seismic", vmin=-50, vmax=50)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/pulse_rad_nolos.pdf")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60, line_of_sight=False)
    star.add_pulsation(l=2, m=2)
    ax.imshow(projector.pulsation_phi(), origin="lower",
              cmap="seismic", vmin=-50, vmax=50)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/pulse_phi_nolos.pdf")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60, line_of_sight=False)
    star.add_pulsation(l=2, m=2)
    ax.imshow(projector.pulsation_theta(), origin="lower",
              cmap="seismic", vmin=-50, vmax=50)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/pulse_theta_nolos.pdf")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
    star.add_pulsation(l=2, m=2)
    ax.imshow(projector.pulsation_rad(), origin="lower",
              cmap="seismic", vmin=-50, vmax=50)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/pulse_rad_proj.pdf")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
    star.add_pulsation(l=2, m=2)
    ax.imshow(projector.pulsation_phi(), origin="lower",
              cmap="seismic", vmin=-50, vmax=50)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/pulse_phi_proj.pdf")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
    star.add_pulsation(l=2, m=2)
    ax.imshow(projector.pulsation_theta(), origin="lower",
              cmap="seismic", vmin=-50, vmax=50)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/pulse_theta_proj.pdf")
    plt.close()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=60, line_of_sight=True)
    star.add_pulsation(l=2, m=2)
    ax.imshow(projector.pulsation(), origin="lower",
              cmap="seismic", vmin=-50, vmax=50)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("imprs_plots/pulse_all_proj.pdf")
    plt.close()
