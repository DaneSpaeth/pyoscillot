import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
from scipy.spatial.transform import Rotation as Rot
from scipy.special import sph_harm
import spherical_geometry as geo
from matplotlib import cm
from plapy.constants import SIGMA
from plapy.utils.utils import round_digits
from astrometric_jitter import calc_photocenter, calc_astrometric_deviation
import copy
import dataloader as load
from physics import radiance_to_temperature, calc_granulation_velocity_rad, calc_granulation_velocity_phi_theta
import random


class ThreeDimStar():
    """ Three Dimensional Star.

        Intended to calculate all masks, displacements, spots etc in
        3d spherical geometry and have a project method to project
        in onto a grid in the end.
    """

    def __init__(self, Teff=4800, v_rot=3000, logg=3, N=250):
        """ Create a 3d star.

            :param int Teff: effective Temperature [K] of star

        """
        (self.phi,
         self.theta,
         self.x,
         self.y,
         self.z) = geo.get_spherical_phi_theta_x_y_z(N=N)

        self.Teff = Teff

        self.v_rot = v_rot
        self.create_rotation(v_rot)

        self.default_maps()

        self.N = N

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

        # Save the granulation velocity components and the temperature
        self.granulation_rad = np.zeros(self.phi.shape, dtype="float64")
        self.granulation_phi = np.zeros(self.phi.shape, dtype="float64")
        self.granulation_theta = np.zeros(self.phi.shape, dtype="float64")
        self.granulation_temp = np.zeros(self.phi.shape, dtype="float64")

        self.temperature = self.Teff * np.ones(self.phi.shape)
        self.base_Temp = copy.deepcopy(self.temperature)
        self.inner_granule_mask = np.zeros(self.phi.shape, dtype="bool")
        self.granular_lane_mask = np.zeros(self.phi.shape, dtype="bool")

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
        print(f"Add spot with temperature {T_spot}")
        self.temperature[self.spotmask.astype(bool)] = T_spot

    def add_granulation(self):
        """ Add granulation to the star starting from Hans models"""
        size = 160
        assert self.N % size == 0, "For granulation we can only create stars with multiples of the cell size (160)"
        N_cells = int(self.N / size)


        granulation_spectral_radiance = load.granulation_map()
        granulation_temperature = radiance_to_temperature(granulation_spectral_radiance)
        timestemp = 992 + 1292
        granulation_temp_local = granulation_temperature[timestemp, :, :]
        granulation_rad_local = calc_granulation_velocity_rad(granulation_temp_local)
        (granulation_phi_local,
         granulation_theta_local,
         _,
         _) = calc_granulation_velocity_phi_theta(granulation_temp_local, granulation_rad_local)

        for i in range(N_cells):
            for j in range(N_cells):
                # timestemp = random.randint(0, intensity.shape[0])
                idx_phi = 0 + i * int(self.N / N_cells)
                idx_theta = 0 + j * int(self.N / N_cells)
                self.temperature[idx_theta:idx_theta + size, idx_phi:idx_phi + size] = granulation_temp_local

                self.granulation_rad[idx_theta:idx_theta + size, idx_phi:idx_phi + size]  = granulation_rad_local
                self.granulation_phi[idx_theta:idx_theta + size, idx_phi:idx_phi + size] = granulation_phi_local
                self.granulation_theta[idx_theta:idx_theta + size, idx_phi:idx_phi + size] = granulation_theta_local

    def get_distance(self, phi_center, theta_center):
        """ Return the great circle distance from position given by phi and theta.
            https://en.wikipedia.org/wiki/Great-circle_distance
        """
        latitude = self.theta - np.radians(90)
        latitude_center = theta_center - np.radians(90)
        #sin(np.sqrt(np.sin((latitude-latitude_center)/2)**2+
        #                       np.cos(latitude_center)*np.cos(latitude)*np.sin((self.theta-theta_center)/2)**2))
        # distance[np.isnan(distance)] = np.max(distance)


        distance = np.arccos(np.cos(latitude)*np.cos(latitude_center)*np.cos(np.abs(self.phi-phi_center))+
                             np.sin(latitude)*np.sin(latitude_center))
        # self.distance = distance
        return distance


    def create_rotation(self, v=3000):
        """ Create a 3D rotation map.

            :param v: Rotation velocity at equator in m/s.
        """
        self.rotation = -v * np.sin(self.theta)

    def add_pulsation_rad(self, t, l, m, nu, v_p, k, T_var, T_phase):
        """ Add the radial component of displacement and pulsation.


            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
        """
        # Calculate the spherical harmonic Y(l,m)
        harm = sph_harm(m, l, self.phi, self.theta)

        displ = harm * np.exp(1j * 2 * np.pi * nu * t)

        # Add a factor of 1j. as the pulsations are yet the radial displacements
        # you need to differentiate the displacements wrt t which introduces
        # a factor 1j * 2 * np.pi * nu
        # but we absorb the  2 * np.pi * nu part in the v_p constant
        # See Kochukhov et al. (2004)
        pulsation = 1j * v_p * displ

        self.displacement_rad += displ
        self.pulsation_rad += pulsation

        # Caution temperature is not reset
        temp_variation = (displ * np.exp(1j * np.radians(T_phase))).real
        temp_variation = T_var * temp_variation / np.nanmax(temp_variation)

        self.temperature += temp_variation

    def add_pulsation_phi(self, t, l, m, nu, v_p, k):
        """ Get phi component of displacement.


            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
        """

        # Calculate the spherical harmonic Y(l,m)
        harmonic = sph_harm(m, l, self.phi, self.theta)
        # You need the partial derivative wrt to phi
        part_deriv = 1 / np.sin(self.theta) * 1j * m * harmonic
        displ = part_deriv * np.exp(1j * 2 * np.pi * nu * t)

        pulsation = 1j * k * v_p * displ


        self.displacement_phi += displ
        self.pulsation_phi += pulsation

    def add_pulsation_theta(self, t, l, m, nu, v_p, k):
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

        displ = part_deriv * np.exp(1j * 2 * np.pi * nu * t)

        pulsation = 1j * k * v_p * displ

        self.displacement_theta += displ
        self.pulsation_theta += pulsation

    def add_pulsation(self, t=0, l=1, m=1, nu=1 / 600, v_p=1, k=100,
                      T_var=0, T_phase=0):
        """ Convenience function to add all pulsations in one go.

            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
            :param nu: Pulsation frequency (without 2pi factor)
            :param v_p: Pulsation velocity in m/s
            :param k: Ratio between radial component and phi/theta component
                      1.2 for g-mode (compare to Hatzes1996)
                      (that info is probably a bit wrong)
                      probably closer to 100 for g-mode
            :param float T_var: Amplitude[K] of Temp variation due to pulsation
            :parm float T_phase: Phase shift of Temp variation wrt to radial
                                 displacement
        """
        self.add_pulsation_rad(t, l, m, nu, v_p, k, T_var, T_phase)
        self.add_pulsation_phi(t, l, m, nu, v_p, k)
        self.add_pulsation_theta(t, l, m, nu, v_p, k)

    def intensity_stefan_boltzmann(self):
        """ Calculate the intensity using the stefan boltzmann law."""
        self.intensity = SIGMA * self.temperature**4

        self.intensity = self.intensity

        return self.intensity


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
        
        # Calculate the weights for the edges
        _, xx, zz, _ = geo.project_2d(self.star.x,
                                                    self.star.y,
                                                    self.star.z,
                                                    self.star.phi,
                                                    self.star.theta,
                                                    np.zeros_like(self.star.x),
                                                    self.N,
                                                    inclination=self.inclination,
                                                    border=self.border,
                                                    line_of_sight=False,
                                                    component=None,
                                                    return_grid_points=True)
        self.weights = geo.percentage_within_circle(xx, zz)
        print(self.weights)
        


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

    def mu(self):
        """ Return a 2D map of the angle mu.

            According to Claret et al. 2014 mu is defined as
            mu = cos(gamma),
            where gamma is the angle between the line of sight and the surface
            normal.

        """
        # Try to be clever
        # Create an array with ones of the same shape as the 3D star
        unit_array = np.ones(self.star.phi.shape)

        # Now project the ones onto the radial component
        # This gives you the cos of the angle between the line of sight
        # and the radial unit vector
        # this should be exactly the definition of mu
        mu = self._project(unit_array,
                           line_of_sight=True,
                           component="rad")

        return mu

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
        print(f"Rotation projection {self.line_of_sight}")
        rotation_2d = self._project(self.star.rotation,
                                    line_of_sight=self.line_of_sight,
                                    component="phi")
        # return np.rint(rotation_2d).astype(int)
        return rotation_2d

    def granulation_rad(self):
        """ Project the radial part of the granulation velocity onto a 2d plane"""
        granulation_rad_2d = self._project(self.star.granulation_rad,
                                           line_of_sight=self.line_of_sight,
                                           component="rad")
        return granulation_rad_2d

    def granulation_phi(self):
        """ Project the phi part of the granulation velocity onto a 2d plane"""
        granulation_phi_2d = self._project(self.star.granulation_phi,
                                           line_of_sight=self.line_of_sight,
                                           component="phi")
        return granulation_phi_2d

    def granulation_theta(self):
        """ Project the theta part of the granulation velocity onto a 2d plane"""
        granulation_theta_2d = self._project(self.star.granulation_theta,
                                             line_of_sight=self.line_of_sight,
                                            component="theta")
        return granulation_theta_2d

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
            print("LINE OF SIGHT IS TRUE!")
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

    def intensity_stefan_boltzmann(self):
        """ Project the intensity using the Stefan Boltzmann Law."""
        intensity = self.star.intensity_stefan_boltzmann()
        intensity_2d = self._project(intensity, line_of_sight=False)
        intensity_2d[np.isnan(intensity_2d)] = 0

        self.calc_photocenter(intensity_2d)
        return intensity_2d

    def calc_photocenter(self, intensity_2d):
        """ Compute the photocenter and its difference from the geometric center.
        """
        self.photocenter = calc_photocenter(intensity_2d)
        self.geom_center = calc_photocenter(self.starmask())
        self.diff_photocenter = (self.photocenter[0] - self.geom_center[0],
                                 self.photocenter[1] - self.geom_center[1])

        return self.photocenter


    def intensity_stefan_boltzmann_global(self):
        """ Get the integrated itensity using the Stefan-Boltzmann law."""
        intensity_2d = self.intensity_stefan_boltzmann()
        global_intensity = np.nansum(intensity_2d)

        return global_intensity

    def radial_velocity(self):
        """ Get the integrated radial velocity. This is in a sense a shortcut
            without creating spectra first.
        """
        pulsation_2d = self.pulsation()
        rv = np.nanmean(pulsation_2d)
        return rv


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
    
    import matplotlib.pyplot as plt
    star = ThreeDimStar(N=1000)
    # star.create_rotation()
    # star.add_pulsation(T_var=100, l=1, m=1, v_p=4, k=100, nu=1/698.61)
    projector = TwoDimProjector(star, N=150, border=3, inclination=90)
    projector.weights

    plt.imshow(projector.weights, vmin=0, vmax=1)
    
    # print(projector.mu())
    # print(np.nanmean(projector.mu()))
    plt.savefig("dbug.png", dpi=500)



