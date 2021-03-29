import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
from scipy.special import sph_harm
import spherical_geometry as geo


class ThreeDimStar():
    """ Three Dimensional Star.

        Intended to calculate all masks, displacements, spots etc in
        3d spherical geometry and have a project method to project
        in onto a grid in the end.
    """

    def __init__(self):
        (self.phi,
         self.theta,
         self.x,
         self.y,
         self.z) = geo.get_spherical_phi_theta_x_y_z()

        # Create default maps
        self.starmask = np.ones(self.phi.shape)
        self.spotmask = np.zeros(self.phi.shape)
        self.rotation = np.zeros(self.phi.shape)
        self.displacement_rad = np.zeros(self.phi.shape, dtype='complex128')
        self.displacement_phi = np.zeros(self.phi.shape, dtype='complex128')
        self.displacement_theta = np.zeros(self.phi.shape, dtype='complex128')

    def add_spot(self, rad, theta_pos=90, phi_pos=90):
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

    def create_rotation(self, v=3000):
        """ Create a 3D rotation map.

            :param v: Rotation velocity at equator in m/s.
        """
        self.rotation = -v * np.sin(self.theta)

    def add_displacement_rad(self, l=1, m=1):
        """ Add the radial component of displacement.


            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
        """
        # Calculate the spherical harmonic Y(l,m)
        displ = sph_harm(m, l, self.phi, self.theta)

        self.displacement_rad += displ

    def add_displacement_phi(self, l=1, m=1):
        """ Get phi component of displacement.


            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
        """

        # Calculate the spherical harmonic Y(l,m)
        harmonic = sph_harm(m, l, self.phi, self.theta)
        # You need the partial derivative wrt to phi
        displ = 1 / np.sin(self.theta) * 1j * m * harmonic

        self.displacement_phi += displ

    def add_displacement_theta(self, l=1, m=1):
        """ Get theta component of displacement.


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

        self.displacement_theta = part_deriv


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

    def starmask(self):
        """ Return a 2D projected starmask."""
        starmask_2d = geo.project_2d(self.star.x,
                                     self.star.y,
                                     self.star.z,
                                     self.star.phi,
                                     self.star.theta,
                                     self.star.starmask,
                                     self.N,
                                     border=self.border,
                                     line_of_sight=False)
        starmask_2d[np.isnan(starmask_2d)] = 0
        starmask_2d = starmask_2d.astype(np.bool)

        return starmask_2d

    def spotmask(self):
        """ Project the spotmask onto a 2d plane."""
        spotmask_2d = geo.project_2d(self.star.x,
                                     self.star.y,
                                     self.star.z,
                                     self.star.phi,
                                     self.star.theta,
                                     self.star.spotmask,
                                     self.N,
                                     inclination=self.inclination,
                                     border=self.border,
                                     line_of_sight=False)
        spotmask_2d[spotmask_2d > 0] = 1
        spotmask_2d[np.isnan(spotmask_2d)] = 0
        spotmask_2d = spotmask_2d.astype(np.bool)
        return spotmask_2d

    def rotation(self):
        """ Project rotation onto a 2d plane."""
        rotation_2d = geo.project_2d(self.star.x,
                                     self.star.y,
                                     self.star.z,
                                     self.star.phi,
                                     self.star.theta,
                                     self.star.rotation,
                                     self.N,
                                     inclination=self.inclination,
                                     border=self.border,
                                     line_of_sight=self.line_of_sight,
                                     component="phi")
        return np.rint(rotation_2d).astype(int)

    def displacement_rad(self):
        """ Project the radial displacement of the star onto a 2d plane.


            Caution: Returns only real part
        """
        displacement_rad_2d = geo.project_2d(self.star.x,
                                             self.star.y,
                                             self.star.z,
                                             self.star.phi,
                                             self.star.theta,
                                             self.star.displacement_rad,
                                             self.N,
                                             border=self.border,
                                             inclination=self.inclination,
                                             component="rad",
                                             line_of_sight=self.line_of_sight)

        return displacement_rad_2d.real

    def displacement_phi(self):
        """ Project the phi displacement of the star onto a 2d plane.


            Caution: Returns only real part
        """
        displacement_phi_2d = geo.project_2d(self.star.x,
                                             self.star.y,
                                             self.star.z,
                                             self.star.phi,
                                             self.star.theta,
                                             self.star.displacement_phi,
                                             self.N,
                                             border=self.border,
                                             inclination=self.inclination,
                                             component="phi",
                                             line_of_sight=self.line_of_sight)

        return displacement_phi_2d.real

    def displacement_theta(self):
        """ Project the theta displacement of the star onto a 2d plane.


            Caution: Returns only real part
        """
        displacement_theta_2d = geo.project_2d(self.star.x,
                                               self.star.y,
                                               self.star.z,
                                               self.star.phi,
                                               self.star.theta,
                                               self.star.displacement_theta,
                                               self.N,
                                               border=self.border,
                                               inclination=self.inclination,
                                               component="theta",
                                               line_of_sight=self.line_of_sight)

        return displacement_theta_2d.real


if __name__ == "__main__":
    star = ThreeDimStar()
    projector = TwoDimProjector(star, inclination=45, line_of_sight=True)
    star.add_displacement_rad()
    star.add_displacement_phi(l=2, m=2)
    star.add_displacement_theta(l=2, m=2)

    plt.imshow(projector.displacement_theta(), vmin=-
               1, vmax=1, cmap="seismic", origin="lower")
    plt.show()
