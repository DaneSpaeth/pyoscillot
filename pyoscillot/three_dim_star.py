import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
from scipy.spatial.transform import Rotation as Rot
from scipy.special import sph_harm
import spherical_geometry as geo
from matplotlib import cm
from constants import SIGMA
from sideprojects_scripts.astrometric_jitter import calc_photocenter
import copy
import plot_settings



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

        self.temperature = self.Teff * np.ones(self.phi.shape)
        self.base_Temp = copy.deepcopy(self.temperature)

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
        self.rotation = v * np.sin(self.theta)

    def add_pulsation_rad(self, t, l, m, nu, v_p, k, T_var, T_phase):
        """ Add the radial component of displacement and pulsation.


            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
            
            Displacement + Velocity checked: 2023-09-27
        """
        # Calculate the spherical harmonic Y(l,m)
        # scipy.sph_harm switches the definition of theta and phi, therefore we need
        # to give the params switched around
        harm = sph_harm(m, l, self.phi, self.theta)
    
        
        # Calculate the displacement without any amplitude (we only need it for the T later)
        displ = harm * np.exp(1j * 2 * np.pi * nu * t)

        # Add a factor of 1j. as the pulsations are yet the radial displacements
        # you need to differentiate the displacements wrt t which introduces
        # a factor 1j * 2 * np.pi * nu
        # but we absorb the  2 * np.pi * nu part in the v_p constant
        # See Kochukhov et al. (2004)
        # v_p is now the amplitude of the pulsation in radial direction
        pulsation = 1j * v_p / self.normalization * displ

        self.displacement_rad += displ
        self.pulsation_rad += pulsation

        # Caution temperature is not reset
        # Old version (no normalization)
        # temp_variation = (displ * np.exp(1j * np.radians(T_phase))).real
        
        temp_variation = T_var * (displ * np.exp(1j * np.radians(T_phase)) / np.max(harm.real)).real

        self.temperature += temp_variation

    def add_pulsation_phi(self, t, l, m, nu, v_p, k):
        """ Get phi component of displacement.


            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
            
            Displacement, Pulsation checked: 2023-09-27
        """

        # Calculate the spherical harmonic Y(l,m)
        harmonic = sph_harm(m, l, self.phi, self.theta)
        
        # You need the partial derivative wrt to phi
        part_deriv =  1j * m * harmonic
        displ = 1 / np.sin(self.theta) * part_deriv * np.exp(1j * 2 * np.pi * nu * t)

        pulsation = 1j * k * v_p / self.normalization * displ


        self.displacement_phi += displ
        self.pulsation_phi += pulsation

    def add_pulsation_theta(self, t, l, m, nu, v_p, k):
        """ Get theta component of displacement.

            :param float t: Time at which to evaluate the pulsation
            :param int l: Number of surface lines of nodes
            :param int m: Number of polar lines of nodes (-l<=m<=l)
            
            Partial Derivative checked: 2023-09-27
            Displacement, Pulsation checked: 2023-09-27
        """
        # Calculate the spherical harmonic Y(l,m)
        harmonic = sph_harm(m, l, self.phi, self.theta)
        
        # You need the partial derivative wrt to theta
        # Taken from
        # https://functions.wolfram.com/Polynomials/SphericalHarmonicY/20/ShowAll.html
        # https://functions.wolfram.com/PDF/SphericalHarmonicY.pdf
        if m < l:
            part_deriv = m * 1 / np.tan(self.theta) * harmonic + \
                np.sqrt((l - m) * (l + m + 1)) * np.exp(-1j * self.phi) * \
                sph_harm(m + 1, l, self.phi, self.theta)
        else:
            # The second part of the above equation is 0
            # But in python it will give you a NaN since you still compute
            # The spherical harmonic
            part_deriv = m * 1 / np.tan(self.theta) * harmonic
        
        displ = part_deriv * np.exp(1j * 2 * np.pi * nu * t)

        pulsation = 1j * k * v_p / self.normalization * displ

        self.displacement_theta += displ
        self.pulsation_theta += pulsation

    def add_pulsation(self, t=0, l=1, m=1, nu=1 / 600, v_p=1, k=100,
                      T_var=0, T_phase=0, normalization="max_imaginary"):
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
        # Calc the normalization
        harmonic = sph_harm(m, l, self.phi, self.theta)
        # We want the maximum of the real part since we want only take the real part later
        if normalization == "None":
            self.normalization = 1
        elif normalization == "max_real":
            raise NotImplementedError
        elif normalization == "max_imaginary":
            self.normalization = np.max(harmonic.imag)
            # That the same as writing, tested 2024-01-31
            # self.normalization = np.max((1j*harmonic).real))
            # Fix for m=0 mode
            if self.normalization == 0:
                self.normalization = np.max(harmonic.real)
            print(f"NORM={self.normalization}")
        elif normalization == "max_abs":
            raise NotImplementedError
            self.normalization = np.max(np.abs(harmonic))
        else:
            raise NotImplementedError
            
        
        
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
        
        # However, since we have defined the projection such that 
        # the radial velocity is negative when pointing to the observer
        # i.e. when the amplitude of the radial oscillation is positive
        # we have introduced a minus that we need to invert
        mu *= -1

        return mu

    def starmask(self):
        """ Return a 2D projected starmask."""
        starmask_2d = self._project(self.star.starmask, line_of_sight=False)
        starmask_2d[np.isnan(starmask_2d)] = 0
        starmask_2d = starmask_2d.astype(bool)

        return starmask_2d

    def spotmask(self):
        """ Project the spotmask onto a 2d plane."""
        spotmask_2d = self._project(self.star.spotmask, line_of_sight=False)
        spotmask_2d[spotmask_2d > 0] = 1
        spotmask_2d[np.isnan(spotmask_2d)] = 0
        spotmask_2d = spotmask_2d.astype(bool)
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
        return rotation_2d


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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def plot_for_phd(t=300, l=1, m=1, inclination=90, vmax=1):
    vmin = -vmax
    from pathlib import Path
    out_dir = Path(f"/home/dspaeth/pyoscillot/PhD_plots/")
    if not out_dir.is_dir():
        out_dir.mkdir()
    outfile = out_dir / f"pulsation_components_l{l}_m{m}_t{t}_latex.png"
    if outfile.is_file():
        print(f"Skip File {outfile}")
        # return None
    plt.rcParams.update({'font.size': 8})
    star = ThreeDimStar(N=1000)
    star.add_pulsation(l=l, m=m, v_p=1, k=1, nu=1/600, t=t, )
    projector = TwoDimProjector(star, N=150, border=3, inclination=inclination, line_of_sight=False)
    projector_los = TwoDimProjector(star, N=150, border=3, inclination=inclination, line_of_sight=True)
    
    fig = plt.figure(figsize=(plot_settings.THESIS_WIDTH, 6.0))
    ax0 = fig.add_subplot(331, projection="3d")
    ax1 = fig.add_subplot(332, projection="3d")
    ax2 = fig.add_subplot(333, projection="3d")
    
    three_d_axes = [ax0, ax1, ax2]
    values_list = [star.pulsation_rad.real, 
                   star.pulsation_theta.real,
                   star.pulsation_phi.real]
    
    cmap = "seismic"
    for ax, values in zip(three_d_axes, values_list):
        print("Plot 3D")
        ax.scatter(star.x, star.y, star.z, c=values, vmin=vmin, vmax=vmax, marker=".", cmap=cmap)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    ax0.set_title(r"$r$-component")
    ax1.set_title(r"$\theta$-component")
    ax2.set_title(r"$\phi$-component")
        
    two_d_axes = [fig.add_subplot(330+i) for i in range(4, 10)]
    values_list = [projector.pulsation_rad(),
                   projector.pulsation_theta(),
                   projector.pulsation_phi(),
                   projector_los.pulsation_rad(),
                   projector_los.pulsation_theta(),
                   projector_los.pulsation_phi()]
    for idx, (ax, values) in enumerate(zip(two_d_axes, values_list)):
        img = ax.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
        
        ax.set_xlabel("x'")
        ax.set_ylabel("z'")
        
        # if idx in (3, 4, 5):
            # ax.set_title(f"SUM={round(np.nansum(values),1)}")
            
    cbar_ax = fig.add_axes([0.88, 0.70, 0.02, 0.20])
    fig.colorbar(img, cax=cbar_ax, label=r"$v [\mathrm{m}\,\mathrm{s}^{-1}]$")
    
    cbar_ax = fig.add_axes([0.88, 0.40, 0.02, 0.20])
    fig.colorbar(img, cax=cbar_ax, label=r"$v [\mathrm{m}\,\mathrm{s}^{-1}]$")
    
    cbar_ax = fig.add_axes([0.88, 0.10, 0.02, 0.20])
    fig.colorbar(img, cax=cbar_ax, label=r"RV (proj.) $[\mathrm{m}\,\mathrm{s}^{-1}]$")
    
    # fig.suptitle(f"i={inclination}, l={l}, m={m}, t={t}")
        
        
    fig.subplots_adjust(left=0.10, right=0.78, top=0.97, bottom=0.04, wspace=0.47, hspace=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    
    
def plot_T_variation():
    
    fig = plt.figure(figsize=(16,9))
    cmap = "seismic"
    
    ax = [fig.add_subplot(240+i, projection="3d") for i in range(1, 9)]
    
    for idx, t in zip(range(4), (0, 150, 300, 450)):
        star = ThreeDimStar(N=1000, Teff=4500)
        star.add_pulsation(l=1, m=0, v_p=1, k=1, nu=1/600, t=t, T_var=1)
        ax[idx].scatter(star.x, star.y, star.z, c=star.pulsation_rad.real,
                        vmin=-1, vmax=1, marker=".", cmap=cmap)
        ax[idx+4].scatter(star.x, star.y, star.z, c=star.temperature,
                        vmin=4499, vmax=4501, marker=".", cmap=cmap)
        
    plt.savefig("PhD_plots/temperature_check.png", dpi=300)


def plot_rotation():
    fig = plt.figure(figsize=(16,9))
    cmap = "seismic"
    
    ax0 = fig.add_subplot(232, projection="3d")
    ax1 = fig.add_subplot(234)
    ax2 = fig.add_subplot(235)
    ax3 = fig.add_subplot(236)
    
    star = ThreeDimStar(N=1000, v_rot=3000)
    star.create_rotation()
    ax0.scatter(star.x, star.y, star.z, c=star.rotation,
                vmin=-3000, vmax=3000, marker=".", cmap=cmap)
    
    projector = TwoDimProjector(star, N=151, inclination=90)
    img = ax1.imshow(projector.rotation(), origin="lower", vmin=-3000, vmax=3000, cmap=cmap)
    print(np.nanmax(projector.rotation()))
    print(np.nanmin(projector.rotation()))
    
    
    projector = TwoDimProjector(star, N=151, inclination=60)
    ax2.imshow(projector.rotation(), origin="lower", vmin=-3000, vmax=3000, cmap=cmap)
    print(np.nanmax(projector.rotation()))
    print(np.nanmin(projector.rotation()))
    
    
    projector = TwoDimProjector(star, N=151, inclination=30)
    ax3.imshow(projector.rotation(), origin="lower", vmin=-3000, vmax=3000, cmap=cmap)
    print(np.nanmax(projector.rotation()))
    print(np.nanmin(projector.rotation()))
    
    plt.colorbar(img)
    
    
    
    plt.savefig("PhD_plots/rotation_check_NEW.png", dpi=300)   
        
def plot_temp_map(t=300, l=1, m=1, inclination=90.0):
    from pathlib import Path
    out_dir = Path(f"/home/dspaeth/pyoscillot/PhD_plots/l{l}_incl{inclination}")
    if not out_dir.is_dir():
        out_dir.mkdir()
    outfile = out_dir / f"temp_{l}_m{m}_t{t}.png"
    if outfile.is_file():
        print(f"Skip File {outfile}")
        return None
    plt.rcParams.update({'font.size': 8})
    star = ThreeDimStar(N=1000, Teff=4500)
    star.add_pulsation(l=l, m=m, v_p=1, k=1, nu=1/600, t=t, normalization="None", T_var=100)
    projector = TwoDimProjector(star, N=151, border=3, inclination=inclination, line_of_sight=False)
    projector_los = TwoDimProjector(star, N=151, border=3, inclination=inclination, line_of_sight=True)
    
    fig = plt.figure(figsize=(6.5, 6.0))
    ax0 = fig.add_subplot(211, projection="3d")
    
    
    three_d_axes = [ax0]
    values_list = [star.temperature]
    
    cmap = "seismic"
    for ax, values in zip(three_d_axes, values_list):
        print("Plot 3D")
        ax.scatter(star.x, star.y, star.z, c=values, vmin=4500-100, vmax=4500+100, marker=".", cmap=cmap)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
    ax0.set_title(r"Temperature")
    
        
    two_d_axes = [fig.add_subplot(210+i) for i in range(2, 3)]
    values_list = [projector.temperature()]
    for idx, (ax, values) in enumerate(zip(two_d_axes, values_list)):
        img = ax.imshow(values, vmin=4500-100, vmax=4500+100, cmap=cmap, origin="lower")
        
        ax.set_xlabel("x'")
        ax.set_ylabel("y'")
    
            
    # cbar_ax = fig.add_axes([0.88, 0.70, 0.02, 0.20])
    # fig.colorbar(img, cax=cbar_ax, label=r"$v_p$ [m/s]")
    
    # cbar_ax = fig.add_axes([0.88, 0.40, 0.02, 0.20])
    # fig.colorbar(img, cax=cbar_ax, label=r"$v_p$ [m/s]")
    
    
    fig.suptitle(f"i={inclination}, l={l}, m={m}, t={t}")
        
        
    fig.subplots_adjust(left=0.08, right=0.78, top=0.95, bottom=0.06, wspace=0.45, hspace=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
        
if __name__ == "__main__":
    # star = ThreeDimStar(N=1000)
    # star.add_pulsation(l=1,m=0)
    # star2 = ThreeDimStar(N=1000)
    # star2.add_pulsation(l=2, m=1)
    # exit()
    
    # import matplotlib.pyplot as plt
    # star = ThreeDimStar()
    # star.add_pulsation(normalization="None", v_p=1/0.34545999660276927)
    
    # star2 = ThreeDimStar()
    # star2.add_pulsation(l=1, m=0, normalization="max_imaginary", v_p=0.6)
    
    # # print((star2.pulsation_rad - star.pulsation_rad))
    # # print((star2.pulsation_rad == star.pulsation_rad).all())
    # print(np.max(star2.pulsation_rad.real))
    # inclinations = [0., 45., 90.]
    # for l in range(2, 3):
    #     for m in range(1, l+1):
    #         for inclination in inclinations:
    #             for t in [0, 75, 150, 225, 300, 375, 450, 525]:
    #                 plot_for_phd(t, l, m, inclination)
    # for l in range(2, 5):
    #     for m in range(-l, l+1):
    #         for inclination in inclinations:
    #             for t in [0, 75, 150, 225, 300, 375, 450, 525]:
    #                 plot_for_phd(t, l, m, inclination)
    
    # inclination = 90.0
    # # for t in [0, 75, 150, 225, 300, 375, 450, 525]:
    # for t in [450, 525]:
    #     # for l in range(1, 3):
    #     l = 3
    #     plot_for_phd(t, l, l, inclination)
    
    # plot_rotation()
    
    # l = 2
    # m = -2
    # for t in [0, 75, 150, 225, 300, 375, 450, 525]:
    #     plot_for_phd(t, l, m, vmax=2)
    
    
    # star = ThreeDimStar(N=1000)
    # star.add_pulsation(l=1,m=1)
    plot_rotation()
    exit()
    # exit()
    
    plot_for_phd()
    exit
    
    N = 1000
    (phi,
    theta,
    _, _, _) = geo.get_spherical_phi_theta_x_y_z(N=N)
    
    l = 1
    for m in range(-l, l+1):
        print(l, m)
        harm = sph_harm(m, l, phi, theta)
        # print(np.max(harm.imag))
        # print(np.max(harm.real))
        print(np.max((1j*harm).real))
        assert np.max(harm.imag) == np.max((1j*harm).real)
