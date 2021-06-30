import matplotlib.pyplot as plt
import numpy as np
from three_dim_star import ThreeDimStar, TwoDimProjector


def plot_transit_limb_darkening_comparison():
    """ Plot a comparison for a spot transit with and without limb darkening.
    """
    phis = np.linspace(0, 200, 50)
    radius = 3
    intensities_limb = []
    intensities_nolimb = []
    for phi in phis:
        star = ThreeDimStar()
        star.add_spot(radius, phi_pos=phi, T_spot=0)
        projector_limb = TwoDimProjector(
            star, N=1000, inclination=90, limb_darkening=True)
        intensity_limb = projector_limb.intensity_stefan_boltzmann_global()
        intensities_limb.append(intensity_limb)

        projector_nolimb = TwoDimProjector(
            star, N=1000, inclination=90, limb_darkening=False)
        intensity_nolimb = projector_nolimb.intensity_stefan_boltzmann_global()
        intensities_nolimb.append(intensity_nolimb)

    intensities_limb = np.array(intensities_limb)
    intensities_limb = intensities_limb / np.max(intensities_limb)

    intensities_nolimb = np.array(intensities_nolimb)
    intensities_nolimb = intensities_nolimb / np.max(intensities_nolimb)

    plt.plot(phis, intensities_limb, "bo", label="Limb darkening")
    plt.plot(phis, intensities_nolimb, "ro", label="No Limb darkening")
    plt.xlabel("Spot Phi Angle")
    plt.ylabel("Relative Intensity")
    plt.legend()
    plt.show()
