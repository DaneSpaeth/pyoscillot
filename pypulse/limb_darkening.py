def schwarzschild_law(limb_angle):
    """ Following the PhD thesis of Müller.

        Use the limb_angle, i.e. the cosine of the agnle between LOS and
        radial unit vector.

        Return the Ratio of F(limb_angle)/F(center)

        So you must calculate the exact F yourself
    """
    return 1 - 2 / 3 * (1 - limb_angle)
