import math


def pixels_per_degree(display_diagonal_mm, height_pix, width_pix, viewing_distance_m):
    """
    computer pixels per degree given display parameters and viewing distance

    This is a convenience function that can be used to provide angular
    resolution of input images for the HDR-VDP-2.

    Note that the function assumes 'square' pixels, so that the aspect ratio
    is resolution[0]:resolution[1].

    Parameters
    ----------
    display_diagonal_mm : int or float
        diagonal display size in millimeters

    height_pix, width_pix : int
        display resolution in pixels as a pair of int, e.g. (1024, 768)

    viewing_distance_m : float
        viewing distance in meters, e.g. 0.5
    """
    ar = width_pix / height_pix  # aspect ratio

    height_mm = (display_diagonal_mm ** 2 / (1 + ar ** 2)) ** .5

    height_rad = 2 * math.atan(0.5 * height_mm / (viewing_distance_m * 1000))
    height_deg = math.degrees(height_rad)

    ppd = height_pix / height_deg
    return ppd
