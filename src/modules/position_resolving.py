from astropy.wcs import WCS
import numpy as np

def resolve_position(pixel_size: float, ori_xpos: int, ori_ypos: int, 
                      obj_xpos: int, obj_ypos: int, 
                      ori_ra: float, ori_dec: float) -> tuple[float, float]:
    '''resolves the RA and DEC of a object from the original image'''
    pixel_scale = np.sqrt(pixel_size)
    pixel_scale_deg = pixel_scale / 3600.0  

    # Create WCS object
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [ori_xpos, ori_ypos]  # Reference pixel
    wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg]  # Pixel scale in degrees
    wcs.wcs.crval = [ori_ra, ori_dec]  # Reference RA and DEC
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Projection type

    # Compute RA, DEC of the given pixel position
    sky_coord = wcs.pixel_to_world(obj_xpos, obj_ypos)  # Use xpos and ypos as separate arguments

    # Return RA and DEC in degrees
    return sky_coord.ra.deg, sky_coord.dec.deg