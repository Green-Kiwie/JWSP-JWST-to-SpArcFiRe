import numpy as np
import sep_pjw as sep
import matplotlib.pyplot as plt
import os
import math

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from astropy.wcs import WCS
from matplotlib.patches import Ellipse

def _fill_nan(image_data: np.ndarray) -> np.ndarray:
    '''uses the astropy function convolute to fill in nan values'''
    gauss_kernal = Gaussian2DKernel(1)
    convolved_data = interpolate_replace_nans(image_data, gauss_kernal)
    return convolved_data

def get_main_fits_data(filepath: str) -> np.ndarray:
    '''opens fits file and returns the 'sci' data '''
    hdul = fits.open(filepath)

    try:
        image_data = hdul['SCI'].data
    except:
        image_data = hdul[0].data

    image_data_no_nan = _fill_nan(image_data)
    byte_swapped_data = image_data_no_nan.byteswap().view(image_data_no_nan.dtype.newbyteorder('='))
    hdul.close()
    return byte_swapped_data

def get_all_fits_meta_data(filepath: str) -> tuple['astropy.header', 'astropy.header.comments']:
    '''gets all meta data of file, 
    both astropy.header and astropy.header.comments can be accessed like dictionaries'''
    hdul = fits.open(filepath)

    try:
        primary_data = hdul['PRIMARY'].header
        primary_data_comments = hdul['PRIMARY'].header.comments
    except:
        primary_data = hdul[0].header
        primary_data_comments = hdul[0].header.comments

    hdul.close()
    return primary_data, primary_data_comments

def get_relevant_fits_meta_data(filepath: str) -> dict:
    '''
    gets relevant meta data to calculating meta data for extracted objects.
    data returned as a dict
    Note: might need to account for offset caused by dither'''
    hdul = fits.open(filepath)

    relevant_meta_data = dict()
    # print(hdul.info())

    try:
        primary_data = hdul['PRIMARY'].header
    except:
        primary_data = hdul[0].header

    # print("primary header: ", primary_data)
    meta_data_list = [('DATE-BEG', 'DATE-BEG', 'Begin datetime of the exposure'),
     ('DATE-END', 'DATE-END', 'END datetime of the exposure'),
     ('OBS_ID', 'OBS_ID', 'Observation ID'),
     ('TARG_RA', 'TARG_RA', 'RA Position of Target'),
     ('TARG_DEC', 'TARG_DEC', 'DEC Position of Target'),
     ('EFFEXPTM', 'EFFEXPTM','Effective exposure time of image in second'),
     ('ORIAXIS1', 'SUBSIZE1', 'number of pixels in original image axis 1'),
     ('ORIAXIS2', 'SUBSIZE2', 'number of pixels in original image axis 2')
    ]
    for new_key, key, description in meta_data_list:
        relevant_meta_data[new_key] = (primary_data[key], description)
    
    try:
        image_data_header = hdul['SCI'].header
    except:
        primary_data = hdul[0].header

    # print("sci header", image_data_header)
    relevant_meta_data['PIXAR_A2'] = (image_data_header['PIXAR_A2'], 'Area of each pixel in arcsecond^2')
    
    hdul.close()
    return relevant_meta_data
    
def scale_fits_data(image_data: np.ndarray) -> np.ndarray:
    '''scales the image using the asinh function'''
    arcsin_data = np.arcsinh(image_data)
    return arcsin_data

def show_fits_image(image_data: np.ndarray, title = 'FITS Image') -> None:
    '''prints the image data using matplotlib'''
    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.show()

def get_image_background(image_data: np.ndarray, show = False) -> np.ndarray:
    '''returns the background of the data as a ndarray'''
    background_image = sep.Background(image_data)
    background_image_arr = background_image.back()
    if show:
        show_fits_image(background_image_arr, 'background data')
    return background_image_arr

def get_bkg_rms(background_data: np.ndarray):
    '''calculates the rms value of a background image'''
    print(type(background_data.rms()))
    return background_data.rms()

def subtract_bkg(image_data: np.ndarray) -> np.ndarray:
    '''calculates the subtract the background data'''
    background_data = get_image_background(image_data)
    backgroundless_data = image_data - background_data
    return backgroundless_data

def extract_objects(clean_image_data: np.ndarray, background_rms) -> np.ndarray:
    '''returns array of celestial objects'''
    objects = sep.extract(clean_image_data, 1.5, err=background_rms)
    print(f"{len(objects)} objects found.")
    return objects

def plot_object_mask_on_galaxy(scaled_image_data: np.ndarray, celestial_objects: np.ndarray) -> None:
    '''plots the galaxy with cirles for the celestial objects'''
    #plot image
    fig, ax = plt.subplots()
    m, s = np.mean(scaled_image_data), np.std(scaled_image_data)
    im = ax.imshow(scaled_image_data, cmap='gray', origin='lower')

    # plot an ellipse for each object
    for i in range(len(celestial_objects)):
        e = Ellipse(xy=(celestial_objects['x'][i], celestial_objects['y'][i]),
                    width=6*celestial_objects['a'][i],
                    height=6*celestial_objects['b'][i],
                    angle=celestial_objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)

    plt.show()

def _get_thumbnail_coord(max_half_size: int, x_coord: int, y_coord: int, image_data: np.ndarray) -> tuple[int, int, int, int]:
    '''gets the bounding box vertices of a thumbnail'''
    x_min = max(0, x_coord - max_half_size)
    x_max = min(image_data.shape[1], x_coord + max_half_size)
    y_min = max(0, y_coord - max_half_size)
    y_max = min(image_data.shape[0], y_coord + max_half_size)
    return x_min, x_max, y_min, y_max

def _save_to_fits(filepath: str, image_data: np.ndarray, image_meta_data: dict) -> None:
    '''saves a file to a non multi extension fits file'''
    hdu = fits.PrimaryHDU(data=image_data)
    for key in image_meta_data:
        hdu.header[key] = image_meta_data[key]
    
    hdu.writeto(filepath, overwrite=True)

def _extract_object(obj: np.void, image_data: np.ndarray, padding: float, min_size: int, verbosity: int) -> np.ndarray|None:
    '''crop out the object from the main data and returns the cropped image. returns none if crop too small'''
    x_center, y_center = int(obj['x']), int(obj['y'])
   
    try: # Determine the bounding box size based on the object's dimensions
        max_half_size = int(padding * max(obj['a'], obj['b']))
    except:
        if verbosity == 1:
            print(f"object bounding box contains nan value")
        return None

    # Define crop boundaries
    x_min, x_max, y_min, y_max = _get_thumbnail_coord(max_half_size, x_center, y_center, image_data)

    if abs(x_max-x_min) < min_size or abs(y_max-y_min) < min_size:
        if verbosity == 1:
            print(f"object size is {abs(x_max-x_min)} and {abs(y_max-y_min)}. Ignored as too small")
        return None

    # Crop the image
    cropped_data = image_data[y_min:y_max, x_min:x_max]

    return cropped_data

def _gal_dist_estimate(gal_diameter: float) -> float:
    '''estimates the distance a galaxy is from earth
    estimation is based on instructions outlined here: https://www.roe.ac.uk/~wkmr/Galaxy_Redshift_webpages/Estimating%20the%20distance%20to%20a%20galaxy.htm'''
    #assume 50 000 light years as reasonable average size
    degrees = gal_diameter/3600
    distance = 50000/degrees * 60
    return distance


def _resolve_position(pixel_size: float, ori_xpos: int, ori_ypos: int, 
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

def _update_meta_data(obj_data: np.ndarray, current_meta_data: dict) -> dict:
    '''updates meta data for particular image'''
    new_meta_data = current_meta_data.copy()
    pixel_size = new_meta_data['PIXAR_A2'][0]
    pixel_width = math.sqrt(pixel_size)

    gal_width = obj_data['a']* pixel_width
    gal_height = obj_data['b']* pixel_width 
    gal_diameter = math.sqrt(gal_width*gal_width + gal_height*gal_height)
    new_meta_data['GALX'] = (gal_width, 'width of galaxy in arcseconds')
    new_meta_data['GALY'] = (gal_height, 'height of galaxy in arcseconds')
    new_meta_data['GALDIAM'] = (gal_diameter, 'diameter of galaxy in arcseconds')
    
    ori_pos_x, ori_pos_y = current_meta_data['ORIAXIS1'][0]/2, current_meta_data['ORIAXIS2'][0]/2
    x_pos_in_image, y_pos_in_image = int(obj_data['x']), int(obj_data['y'])
    ori_ra = current_meta_data['TARG_RA'][0]
    ori_dec = current_meta_data['TARG_DEC'][0]
    obj_ra, obj_dec = _resolve_position(pixel_size, ori_pos_x, ori_pos_y, 
                                x_pos_in_image, y_pos_in_image, 
                                ori_ra, ori_dec)
    new_meta_data['OBJ_RA'] = (obj_ra, 'RA of galaxy')
    new_meta_data['OBJ_DEC'] = (obj_dec, 'DEC of galaxy')
    
    gal_distance = _gal_dist_estimate(gal_diameter)
    new_meta_data['GALDIST'] = (gal_distance, 'distance to galaxy in light years')

    return new_meta_data


def extract_objects_to_file(image_data: np.ndarray, image_meta_data: dict, file_name: str, celestial_objects: np.ndarray, 
                            output_dir: str, padding: float = 1.5, min_size: int = 0,
                            verbosity: int = 0) -> None:
    '''extracts celestials objects from image_data to fits fil in directory'''
    
    total_files = 0
    for i, obj in enumerate(celestial_objects):
        cropped_data = _extract_object(obj, image_data, padding, min_size, verbosity)
      
        if type(cropped_data) != type(None):

            curr_file_name = file_name + f"object_{i + 1}.fits"
            file_address = os.path.join(output_dir, curr_file_name)

            image_meta_data = _update_meta_data(obj, image_meta_data)

            _save_to_fits(file_address, cropped_data, image_meta_data)

            print(f"Saved: {curr_file_name}")
            total_files += 1

    print(f"total files cropped: {total_files}")