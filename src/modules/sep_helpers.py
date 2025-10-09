import numpy as np
import sep_pjw as sep
import matplotlib.pyplot as plt
import os
import math

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from matplotlib.patches import Ellipse

import file_handling_class as fhc
import position_resolving as resolve

def _fill_nan(image_data: np.ndarray) -> np.ndarray:
    '''uses the astropy function convolute to fill in nan values'''
    gauss_kernal = Gaussian2DKernel(1)
    convolved_data = interpolate_replace_nans(image_data, gauss_kernal)
    
    return convolved_data

def get_main_fits_data(hdul: fits.HDUList) -> np.ndarray:
    '''reads 'sci' data from in-memory HDUList object'''
    try:
        image_data = hdul['SCI'].data
    except KeyError:
        image_data = hdul[0].data
    
    image_data_no_nan = _fill_nan(image_data)
    byte_swapped_data = image_data_no_nan.byteswap().view(image_data_no_nan.dtype.newbyteorder('='))
    return byte_swapped_data

def get_all_fits_meta_data(hdul: fits.HDUList) -> dict[tuple[any, any]]:
    '''gets all meta data from an HDUList object.'''
    try:
        primary_data = hdul['PRIMARY'].header
        primary_data_comments = hdul['PRIMARY'].header.comments
    except KeyError:
        primary_data = hdul[0].header
        primary_data_comments = hdul[0].header.comments

    output = {key: (primary_data[key], primary_data_comments[key]) for key in primary_data.keys()}
    return output

def get_relevant_fits_meta_data(hdul: fits.HDUList) -> dict:
    '''gets relevant meta data from an HDUList object for later calculations'''
    relevant_meta_data = {}
    
    try:
        primary_header = hdul['PRIMARY'].header
    except KeyError:
        primary_header = hdul[0].header

    for key in primary_header.keys():
        relevant_meta_data[key] = (primary_header[key], primary_header.comments[key])

    try:
        sci_header = hdul['SCI'].header
        for key in sci_header.keys():
            relevant_meta_data[key] = (sci_header[key], sci_header.comments[key])
    except KeyError:
        # If 'SCI' extension doesn't exist, we've likely already read all necessary headers
        pass

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

def save_to_fits(filepath: str, image_data: np.ndarray, image_meta_data: dict) -> None:
    '''saves a file to a non multi extension fits file'''
    hdu = fits.PrimaryHDU(data=image_data)
    for key in image_meta_data:
        try:
            hdu.header[key] = image_meta_data[key]
        except:
            # print("unable to save pair in thumbnail meta_data: key:", key, ", value: ", image_data_header[key])
            pass
        # hdu.header[key] = image_meta_data[key]
    
    hdu.writeto(filepath, overwrite=True)
    print(f"Saved: {filepath}")
    

def _extract_object(obj: np.void, image_data: np.ndarray, padding: float, min_size: int, max_size: int, verbosity: int) -> np.ndarray|None:
    '''crop out the object from the main data and returns the cropped image. returns none if crop too small or too large'''
    x_center, y_center = int(obj['x']), int(obj['y'])
   
    try: 
        max_half_size = int(padding * max(obj['a'], obj['b']))
    except:
        if verbosity == 1:
            print(f"object bounding box contains nan value")
        return None

    x_min, x_max, y_min, y_max = _get_thumbnail_coord(max_half_size, x_center, y_center, image_data)

    if abs(x_max-x_min) < min_size or abs(y_max-y_min) < min_size:
        if verbosity == 1:
            print(f"object size is {abs(x_max-x_min)} and {abs(y_max-y_min)}. Ignored as too small")
        return None
    
    if abs(x_max-x_min) > max_size or abs(y_max-y_min) > max_size:
        if verbosity == 1:
            print(f"object size is {abs(x_max-x_min)} and {abs(y_max-y_min)}. Ignored as too large")
        return None

    cropped_data = image_data[y_min:y_max, x_min:x_max]

    return cropped_data

def _gal_dist_estimate(gal_diameter: float) -> float:
    '''estimates the distance a galaxy is from earth
    estimation is based on instructions outlined here: https://www.roe.ac.uk/~wkmr/Galaxy_Redshift_webpages/Estimating%20the%20distance%20to%20a%20galaxy.htm'''
    #assume 50 000 light years as reasonable average size
    degrees = gal_diameter/3600
    distance = 50000/degrees * 60
    return distance


def _get_gal_size(obj_data: np.ndarray, meta_data: dict) -> tuple[float, float]:
    '''returns the galaxy height and width'''
    pixel_size = meta_data['PIXAR_A2'][0]
    pixel_width = math.sqrt(pixel_size)

    gal_width = obj_data['a']* pixel_width
    gal_height = obj_data['b']* pixel_width 
    return gal_width, gal_height

def _get_gal_position(obj_data: np.ndarray, meta_data: dict) -> tuple[float, float]:
    '''returns the RA and DEC of the extracted galaxy'''
    ori_pos_x, ori_pos_y = meta_data['SUBSIZE1'][0]/2, meta_data['SUBSIZE2'][0]/2
    x_pos_in_image, y_pos_in_image = int(obj_data['x']), int(obj_data['y'])
    ori_ra = meta_data['TARG_RA'][0]
    ori_dec = meta_data['TARG_DEC'][0]
    pixel_size = meta_data['PIXAR_A2'][0]
    obj_ra, obj_dec = resolve.resolve_position(pixel_size, ori_pos_x, ori_pos_y, 
                                x_pos_in_image, y_pos_in_image, 
                                ori_ra, ori_dec)
    return obj_ra, obj_dec

def _update_meta_data(obj_data: np.ndarray, image_data: np.ndarray, current_meta_data: dict) -> dict:
    '''updates meta data for particular image'''
    new_meta_data = current_meta_data.copy()

    gal_width,  gal_height = _get_gal_size(obj_data, new_meta_data)

    gal_pix_width = image_data.shape[1]
    gal_pix_height = image_data.shape[0]
    new_meta_data['N_NAXIS1'] = (gal_pix_width, 'width of thumbnail')
    new_meta_data['N_NAXIS2'] = (gal_pix_height, 'height of thumbnail')

    x_center, y_center = int(obj_data['x']), int(obj_data['y'])
    new_meta_data['ORI_POSX'] = (x_center, 'original pixel x position of thumbnail')
    new_meta_data['ORI_POSY'] = (y_center, 'original pixel y position of thumbnail')


    gal_diameter = math.sqrt(gal_width*gal_width + gal_height*gal_height)
    new_meta_data['N_GALX'] = (gal_width, 'width of galaxy in arcseconds')
    new_meta_data['N_GALY'] = (gal_height, 'height of galaxy in arcseconds')
    new_meta_data['N_GALDIA'] = (gal_diameter, 'diameter of galaxy in arcseconds')
    
    obj_ra, obj_dec = _get_gal_position(obj_data, new_meta_data)
    new_meta_data['OBJ_RA'] = (obj_ra, 'RA of galaxy')
    new_meta_data['OBJ_DEC'] = (obj_dec, 'DEC of galaxy')
    
    gal_distance = _gal_dist_estimate(gal_diameter)
    new_meta_data['GALDIST'] = (gal_distance, 'distance to galaxy in light years')

    return new_meta_data


def extract_objects_to_file(image_data: np.ndarray, image_meta_data: dict, file_name: str, celestial_objects: np.ndarray, 
                            output_dir: str, padding: float = 1.5, min_size: int = 0, max_size: int = 100,
                            verbosity: int = 0, records_class: fhc.Thumbnail_Handling = None) -> None:
    '''extracts celestials objects from image_data to fits fil in directory'''
    
    total_files = 0
    if records_class == None:
        records_class = fhc.Thumbnail_Handling(save_to_fits, output_dir)
    
    for i, obj in enumerate(celestial_objects):
        cropped_data = _extract_object(obj, image_data, padding, min_size, max_size, verbosity)
      
        if type(cropped_data) != type(None):

            base_filename = file_name + f"object_{i + 1}.fits"
            
            # This is the fix: Remove any leading slashes to ensure a relative path
            curr_file_name = base_filename.lstrip('/\\')

            # Now the path will be joined correctly
            file_address = os.path.join(output_dir, curr_file_name)

            image_meta_data = _update_meta_data(obj, cropped_data, image_meta_data)
            records_class.save_file_include_duplicate((file_address, cropped_data, image_meta_data), image_meta_data, curr_file_name)

    print(f"total files cropped: {total_files}")