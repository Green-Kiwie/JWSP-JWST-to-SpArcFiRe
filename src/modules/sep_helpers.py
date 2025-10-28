import numpy as np
from pathlib import Path
import os
import math

from skimage.transform import resize
import matplotlib.pyplot as plt

import sep_pjw as sep

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from matplotlib.patches import Ellipse

import file_handling_class as fhc
import position_resolving as resolve

import warnings
from scipy import ndimage

def _fill_nan(image_data: np.ndarray) -> np.ndarray:
    '''Replace NaN values in the image using fast median filtering.
    Optimized for speed on large FITS images with moderate NaN fractions.'''
    
    if not np.any(np.isnan(image_data)):
        return image_data
    
    image_copy = image_data.copy()
    nan_mask = np.isnan(image_copy)
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        # Apply filter only to NaN regions
        if np.any(nan_mask):
            dilated_mask = ndimage.binary_dilation(nan_mask, iterations=2)
            median_filtered = ndimage.median_filter(image_copy, size=5)
            
            # Replace NaNs with the filtered values
            image_copy[nan_mask] = median_filtered[nan_mask]
        
        if np.any(np.isnan(image_copy)):
            mean_val = np.nanmean(image_copy)
            if not np.isnan(mean_val):
                image_copy[np.isnan(image_copy)] = mean_val
            else:
                image_copy[np.isnan(image_copy)] = 0.0
    
    return image_copy

def get_main_fits_data(filepath: str) -> np.ndarray:
    '''Loads main image data from a FITS file.
    Example input: "path/to/file.fits"
    Example output: numpy ndarray of image data'''

    hdul = fits.open(filepath)

    try:
        image_data = hdul['SCI'].data
    except:
        image_data = hdul[0].data

    image_data_no_nan = _fill_nan(image_data)
    byte_swapped_data = image_data_no_nan.byteswap().view(image_data_no_nan.dtype.newbyteorder('='))
    hdul.close()
    return byte_swapped_data

def get_all_fits_meta_data(filepath: str) -> dict[tuple[any, any]]:
    '''Get all metadata from a FITS file's PRIMARY header.
    Example input: "path/to/file.fits"
    Example output: dict of {keyword: (value, comment)}'''

    hdul = fits.open(filepath)

    try:
        primary_data = hdul['PRIMARY'].header
        primary_data_comments = hdul['PRIMARY'].header.comments
    except:
        primary_data = hdul[0].header
        primary_data_comments = hdul[0].header.comments

    hdul.close()

    output = dict()
    for key in primary_data.keys():
        output[key] = (primary_data[key], primary_data_comments[key])

    return output


def get_relevant_fits_meta_data(filepath: str) -> dict:
    '''Store all relevant metadata from a FITS file's headers. Relevancy is determined by comments in the headers, with callers specifying which keywords they need.
    Example input: "path/to/file.fits"
    Example output: dict of {keyword: (value, comment)}
    '''

    hdul = fits.open(filepath)

    relevant_meta_data = dict()

    # check primary header
    try:
        primary_data = hdul['PRIMARY'].header
    except KeyError:
        primary_data = hdul[0].header

    # collect relevant metadata from primary header
    description_data = primary_data.comments
    for key in primary_data.keys():
        relevant_meta_data[key] = (primary_data[key], description_data[key])

    # check SCI header
    try:
        image_data_header = hdul['SCI'].header
    except KeyError:
        image_data_header = hdul[0].header

    # collect relevant metadata from SCI header
    description_data2 = image_data_header.comments
    for key in image_data_header.keys():
        relevant_meta_data[key] = (image_data_header[key], description_data2[key])

    hdul.close()
    return relevant_meta_data
    
def scale_fits_data(image_data: np.ndarray) -> np.ndarray:
    '''Applies asinh scaling to image pixel values for visualization.
    Example input: numpy ndarray of image data
    Example output: numpy ndarray of scaled image data'''

    arcsin_data = np.arcsinh(image_data)
    return arcsin_data

def save_fits_as_png(image_data: np.ndarray, save_directory: Path, png_dimension: int = 512) -> None:
    """scales image and save as a png"""
    scaled_data = scale_fits_data(image_data)
    scaled_data = np.nan_to_num(scaled_data, nan=0.0, posinf=0.0, neginf=0.0)
    resized_data = resize(scaled_data, (png_dimension, png_dimension), order=0, preserve_range=True, anti_aliasing=False)
    plt.imsave(save_directory, resized_data, cmap='gray', origin='lower')

def show_fits_image(image_data: np.ndarray, title = 'FITS Image') -> None:
    '''Displays a 2D grayscale image of a FITS image.
    Example input: numpy ndarray of image data, optional title to show above plot
    Example output: None'''

    plt.imshow(image_data, cmap='gray')
    plt.colorbar()
    plt.title(title)
    plt.show()

def get_image_background(image_data: np.ndarray, show = False) -> np.ndarray:
    '''Get image background using SEP's background estimator.
    Example input: numpy ndarray of image data, optional boolean to show background
    Example output: numpy ndarray of background data'''

    background_image = sep.Background(image_data)
    background_image_arr = background_image.back()
    if show:
        show_fits_image(background_image_arr, 'background data')
    return background_image_arr

def get_bkg_rms(background_data: np.ndarray):
    '''Get the RMS value of the background image.
    Example input: numpy ndarray of background data
    Example output: float RMS value'''

    print(type(background_data.rms()))
    return background_data.rms()

def subtract_bkg(image_data: np.ndarray) -> np.ndarray:
    '''Subtract background from image data using SEP.'''

    background_data = get_image_background(image_data)
    backgroundless_data = image_data - background_data
    return backgroundless_data

def extract_objects(clean_image_data: np.ndarray, background_rms: float) -> np.ndarray:
    '''Get objects in the cropped image using SEP.
    Example input: numpy ndarray of background-subtracted image data, float RMS of background
    Example output: numpy structured array of detected objects'''

    objects = sep.extract(clean_image_data, 1.5, err=background_rms)
    print(f"{len(objects)} objects found.")
    return objects

def plot_object_mask_on_galaxy(scaled_image_data: np.ndarray, celestial_objects: np.ndarray) -> None:
    '''Display image with ellipses overlaid for detected objects.'''

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
    '''Compute a bounding box (x_min, x_max, y_min, y_max) for a thumbnail where box is clipped to image boundaries.
    Parameters:
        - max_half_size: int, half-size in pixels of the thumbnail (radius from center)
        - x_coord, y_coord: int, center coordinates in image pixel space
        - image_data: numpy ndarray, reference image for boundary checks
    Output:
        - tuple of (x_min, x_max, y_min, y_max) coordinates within the image'''
    
    x_min = max(0, x_coord - max_half_size)                    
    x_max = min(image_data.shape[1], x_coord + max_half_size)   
    y_min = max(0, y_coord - max_half_size)                     
    y_max = min(image_data.shape[0], y_coord + max_half_size)
    return x_min, x_max, y_min, y_max

def save_to_fits(filepath: str, image_data: np.ndarray, image_meta_data: dict) -> None:
    '''Save image data and metadata to a FITS file.
    Parameters:
        - filepath: str, destination file path
        - image_data: numpy ndarray, 2D array to save as the primary HDU
        - image_meta_data: dict, mapping of header keywords to (value, comment) or value'''

    hdu = fits.PrimaryHDU(data=image_data)
    for key in image_meta_data:
        try:
            hdu.header[key] = image_meta_data[key]
        except:
            pass
    
    hdu.writeto(filepath, overwrite=True)
    #print(f"Saved: {filepath}")
    

def _extract_object(obj: np.void, image_data: np.ndarray, padding: float, min_size: int, max_size: int, verbosity: int) -> np.ndarray|None:
    '''Crop a thumbnail around a detected object and enforce size limits.
    Parameters:
        - obj: np.void, single row from SEP detection table (must contain 'x','y','a','b')
        - image_data: np.ndarray, full image array to crop from
        - padding: float, multiplier applied to max(a,b) to determine thumbnail half-size
        - min_size: int, minimum allowed side length in pixels; smaller crops return None
        - max_size: int, maximum allowed side length in pixels; larger crops return None
        - verbosity: int, if 1, prints brief diagnostic messages when objects are rejected
    Returns:
        - np.ndarray or None, cropped thumbnail array on success, or None if the crop is invalid'''
    
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
    '''Estimate the APPROXIMATE distance to a galaxy from its angular diameter. Code assumes a typical physical diameter (~50,000 light years).
    Parameters:
        - gal_diameter: float, angular diameter in arcseconds
    Returns:
        - float, Estimated distance in light years.'''

    # assume 50 000 light years as reasonable average size
    degrees = gal_diameter/3600
    distance = 50000/degrees * 60
    return distance


def _get_gal_size(obj_data: np.ndarray, meta_data: dict) -> tuple[float, float]:
    '''Calculate galaxy angular size (width/height) from pixel measurements.
    Parameters:
        - obj_data: np.ndarray of SEP detection row with 'a' and 'b' (semi-major/minor axes in pixels).
        - meta_data: dict of FITS metadata expected to contain 'PIXAR_A2'.
    Returns:
        - tuple of (width, height) in arcseconds.'''

    pixel_size = meta_data['PIXAR_A2'][0]
    pixel_width = math.sqrt(pixel_size)

    gal_width = obj_data['a'] * pixel_width
    gal_height = obj_data['b'] * pixel_width 
    return gal_width, gal_height

def _get_gal_position(obj_data: np.ndarray, meta_data: dict) -> tuple[float, float]:
    '''Calculate RA and DEC using 'x' and 'y' pixel coordinates from SEP detection.
    Parameters:
        - obj_data: np.ndarray of SEP detection row with 'x' and 'y' pixel coordinates.
        - meta_data: dict of FITS metadata expected to contain 'SUBSIZE1','SUBSIZE2','TARG_RA','TARG_DEC','PIXAR_A2'.
    Returns:
        - tuple of (RA, DEC) in degrees.'''

    ori_pos_x, ori_pos_y = meta_data['SUBSIZE1'][0]/2, meta_data['SUBSIZE2'][0]/2       # original image center in pixels
    x_pos_in_image, y_pos_in_image = int(obj_data['x']), int(obj_data['y'])             # object position in pixels
    ori_ra = meta_data['TARG_RA'][0]                                                    # original RA
    ori_dec = meta_data['TARG_DEC'][0]                                                  # original DEC
    pixel_size = meta_data['PIXAR_A2'][0]                                               # pixel area in arcsec^2
    obj_ra, obj_dec = resolve.resolve_position(pixel_size, ori_pos_x, ori_pos_y,
                                x_pos_in_image, y_pos_in_image,
                                ori_ra, ori_dec)
    return obj_ra, obj_dec

def _update_meta_data(obj_data: np.ndarray, image_data: np.ndarray, current_meta_data: dict, 
                      source_filename: str = None) -> dict:
    '''Update metadata for a cropped thumbnail to add thumbnail dimensions, object position, galaxy size, RA/DEC, distance, and source file.
    Parameters:
        - obj_data: np.ndarray, SEP detection row for the object.
        - image_data: np.ndarray, cropped thumbnail image array.
        - current_meta_data: dict, original FITS metadata to copy and extend.
        - source_filename: str, optional, name of the original source file.
    Returns:
        - dict, updated metadata. Keys: N_NAXIS1, N_NAXIS2, ORI_POSX, ORI_POSY, N_GALX, N_GALY, N_GALDIA, OBJ_RA, OBJ_DEC, GALDIST, and optionally ORGSRC.'''

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
    new_meta_data['GALDIST'] = (gal_distance, 'distance in light years')
    
    if source_filename is not None:
        # Use FITS-compliant 8-char keyword name with short comment
        new_meta_data['ORGSRC'] = (source_filename, 'source file')

    return new_meta_data


def extract_objects_to_file(image_data: np.ndarray, image_meta_data: dict, file_name: str, celestial_objects: np.ndarray, 
                            output_dir: str, padding: float = 1.5, min_size: int = 20, max_size: int = 100,
                            verbosity: int = 0, records_class: fhc.Thumbnail_Handling = None) -> None:
    '''Crop and save detected objects and save each as a FITS thumbnail with coordinate-based naming.
    Parameters:
        - image_data: np.ndarray, Full image array from which to crop thumbnails.
        - image_meta_data: dict, Original image metadata that will be extended for each thumbnail.
        - file_name: str, Original source filename (stored in metadata for reference).
        - celestial_objects: np.ndarray, SEP detection array with one entry per object.
        - output_dir: str, Directory in which to write thumbnail FITS files.
        - padding: float, optional, Multiplier for object size to determine thumbnail half-size, by default 1.5.
        - min_size: int, optional, Minimum allowed thumbnail side length in pixels, by default 0.
        - max_size: int, optional, Maximum allowed thumbnail side length in pixels, by default 100.
        - verbosity: int, optional, If 1, prints messages when objects are skipped, by default 0.
        - records_class: fhc.Thumbnail_Handling, optional, If provided, used to manage saving and deduplication of files. If None, a default `Thumbnail_Handling` instance is created.'''

    total_files = 0
    if records_class == None:
        records_class = fhc.Thumbnail_Handling(save_to_fits, output_dir)
    
    for i, obj in enumerate(celestial_objects):
        cropped_data = _extract_object(obj, image_data, padding, min_size, max_size, verbosity)
      
        if type(cropped_data) != type(None):
            # update metadata with source filename
            updated_meta_data = _update_meta_data(obj, cropped_data, image_meta_data, source_filename=file_name)
            
            # generate coordinate-based filename with version
            gal_pos = fhc.Gal_Pos(updated_meta_data)
            coord_filename, version = records_class.get_galaxy_filepath(gal_pos, output_dir)
            
            # save using the new naming convention and file
            records_class.save_file_include_duplicate((coord_filename, cropped_data, updated_meta_data), 
                                                      updated_meta_data, coord_filename)

            total_files = records_class.get_total_files()
            if verbosity > 0:
                print(f"Saved object {i+1} as version {version} to {os.path.basename(coord_filename)}")

    print(f"total files cropped: {total_files}")
