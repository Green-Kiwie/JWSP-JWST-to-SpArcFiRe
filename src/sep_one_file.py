''' 
This script runs SEP on one fits file.

Takes a fits file as input, finds celestial objects, and saves cropped images (thumbprints) to files.
'''

import sys

sys.path[0] += ('/modules')
print(sys.path[0])

import sep_helpers as sh
import pathlib
import numpy as np
import os
import shutil

_MIN_SIZE = 15 #minimum size of object before object is cropped
_MAX_SIZE = 100

def _get_file_name() -> str:
    '''ask and validates file name from user'''
    while True:
        file_name = input("filename: ")
        if file_name.endswith('.fits') and len(file_name) > 5:
            if _check_file_exists(file_name):
                break

        print("Invalid file, name. Please give another file name.")

    return file_name

def _check_file_exists(filename: str) -> bool:
    '''validates if filename exist'''
    return pathlib.Path(filename).exists()

def _print_image(image_data: 'input_for_function_list', print_function: 'function', image_type: str|None = None) -> None:
    '''ask if user would like to print image and prints image'''
    print_normal_flag = input(f"print {image_type} image (Y/N)? (default no): ")
    if print_normal_flag == 'Y':
        print_function(*image_data)
    else:
        return None
        

def _create_directory(name: str) -> None:
    '''creates directory of given name'''

    if pathlib.Path(name).exists():
        print("path currently exists, deleting file contents.")
        shutil.rmtree(name)
        
    try:
        os.makedirs(name)
        print(f"Directory '{name}' created successfully.")
    except FileExistsError:
        print(f"Directory '{name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def _get_required_verbosity() -> int:
    '''gets the required verbosity from the user'''
    while True:
        try:
            verbosity_str = input("Verbosity of thumbprint extraction? (0 for minimum, 1 for maximum, default 0)").strip()
            if verbosity_str == '':
                return 0
            verbosity = int(verbosity_str)
            if verbosity != 1 and verbosity != 0:
                raise ValueError
            break
        except:
            print("invalid verbosity, please try again.")

    return verbosity
    
def _get_thumbprints(output_filepath: str, image_data: np.ndarray, image_meta_data: dict, celestial_objects: np.ndarray) -> None:
    '''splits image image into files'''
    verbosity = _get_required_verbosity()
    directory = output_filepath
    _create_directory(directory)
    sh.extract_objects_to_file(image_data, image_meta_data, '', celestial_objects, directory, min_size = _MIN_SIZE, max_size = _MAX_SIZE, verbosity = verbosity)
    
def run():
    file_name = _get_file_name()

    image_data = sh.get_main_fits_data(file_name)
    

    background_rms = sh.get_image_background(image_data)
    backgroundless_data = sh.subtract_bkg(image_data)
    celestial_objects = sh.extract_objects(backgroundless_data, background_rms)
    scaled_image = sh.scale_fits_data(backgroundless_data)

    image_meta_data = sh.get_relevant_fits_meta_data(file_name)

    _print_image([scaled_image], sh.show_fits_image, 'scaled')
    _print_image([background_rms], sh.show_fits_image, 'background')
    _print_image([scaled_image, celestial_objects], sh.plot_object_mask_on_galaxy, 'mask')

    pure_name = pathlib.Path(file_name).name
    output_filepath = 'output/' + pure_name[:-5]
    _get_thumbprints(output_filepath, scaled_image, image_meta_data, celestial_objects,)
    
if __name__ == "__main__":
    run()