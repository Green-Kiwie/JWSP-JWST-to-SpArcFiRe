'''script takes a csv file, gets all mast uri from the file and downloads each file and runs sep on it'''

import sys

sys.path[0] += ('/modules')
print(sys.path[0])

import sep_helpers as sh
import mast_interactor as mi
import file_handling_class as fhc

import pathlib
import pandas as pd
import numpy as np
import shutil 
import os

_MIN_SIZE = 15 #minimum size of object before object is cropped
_MAX_SIZE = 100

def _check_file_exists(filename: str) -> bool:
    '''validates if filename exist'''
    return pathlib.Path(filename).exists()

def _get_filepath() -> str:
    '''get the filepath of the csv file from user'''
    while True:
        file_name = input("csv filename (must end with csv extension): ")
        if file_name.endswith('.csv') and _check_file_exists(file_name):
            break

        print("Invalid file, name. Please give another file name.")

    return file_name

def _get_uri_list(filepath: str) -> pd.Series:
    '''returns a list of all uris'''
    csv_data = pd.read_csv(filepath, skiprows=4)
    uri_list = csv_data['dataURI']
    return uri_list

def _get_user_filter_requirement() -> bool:
    '''ask if user wants to filter data'''
    while True:
        flag_str = input("filter uris for _i2d.fits only? (Y/N, default Y)").strip()
        if flag_str == '':
            return True

        if flag_str == 'Y':
            return True
        elif flag_str == 'N':
            return False
        
        print("invalid response, please try again.")

def _filter_i2d_only(uri_list: pd.Series) -> pd.Series:
    '''filter uris to only return uri that end with _i2d.fits'''
    filter_uri = uri_list[uri_list.fillna('').astype(str).str.endswith('_i2d.fits')]
    return filter_uri

def _filter_uris(uri_list: pd.Series) -> pd.Series:
    '''asks user if list of uri is to be filter for "_i2d.fits" only and filers if requested'''
    filter_flag = _get_user_filter_requirement()
    if filter_flag == True:
        return _filter_i2d_only(uri_list)
    else:
        return uri_list
    
def _download_uri(uri: str) -> tuple[str, bool]:
    '''downloads uri from mast database. returns true if successfully download. else, return false'''
    filepath = 'output/' + uri[17:]
    if (os.path.exists(filepath)):
        return filepath
    mi.download_file(uri, filepath)
    return filepath

def _remove_fits(download_filepath: str) -> None:
    '''removes the downloaded fits path'''
    os.remove(download_filepath)

def _create_directory(name: str) -> None:
    '''creates directory of given name. does not delete directory if it exists'''
        
    try:
        os.makedirs(name)
        print(f"Output Directory '{name}' created successfully.")
    except FileExistsError:
        print(f"Output Directory '{name}' already exists.")
    except PermissionError:
        print(f"output directory Permission denied: Unable to create '{name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def _get_thumbprints(output_dir: str, original_image_name: str, image_data: np.ndarray, 
                     image_meta_data: dict, celestial_objects: np.ndarray, records_object: fhc.Thumbnail_Handling) -> None:
    '''splits image image into files'''
    verbosity = 0
    _create_directory(output_dir)
    file_prename = original_image_name[:-5] + '_'
    sh.extract_objects_to_file(image_data, image_meta_data, file_prename, celestial_objects, 
                               output_dir, min_size = _MIN_SIZE, max_size = _MAX_SIZE, 
                               verbosity = verbosity, records_class = records_object)
    

def _run_sep(download_filepath: str, output_filepath: str, records_object: fhc.Thumbnail_Handling) -> bool:
    '''runs sep on filepath and return bool value for success'''
    image_data = sh.get_main_fits_data(download_filepath)
    

    background_rms = sh.get_image_background(image_data)
    backgroundless_data = sh.subtract_bkg(image_data)
    celestial_objects = sh.extract_objects(backgroundless_data, background_rms)
    # scaled_image = sh.scale_fits_data(backgroundless_data)

    image_meta_data = sh.get_relevant_fits_meta_data(download_filepath)

    original_image_name = pathlib.Path(download_filepath).name
    _get_thumbprints(output_filepath, original_image_name, backgroundless_data, 
                     image_meta_data, celestial_objects, records_object)

def _log_errored_download(filename_prefix: str, uri: str, error: str) -> None:
    '''writes to a file a particular file that was unable to download'''
    filepath = f"{filename_prefix}_log.txt"
    with open(filepath, 'a') as file:
        file.write(f"unable to download {uri}, {error} \n")

if __name__ == '__main__':
    filepath = _get_filepath()
    uris = _get_uri_list(filepath)
    uris = _filter_uris(uris)

    count = 0
    total = len(uris)
    output_filepath = 'output/objects_from_mast_csv/'
    _create_directory(output_filepath)

    records_class = fhc.Thumbnail_Handling(sh.save_to_fits, output_filepath)
    
    for uri in uris:
        for _ in range(5):
            try:
                download_filepath = _download_uri(uri)
                sep_run = _run_sep(download_filepath, output_filepath, records_class)
                # file_remove = _remove_fits(download_filepath)
                
                count += 1
                print(f"SEP run on {count}/{total} files")
                break
            except Exception as e:
                print(f"error: {e}")
                error = e
                pass
        else:
            _log_errored_download("output/mast_csv_error", uri, error)
            count += 1
            print("File unable to download, error logged.")
        

    print(f'sep run successfully on {len(uris)} files')


