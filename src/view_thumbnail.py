'''this script allows to view one scaled thumbnail, 
provides relevant meta data in print'''

import sys

sys.path[0] += ('/modules')
print(sys.path[0])

import sep_helpers as sh
import pathlib

def _check_file_exists(filename: str) -> bool:
    '''validates if filename exist'''
    return pathlib.Path(filename).exists()

def _get_file_name() -> str:
    '''ask and validates file name from user'''
    while True:
        file_name = input("filename: ")
        if file_name.endswith('.fits') and len(file_name) > 5:
            if _check_file_exists(file_name):
                break

        print("Invalid file, name. Please give another file name.")

    return file_name

def _get_val_len(value: bool|str|int|float) -> int:
    '''returns the length of the value when printed'''
    val_type = type(value)
    if val_type == str:
        length = len(value)
    elif val_type == int or val_type == float:
        length = len(str(value))
    elif val_type == bool:
        if value == True:
            length = 4
        else:
            length = 5
    return length


def _get_max_len(values: list) -> int:
    '''get the max length of a data name/key'''
    max_length = 1
    for val in values:
        length = _get_val_len(val)
        
        if length > max_length:
            max_length = length

    return max_length

def _print_info(key: str, meta_data: str, comment: str, max_name_len: int, max_value_len: int) -> None:
    '''formats a single line of thumbnail info'''
    print(key, end = '')
    print(' '*(max_name_len - _get_val_len(key)), end = '')
    print(': ', end = '')
    print(meta_data, end = '')
    print(' '*(max_value_len - _get_val_len(meta_data)), end = '')
    print(': ', end = '')
    print(comment)

def _print_thumbnail_info(meta_data: 'astropy.header', meta_data_comments: 'astropy.header.comments') -> None:
    '''prints thumbnail in nice format'''
    max_name_len = _get_max_len(meta_data.keys())
    max_value_len =  _get_max_len(meta_data.values())
    for key in meta_data.keys():
        _print_info(key, meta_data[key], meta_data_comments[key], max_name_len, max_value_len)

if __name__ == '__main__':
    file_name = _get_file_name()

    meta_data, meta_data_comments = sh.get_all_fits_meta_data(file_name)
    print("Thumbnail information: ")
    _print_thumbnail_info(meta_data, meta_data_comments)

    image_data = sh.get_main_fits_data(file_name)
    sh.show_fits_image(image_data, f"Thumbnail of {file_name}")


    