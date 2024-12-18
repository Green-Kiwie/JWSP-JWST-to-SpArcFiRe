import sep_helpers as sh
import position_resolving as resolve

import numpy as np
import pathlib
import math

def _get_pos_of_file(filepath: str) -> tuple[float, float]:
    meta_data = sh.get_all_fits_meta_data(filepath)
    obj_ra = meta_data['OBJ_RA']
    obj_dec = meta_data['OBJ_DEC']

    gal_pos = Gal_Pos(meta_data)
    # print(f'object: {obj_ra}, {obj_dec}')
    return gal_pos

class Gal_Pos:
    def __init__(self, image_meta_data: np.ndarray):
        pixel_size = image_meta_data['PIXAR_A2'][0]
        pixel_width = math.sqrt(pixel_size)
        ori_pos_x = image_meta_data['ORIAXIS1'][0]/2
        ori_pos_y = image_meta_data['ORIAXIS2'][0]/2
        img_width = image_meta_data['NAXIS1'][0]* pixel_width
        img_height = image_meta_data['NAXIS2'][0]* pixel_width 
        x_pos_in_image = image_meta_data['ORI_POSX'][0]
        y_pos_in_image = image_meta_data['ORI_POSY'][0]
        ori_ra = image_meta_data['ORI_RA'][0]
        ori_dec = image_meta_data['ORI_DEC'][0]
        
        topleft_x = x_pos_in_image - img_width/2
        topleft_y = y_pos_in_image - img_height/2
        self._topleft = resolve.resolve_position(pixel_size, ori_pos_x, ori_pos_y, 
                                topleft_x, topleft_y, 
                                ori_ra, ori_dec)

        bottomright_x = x_pos_in_image + img_width/2
        bottomright_y = y_pos_in_image + img_height/2
        self._bottomright = resolve.resolve_position(pixel_size, ori_pos_x, ori_pos_y, 
                                bottomright_x, bottomright_y, 
                                ori_ra, ori_dec)

        self._position = resolve.resolve_position(pixel_size, ori_pos_x, ori_pos_y, 
                                x_pos_in_image, y_pos_in_image, 
                                ori_ra, ori_dec)

    def contains(self, coord: tuple[float, float]) -> bool:
        '''checks if a particiular RA and DEC is within the object bounds'''
        ra, dec = coord

        if self._topleft[0] <= self._bottomright[0]:
            within_ra = self._topleft[0] >= ra >= self._bottomright[0]
        else:  
            within_ra = ra <= self._topleft[0] or ra >= self._bottomright[0]

        within_dec = (dec >= self._topleft[1]) and (dec <= self._bottomright[1])
        
        if (within_dec and within_ra) == True:
            return True
        else:
            return False

    def get_pos(self) -> tuple[float, float]:
        '''returns the ra and dec of a galaxy'''
        return self._position

class Thumbnail_Handling:
    def __init__(self, saving_function: 'function', filepath: str = None):
        self._coords = set()
        self._saving_function = saving_function
        if filepath != None:
            self._add_files_to_record(filepath)

    def _add_files_to_record(self, filepath: str) -> None:
        '''add all obj RA and DEC to coords set'''
        directory = pathlib.Path(filepath)
        for fits_file in directory.iterdir():
            if str(fits_file).endswith('fits'):
                gal_pos= _get_pos_of_file(fits_file)
                self._coords.add(gal_pos)

        print(f"{len(self._coords)} images exists. All are added to record object for duplicate prevention")
    
    def _check_record_new(self, position: tuple[float, float]) -> float:
        '''checks if coords exists in the set'''
        # print(f"new position: {position}")
        for gal_pos in self._coords:
            if gal_pos.contains(position):
                return False
        
        return True

    def _add_record(self, gal_pos: Gal_Pos) -> bool:
        '''if position does not exist, add record to self. 
        if position is added, return true. else, return false'''
        position = gal_pos.get_pos()
        if self._check_record_new(position) == True:
            self._coords.add(gal_pos)
            return True
        else:
            return False


    def save_file(self, save_function_parameters: tuple[any], image_meta_data: np.ndarray, filename: str) -> None:
        '''if position does not exist, save file. else, ignore file'''
        gal_pos = Gal_Pos(image_meta_data)

        if self._add_record(gal_pos) == False:
            print(f"object {filename} not saved as position already exists")
            return None
        
        self._saving_function(*save_function_parameters)

    def get_total_files(self) -> int:
        '''returns the total number of files saved'''
        return len(self._coords)


