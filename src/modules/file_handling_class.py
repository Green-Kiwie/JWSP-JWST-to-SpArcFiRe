import sep_helpers as sh
import position_resolving as resolve
import galaxy_naming as gn

import numpy as np
import pathlib
import math

def _get_pos_of_file(filepath: str) -> tuple[float, float]:
    meta_data = sh.get_all_fits_meta_data(filepath)

    gal_pos = Gal_Pos(meta_data)
    # print(f'object: {meta_data['OBJ_RA']}, {meta_data['OBJ_DEC']}')
    return gal_pos

class Gal_Pos:
    def __init__(self, image_meta_data: np.ndarray):
        pixel_size = image_meta_data['PIXAR_A2'][0]
        pixel_width = math.sqrt(pixel_size)
        ori_pos_x = image_meta_data['SUBSIZE1'][0]/2
        ori_pos_y = image_meta_data['SUBSIZE2'][0]/2
        img_width = image_meta_data['N_NAXIS1'][0]* pixel_width
        img_height = image_meta_data['N_NAXIS2'][0]* pixel_width 
        x_pos_in_image = image_meta_data['ORI_POSX'][0]
        y_pos_in_image = image_meta_data['ORI_POSY'][0]
        ori_ra = image_meta_data['TARG_RA'][0]
        ori_dec = image_meta_data['TARG_DEC'][0]
        
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

        # store approximate angular size of the object (in arcseconds)
        try:
            galx = image_meta_data.get('N_GALX', (None,))[0]
            galy = image_meta_data.get('N_GALY', (None,))[0]
            if galx is not None and galy is not None:
                self._approx_size = (float(galx) + float(galy)) / 2.0
            else:
                self._approx_size = None
        except Exception:
            self._approx_size = None

    def contains(self, coord: tuple[float, float]) -> bool:
        '''checks if a particiular RA and DEC is within the object bounds'''
        ra, dec = coord

        if self._topleft[0] <= self._bottomright[0]:
            within_ra = self._topleft[0] >= ra >= self._bottomright[0]
        else:  
            within_ra = ra <= self._topleft[0] or ra >= self._bottomright[0]

        within_dec = (dec >= self._topleft[1]) and (dec <= self._bottomright[1])
        
        return bool(within_dec and within_ra)

    def get_pos(self) -> tuple[float, float]:
        '''returns the ra and dec of a galaxy'''
        return self._position

    def approx_size(self) -> float | None:
        '''Return approximate angular size (arcseconds) if available.'''
        return self._approx_size

class Thumbnail_Handling:
    def __init__(self, saving_function: 'function', filepath: str = None):
        # Map of canonical position -> list of Gal_Pos entries (versions)
        self._records = dict()
        self._saving_function = saving_function
        if filepath != None:
            self._add_files_to_record(filepath)

    def _add_files_to_record(self, filepath: str) -> None:
        """Scan existing FITS in a directory and register them as records.

        Existing files are grouped by approximate position and size so that
        multiple versions (different filters) can be associated together.
        """
        directory = pathlib.Path(filepath)
        for fits_file in directory.iterdir():
            if str(fits_file).endswith('fits'):
                gal_pos= _get_pos_of_file(fits_file)
                key = self._find_key_for(gal_pos)
                if key is None:
                    self._records[gal_pos] = [gal_pos]
                else:
                    self._records[key].append(gal_pos)

        total = sum(len(v) for v in self._records.values())
        print(f"{total} images exists. All are added to record object")
    
    def _check_record_new(self, position: tuple[float, float]) -> float:
        '''checks if coords exists in the records'''
        for key in self._records.keys():
            if key.contains(position):
                return False
        return True

    def _find_key_for(self, gal_pos: Gal_Pos, pos_tol_arcsec: float = 1.0, size_tol_frac: float = 0.5) -> Gal_Pos | None:
        """Find an existing record key that matches gal_pos by position and size.

        Returns the matching key Gal_Pos if found, otherwise None.
        """
        pos = gal_pos.get_pos()
        size = gal_pos.approx_size()
        for key in self._records.keys():
            # position containment test
            if key.contains(pos):
                # if both sizes available, check size similarity
                key_size = key.approx_size()
                if key_size is None or size is None:
                    return key
                # allow size within size_tol_frac (fractional) or within pos_tol_arcsec
                if abs(key_size - size) <= max(size_tol_frac * key_size, pos_tol_arcsec):
                    return key
        return None

    def _add_record_ignore_duplicate(self, gal_pos: Gal_Pos) -> bool:
        '''if position does not exist, add record to self. 
        if position is added, return true. else, return false'''
        key = self._find_key_for(gal_pos)
        if key is None:
            self._records[gal_pos] = [gal_pos]
            return True
        else:
            self._records[key].append(gal_pos)
            return False
        
    def _add_record_include_duplicate(self, gal_pos: Gal_Pos) -> bool:
        '''Always keep the record, grouping by position/size.'''
        key = self._find_key_for(gal_pos)
        if key is None:
            self._records[gal_pos] = [gal_pos]
        else:
            self._records[key].append(gal_pos)
        return True


    def save_file_ignore_duplicate(self, save_function_parameters: tuple[any], image_meta_data: np.ndarray, filename: str) -> None:
        '''if position does not exist, save file. else, still save but record as additional version'''
        gal_pos = Gal_Pos(image_meta_data)

        added = self._add_record_ignore_duplicate(gal_pos)
        # always save the file but inform if it was an existing object
        if not added:
            print(f"object {filename} corresponds to an existing object; saved as additional version")
        self._saving_function(*save_function_parameters)

    def save_file_include_duplicate(self, save_function_parameters: tuple[any], image_meta_data: np.ndarray, filename: str) -> None:
        '''Save a file and record it grouped by position/size so multiple versions are retained.'''
        gal_pos = Gal_Pos(image_meta_data)
        self._add_record_include_duplicate(gal_pos)
        self._saving_function(*save_function_parameters)

    def get_total_files(self) -> int:
        '''returns the total number of files saved'''
        return sum(len(v) for v in self._records.values())

    def get_galaxy_filename(self, gal_pos: Gal_Pos, output_dir: str) -> tuple[str, int]:
        """Generate a coordinate-based filename for a galaxy."""
        ra, dec = gal_pos.get_pos()
        filename, version = gn.generate_galaxy_filename(ra, dec, output_dir, return_full_path=False)
        return filename, version

    def get_galaxy_filepath(self, gal_pos: Gal_Pos, output_dir: str) -> tuple[str, int]:
        """Generate a full coordinate-based filepath for a galaxy.
        
        Parameters
        ----------
        gal_pos : Gal_Pos
            Galaxy position object with RA/DEC coordinates.
        output_dir : str
            Directory where galaxy files are stored.
        
        Returns
        -------
        tuple[str, int]
            (full_filepath, version_number) - complete path ready for saving
        """
        ra, dec = gal_pos.get_pos()
        filepath, version = gn.generate_galaxy_filename(ra, dec, output_dir, return_full_path=True)
        return filepath, version


