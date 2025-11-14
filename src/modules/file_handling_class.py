import sep_helpers as sh
import position_resolving as resolve
import galaxy_naming as gn

import numpy as np
import pathlib
import math
# import group_discovery as gd  # Not used in current code path

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
    def __init__(self, saving_function: 'function', filepath: str = None, discovery_mode: bool = False):
        # Map of canonical position -> list of Gal_Pos entries (versions)
        self._records = dict()
        self._saving_function = saving_function
        self._discovery_mode = discovery_mode
        if filepath != None:
            self._add_files_to_record(filepath)
        if discovery_mode:
            pass  # self._discovery = gd.GroupDiscovery()  # Not used in current code path
        
        # For discovery mode: store minimal metadata for each discovered object
        self._discovery_objects = []

    def add_object_discovery_only(self, ra: float, dec: float, height: int, width: int, 
                                   source_filename: str) -> None:
        '''Add object metadata without geometric checks (discovery mode)'''
        if not self._discovery_mode:
            raise RuntimeError("Not in discovery mode")
        
        # Store minimal metadata
        obj_data = {
            'ra': ra,
            'dec': dec,
            'height': height,
            'width': width,
            'source': source_filename
        }
        self._discovery_objects.append(obj_data)

    def get_discovery_summary(self) -> dict:
        '''Get summary without geometry calculations'''
        return {
            'total_members': len(self._discovery_objects),
            'group_count': len(set(obj['source'] for obj in self._discovery_objects))
        }
    
    def save_discovery_metadata(self, filepath: str) -> None:
        '''Save discovery metadata to JSON'''
        import json
        data = {
            'total_objects': len(self._discovery_objects),
            'objects': self._discovery_objects
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_discovery_report(self, detailed: bool = False, detail_limit: int = 20) -> None:
        '''Print discovery report.'''
        if not self._discovery_mode:
            raise RuntimeError("print_discovery_report() requires discovery_mode=True")
        
        self._discovery.print_summary()
        if detailed:
            self._discovery.print_group_details(limit=detail_limit)

    def _add_files_to_record(self, filepath: str) -> None:
        '''Scan existing FITS in a directory and register them as records.

        Existing files are grouped by approximate position and size so that
        multiple versions (different filters) can be associated together.
        '''
        directory = pathlib.Path(filepath)
        for fits_file in directory.iterdir():
            if str(fits_file).endswith('fits'):
                gal_pos= _get_pos_of_file(fits_file)
                key = self._find_key_for(gal_pos)
                if key is None:
                    self._records[gal_pos] = {'members': [gal_pos], 'crops': [], 'max_dims': (0,0)}
                else:
                    self._records[key]['members'].append(gal_pos)

        total = sum(len(v) for v in self._records.values())
        print(f"{total} images exists. All are added to record object")
    
    def _check_record_new(self, position: tuple[float, float]) -> float:
        '''checks if coords exists in the records'''
        for key in self._records.keys():
            if key.contains(position):
                return False
        return True

    def _find_key_for(self, gal_pos: Gal_Pos, pos_tol_arcsec: float = 1.0, size_tol_frac: float = 0.5) -> Gal_Pos | None:
        '''Find an existing record key that matches gal_pos by position and size.

        Returns the matching key Gal_Pos if found, otherwise None.
        '''
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

    def _add_crop_to_group(self, gal_pos: Gal_Pos, crop_data: np.ndarray, metadata: dict, filename: str, waveband: str = None) -> None:
        '''adds cropped object to its group, creating if necessary'''
        key = self._find_key_for(gal_pos)
        if key is None:
            self._records[gal_pos] = {'members': [gal_pos], 'crops': [], 'max_dims': (0,0), 'wavebands': []}
            key = gal_pos
        else:
            self._records[key]['members'].append(gal_pos)
        
        self._records[key]['crops'].append((crop_data, metadata, filename))
        self._records[key]['wavebands'].append(waveband)
        curHeight, curWidth = crop_data.shape
        maxHeight, maxWidth = self._records[key]['max_dims']
        self._records[key]['max_dims'] = (max(curHeight, maxHeight), max(curWidth, maxWidth))

    def _standardize_group(self, key: Gal_Pos) -> list:
        '''crop all images in a group to the same size (largest + 10% padding)'''
        group = self._records[key]
        maxHeight, maxWidth = group['max_dims']
        standardized_crops = []

        for version_num, ((crop_data, metadata, filename), waveband) in enumerate(
            zip(group['crops'], group['wavebands']), start=1
        ):
            std_crop = sh.standardize_crop_dimensions(crop_data, maxHeight, maxWidth, padding_fraction=0.1)
            standardized_crops.append((std_crop, metadata, version_num, waveband))

        return standardized_crops

    def _finalize_and_save(self, output_dir: str) -> int:
        '''standardize all groups and save to files with new naming convention'''
        total_saved = 0

        for key in self._records.keys():
            standardized_crops = self._standardize_group(key)

            # prepare to save to disk
            for std_crop, metadata, version_num, waveband in standardized_crops:
                ra, dec = key.get_pos()
                # Round coordinates to 6 decimal places
                ra_rounded = gn.round_coordinate(ra, decimals=6)
                dec_rounded = gn.round_coordinate(dec, decimals=6)
                
                # Use default waveband if not specified
                waveband_str = waveband if waveband else "unknown"
                
                # Generate new filename format: "RA_DEC_waveband_vN.fits"
                final_filename = f"{ra_rounded:.6f}_{dec_rounded:.6f}_{waveband_str}_v{version_num}.fits"
                filepath = pathlib.Path(output_dir) / final_filename

                self._saving_function(str(filepath), std_crop, metadata)
                total_saved += 1
                print(f"Saved: {final_filename} (v{version_num})")
        return total_saved

    def get_total_files(self) -> int:
        '''returns the total number of files saved'''
        return sum(len(v) for v in self._records.values())

    def get_galaxy_filename(self, gal_pos: Gal_Pos, output_dir: str) -> tuple[str, int]:
        '''Generate a coordinate-based filename for a galaxy.'''
        ra, dec = gal_pos.get_pos()
        filename, version = gn.generate_galaxy_filename(ra, dec, output_dir, return_full_path=False)
        return filename, version

    def get_galaxy_filepath(self, gal_pos: Gal_Pos, output_dir: str) -> tuple[str, int]:
        '''Generate a full coordinate-based filepath for a galaxy.'''
        ra, dec = gal_pos.get_pos()
        filepath, version = gn.generate_galaxy_filename(ra, dec, output_dir, return_full_path=True)
        return filepath, version

    def get_memory_stats(self) -> dict:
        import memory_usage as mt
        mt.get_records_class_memory(self)
    
    def print_memory_report(self, file_count: int = None, processed_files: int = None) -> None:
        import memory_usage as mt
        mt.print_memory_report(self, file_count, processed_files)