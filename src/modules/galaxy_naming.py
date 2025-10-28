'''
Module for generating galaxy filenames based on coordinates and version tracking.
'''

import os
import pathlib
from typing import Optional


def decimal_to_dms(decimal_value: float, is_ra: bool= True) -> str:
    '''Convert decimal degrees to degrees/arcminutes/arcseconds string.
    Example is like 22h14m55s for RA or +45d22m30s for DEC. '''

    is_negative = decimal_value < 0
    value = abs(decimal_value)
    
    if is_ra:
        hours = value / 15.0
        h = int(hours)
        m = int((hours - h) * 60)
        s = ((hours - h) * 60 - m) * 60
        return f"{h:02d}h{m:02d}m{s:05.2f}s"
    else:
        d = int(value)
        m = int((value - d) * 60)
        s = ((value - d) * 60 - m) * 60
        sign = "-" if is_negative else "+"
        return f"{sign}{d:02d}d{m:02d}m{s:05.2f}s"


def coords_to_filename_prefix(ra: float, dec: float) -> str:
    '''Combines RA and DEC for a filename.'''

    ra_str = decimal_to_dms(ra, is_ra=True)
    dec_str = decimal_to_dms(dec, is_ra=False)
    return f"RA{ra_str}_DEC{dec_str}"


def get_next_version(output_dir: str, coord_prefix: str) -> int:
    '''Finds next version for a group of files with the same coordinate prefix.'''

    directory = pathlib.Path(output_dir)
    if not directory.exists():
        return 1
    
    max_version = 0
    pattern = f"{coord_prefix}_v"
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.name.startswith(pattern):
            # Extract version number from filename ("prefix_v#.fits")
            try:
                base = file_path.stem  # removes .fits
                version_str = base.split('_v')[-1]
                version = int(version_str)
                max_version = max(max_version, version)
            except (IndexError, ValueError):    # not expected format
                continue
    
    return max_version + 1


def generate_galaxy_filename(ra: float, dec: float, output_dir: str, 
                            return_full_path: bool = False) -> tuple[str, int]:
    '''Generate full filename with both coordinates + version.
    Full example: "RA22h14m55s_DEC+45d22m30s_v1.fits"'''

    coord_prefix = coords_to_filename_prefix(ra, dec)       # get coordinates
    version = get_next_version(output_dir, coord_prefix)    # get next version
    filename = f"{coord_prefix}_v{version}.fits"
    
    if return_full_path:
        return os.path.join(output_dir, filename), version
    else:
        return filename, version


def parse_galaxy_filename(filename: str) -> Optional[dict]:
    '''Parse a coordinate-based filename to extract RA and DEC IN DEGREES, and version in a dictionary.

    Example input: 
        "RA22h14m55s_DEC+45d22m30s_v1.fits"
    Example output:
        {'ra': 333.729167, 'dec': 45.375000, 'version': 1}    
    '''

    # Remove .fits extension
    if filename.endswith('.fits'):
        filename = filename[:-5]
    
    try:
        # get version
        parts = filename.split('_v')
        if len(parts) != 2:
            return None
            
        version = int(parts[1])
        coord_part = parts[0]
        
        # get RA and DEC parts
        ra_dec = coord_part.split('_DEC')
        if len(ra_dec) != 2:
            return None
            
        ra_str = ra_dec[0].replace('RA', '')
        dec_str = ra_dec[1]
        
        # get RA in degrees
        h_idx = ra_str.find('h')
        m_idx = ra_str.find('m')
        s_idx = ra_str.find('s')
        
        if h_idx < 0 or m_idx < 0 or s_idx < 0:
            return None
            
        hours = int(ra_str[:h_idx])
        minutes = int(ra_str[h_idx+1:m_idx])
        seconds = float(ra_str[m_idx+1:s_idx])
        ra = (hours + minutes/60.0 + seconds/3600.0) * 15.0 

        # get DEC in degrees
        sign = 1 if dec_str[0] == '+' else -1
        d_idx = dec_str.find('d')
        m_idx = dec_str.find('m')
        s_idx = dec_str.find('s')
        
        if d_idx < 0 or m_idx < 0 or s_idx < 0:
            return None
            
        degrees = int(dec_str[1:d_idx])
        arcmin = int(dec_str[d_idx+1:m_idx])
        arcsec = float(dec_str[m_idx+1:s_idx])
        dec = sign * (degrees + arcmin/60.0 + arcsec/3600.0)
        
        return {
            'ra': ra,
            'dec': dec,
            'version': version
        }
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    # ! FOR TESTING ONLY ! 
    test_ra = 333.728958  
    test_dec = 45.375834   
    
    print("Inputs are...")
    print(f"Input: RA={test_ra}, DEC={test_dec}")
    
    coord_prefix = coords_to_filename_prefix(test_ra, test_dec)
    print(f"Coordinate prefix: {coord_prefix}")
    
    filename, version = generate_galaxy_filename(test_ra, test_dec, ".", return_full_path=False)
    print(f"Generated filename: {filename} (version {version})")
    
    parsed = parse_galaxy_filename(filename)
    print(f"Parsed back: {parsed}")
