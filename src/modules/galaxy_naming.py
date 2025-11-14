'''
Module for generating galaxy filenames based on coordinates and version tracking.

New naming convention (v2):
  RA_DEC_waveband_vN.fits
  
  Example: 80.485303_-69.517493_f150w_v1.fits
  
  Where:
    - RA: Right Ascension rounded to 6 decimal places
    - DEC: Declination rounded to 6 decimal places
    - waveband: Filter/waveband (e.g., f150w, f070w, clear)
    - vN: Version number for this coordinate+waveband combo
'''

import os
import pathlib
from typing import Optional


def round_coordinate(coord: float, decimals: int = 6) -> float:
    '''Round coordinate to specified decimal places.
    
    Args:
        coord: Coordinate value (RA or DEC in decimal degrees)
        decimals: Number of decimal places to round to (default: 6)
    
    Returns:
        Rounded coordinate value
    
    Examples:
        >>> round_coordinate(80.53921742149545, 6)
        80.539217
        >>> round_coordinate(-69.51885006523123, 6)
        -69.518850
    '''
    return round(coord, decimals)


def coords_to_filename_string(ra: float, dec: float, decimals: int = 6) -> str:
    '''Convert RA/DEC to filename string format.
    
    Args:
        ra: Right Ascension in decimal degrees
        dec: Declination in decimal degrees
        decimals: Decimal places to round to (default: 6)
    
    Returns:
        Coordinate string for filename (e.g., "80.485303_-69.517493")
    
    Examples:
        >>> coords_to_filename_string(80.53921742149545, -69.51885006523123)
        "80.539217_-69.518850"
    '''
    ra_rounded = round_coordinate(ra, decimals)
    dec_rounded = round_coordinate(dec, decimals)
    
    # Format with consistent number of decimals
    return f"{ra_rounded:.{decimals}f}_{dec_rounded:.{decimals}f}"


def decimal_to_dms(decimal_value: float, is_ra: bool= True) -> str:
    '''Convert decimal degrees to degrees/arcminutes/arcseconds string.
    Example is like 22h14m55s for RA or +45d22m30s for DEC. 
    
    (Legacy function - kept for backwards compatibility)
    '''

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



def get_next_version(output_dir: str, coord_string: str, waveband: str) -> int:
    '''Finds next version for a given coordinate and waveband combination.
    
    Args:
        output_dir: Output directory to search
        coord_string: Coordinate string (e.g., "80.485303_-69.517493")
        waveband: Waveband/filter name (e.g., "f150w")
    
    Returns:
        Next available version number (starting at 1)
    
    Examples:
        >>> get_next_version("output/", "80.485303_-69.517493", "f150w")
        1  # First time seeing this combo
    '''

    directory = pathlib.Path(output_dir)
    if not directory.exists():
        return 1
    
    max_version = 0
    pattern = f"{coord_string}_{waveband}_v"
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.name.startswith(pattern):
            # Extract version number from filename ("coord_waveband_v#.fits")
            try:
                base = file_path.stem  # removes .fits
                version_str = base.split('_v')[-1]
                version = int(version_str)
                max_version = max(max_version, version)
            except (IndexError, ValueError):    # not expected format
                continue
    
    return max_version + 1


def generate_galaxy_filename_v2(ra: float, dec: float, waveband: str, output_dir: str, 
                                 return_full_path: bool = False) -> tuple[str, int]:
    '''Generate galaxy filename using new convention: RA_DEC_waveband_vN.fits
    
    Args:
        ra: Right Ascension in decimal degrees
        dec: Declination in decimal degrees
        waveband: Filter/waveband name (e.g., "f150w", "clear")
        output_dir: Output directory (used to determine next version number)
        return_full_path: If True, return full path; if False, return filename only
    
    Returns:
        Tuple of (filename_or_path, version_number)
    
    Examples:
        >>> generate_galaxy_filename_v2(80.485303, -69.517493, "f150w", "output/")
        ("80.485303_-69.517493_f150w_v1.fits", 1)
    '''

    coord_string = coords_to_filename_string(ra, dec, decimals=6)
    version = get_next_version(output_dir, coord_string, waveband)
    filename = f"{coord_string}_{waveband}_v{version}.fits"
    
    if return_full_path:
        return os.path.join(output_dir, filename), version
    else:
        return filename, version


def parse_galaxy_filename_v2(filename: str) -> Optional[dict]:
    '''Parse new-style coordinate-based filename to extract coordinates, waveband, and version.

    Args:
        filename: Filename to parse (e.g., "80.485303_-69.517493_f150w_v1.fits")

    Returns:
        Dictionary with keys: {'ra', 'dec', 'waveband', 'version'}, or None if parsing fails

    Examples:
        >>> parse_galaxy_filename_v2("80.485303_-69.517493_f150w_v1.fits")
        {'ra': 80.485303, 'dec': -69.517493, 'waveband': 'f150w', 'version': 1}
    '''

    # Remove .fits extension
    if filename.endswith('.fits'):
        filename = filename[:-5]
    
    try:
        # Split by underscore to separate components
        # Format: RA_DEC_waveband_vN
        parts = filename.split('_')
        
        if len(parts) < 4:
            return None
        
        # Last part should be vN
        version_part = parts[-1]
        if not version_part.startswith('v'):
            return None
        
        try:
            version = int(version_part[1:])
        except ValueError:
            return None
        
        # Waveband is second to last
        waveband = parts[-2]
        
        # RA and DEC - need to handle negative DEC
        # Format: RA_DEC where DEC might start with - sign
        # Find where DEC starts (after the RA part)
        ra_str = parts[0]
        
        try:
            ra = float(ra_str)
        except ValueError:
            return None
        
        # DEC could be split across multiple parts if it's negative
        # parts[1] should be the DEC (could be negative)
        if len(parts) >= 2:
            dec_str = parts[1]
            try:
                dec = float(dec_str)
            except ValueError:
                return None
        else:
            return None
        
        return {
            'ra': ra,
            'dec': dec,
            'waveband': waveband,
            'version': version
        }
    except (ValueError, IndexError):
        return None


def coords_to_filename_prefix(ra: float, dec: float) -> str:
    '''(Legacy function - kept for backwards compatibility)
    
    Combines RA and DEC for a filename using old DMS format.'''

    ra_str = decimal_to_dms(ra, is_ra=True)
    dec_str = decimal_to_dms(dec, is_ra=False)
    return f"RA{ra_str}_DEC{dec_str}"


def get_next_version_legacy(output_dir: str, coord_prefix: str) -> int:
    '''(Legacy function - kept for backwards compatibility)
    
    Finds next version for a group of files with the same coordinate prefix.'''

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


def generate_galaxy_filename_legacy(ra: float, dec: float, output_dir: str, 
                                     return_full_path: bool = False) -> tuple[str, int]:
    '''(Legacy function - kept for backwards compatibility)
    
    Generate full filename with both coordinates + version using old DMS format.
    Full example: "RA22h14m55s_DEC+45d22m30s_v1.fits"'''

    coord_prefix = coords_to_filename_prefix(ra, dec)       # get coordinates
    version = get_next_version_legacy(output_dir, coord_prefix)    # get next version
    filename = f"{coord_prefix}_v{version}.fits"
    
    if return_full_path:
        return os.path.join(output_dir, filename), version
    else:
        return filename, version


def parse_galaxy_filename_legacy(filename: str) -> Optional[dict]:
    '''(Legacy function - kept for backwards compatibility)
    
    Parse a coordinate-based filename to extract RA and DEC IN DEGREES, and version in a dictionary.

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
    # ! FOR TESTING - NEW FORMAT (v2)
    test_ra = 80.53921742149545  
    test_dec = -69.51885006523123   
    test_waveband = "f150w"
    
    print("=" * 80)
    print("NEW FORMAT (v2) TESTS")
    print("=" * 80)
    print(f"Input: RA={test_ra}, DEC={test_dec}, Waveband={test_waveband}")
    
    coord_string = coords_to_filename_string(test_ra, test_dec)
    print(f"Coordinate string: {coord_string}")
    
    filename, version = generate_galaxy_filename_v2(test_ra, test_dec, test_waveband, ".", return_full_path=False)
    print(f"Generated filename: {filename} (version {version})")
    
    parsed = parse_galaxy_filename_v2(filename)
    print(f"Parsed back: {parsed}")
    
    # Test with multiple wavebands
    print()
    for wb in ["f150w", "f070w", "f277w", "clear"]:
        fn, v = generate_galaxy_filename_v2(test_ra, test_dec, wb, ".", return_full_path=False)
        print(f"  {wb}: {fn}")

