'''
Module for extracting waveband information from JWST FITS filenames.

JWST filenames typically follow the pattern:
  jw[program]-[visit]_t[exposure]_[instrument]_[filter]_i2d.fits

Examples:
  jw01018-c1000_t003_niriss_clear-f150w_i2d.fits → f150w
  jw01021-o001_t005_nircam_clear-f070w_i2d.fits → f070w
  jw01021-o003_t006_nircam_f405n-f444w_i2d.fits → f405n-f444w (prism)
  jw01018-o005_t003_fgs_clear_i2d.fits → clear (FGS has no filter)
'''

import re
from typing import Optional


def extract_waveband_from_filename(filename: str) -> Optional[str]:
    '''Extract waveband/filter from JWST FITS filename.
    
    Args:
        filename: FITS filename (e.g., "jw01018-c1000_t003_niriss_clear-f150w_i2d.fits")
    
    Returns:
        Waveband string (e.g., "f150w") or "unclassified" if not found
    
    Examples:
        >>> extract_waveband_from_filename("jw01018-c1000_t003_niriss_clear-f150w_i2d.fits")
        "f150w"
        >>> extract_waveband_from_filename("jw01021-o003_t006_nircam_f405n-f444w_i2d.fits")
        "f405n-f444w"
        >>> extract_waveband_from_filename("jw01018-o005_t003_fgs_clear_i2d.fits")
        "clear"
        >>> extract_waveband_from_filename("jw01685013001_02101_00004_nrca1_i2d.fits")
        "unclassified"
    '''
    
    # Remove .fits extension if present
    if filename.endswith('.fits'):
        filename = filename[:-5]
    
    # JWST filename pattern: jw[program]-[visit]_t[exposure]_[instrument]_[filter]_i2d
    # We want to extract the [filter] part
    
    # Pattern: Look for parts separated by underscores
    parts = filename.split('_')
    
    if len(parts) < 4:
        return "unclassified"
    
    # The filter/waveband is typically before '_i2d'
    # parts[-1] should be 'i2d'
    # parts[-2] should be the filter/waveband
    
    if parts[-1] == 'i2d' and len(parts) >= 2:
        waveband = parts[-2]
        
        # Clean up the waveband string
        # It might contain instrument info like "clear-f150w" or "f405n-f444w"
        # We want to extract just the filter parts, preferring f### patterns
        
        # Common patterns:
        # - "clear" (FGS or clear aperture)
        # - "f150w" (single filter)
        # - "clear-f150w" (clear aperture + filter) → should return f150w
        # - "f405n-f444w" (prism/grism modes)
        # - "clearp-f277w" (for NIRSpec/NIRISS modes) → should return f277w
        
        # Priority: look for f### patterns first, fall back to clear
        filter_patterns = [
            r'(f\d+[a-z]?(?:-f\d+[a-z]?)*)',  # Filter patterns like f150w or f405n-f444w
            r'(clear(?:p)?)',  # Clear/clearp patterns
        ]
        
        for pattern in filter_patterns:
            match = re.search(pattern, waveband, re.IGNORECASE)
            if match:
                return match.group(1).lower()
    
    # If we couldn't extract a waveband, return "unclassified" instead of None
    # This allows processing of non-standard FITS files
    return "unclassified"


def extract_instrument_from_filename(filename: str) -> Optional[str]:
    '''Extract instrument name from JWST FITS filename.
    
    Args:
        filename: FITS filename
    
    Returns:
        Instrument name (e.g., "nircam", "niriss", "fgs")
    
    Examples:
        >>> extract_instrument_from_filename("jw01018-c1000_t003_niriss_clear-f150w_i2d.fits")
        "niriss"
    '''
    
    # Remove .fits extension if present
    if filename.endswith('.fits'):
        filename = filename[:-5]
    
    parts = filename.split('_')
    
    if len(parts) >= 3:
        # Instrument is typically the third component
        instrument = parts[2].lower()
        # Clean up if there's additional info
        if instrument in ['nircam', 'niriss', 'niri', 'miri', 'fgs']:
            return instrument
    
    return None


if __name__ == "__main__":
    # Test cases
    test_filenames = [
        "jw01018-c1000_t003_niriss_clear-f150w_i2d.fits",
        "jw01021-o001_t005_nircam_clear-f070w_i2d.fits",
        "jw01021-o003_t006_nircam_f405n-f444w_i2d.fits",
        "jw01018-o005_t003_fgs_clear_i2d.fits",
        "jw01021-o005_t015_niriss_clearp-f277w_i2d.fits",
        "jw01021-o004_t006_nircam_f322w2-f323n_i2d.fits",
    ]
    
    print("Testing waveband extraction:")
    print("=" * 80)
    for filename in test_filenames:
        waveband = extract_waveband_from_filename(filename)
        instrument = extract_instrument_from_filename(filename)
        print(f"{filename}")
        print(f"  → Waveband: {waveband}, Instrument: {instrument}\n")
