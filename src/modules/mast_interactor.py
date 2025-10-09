from astroquery.mast import Observations
from astropy.io import fits
import os
import pathlib
import tempfile

def get_fits_from_uri(mast_uri: str) -> fits.HDUList | None:
    '''
    
    Downloads a FITS file from a MAST URI to a temporary file,
    loads into memory as HDUList, then deletes temporary file.

    '''
    
    # Create a temporary file path
    tmp_file_handle = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    temp_filepath = tmp_file_handle.name
    tmp_file_handle.close() # Close the handle so Observations can write to the path

    try:
        # Download the data to the temporary file path
        print(f"Downloading {mast_uri} to temporary location...")
        Observations.download_file(mast_uri, local_path=temp_filepath)

        # Open the temporary file and copy its contents into memory
        with fits.open(temp_filepath) as hdul:
            hdul_in_memory = fits.HDUList([hdu.copy() for hdu in hdul])
        
        print(f"Successfully loaded {mast_uri} into memory.")
        return hdul_in_memory
    except Exception as e:
        print(f"Failed to process {mast_uri}. Error: {e}")
        return None
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
            print(f"Temporary file {temp_filepath} deleted.")

if __name__ == '__main__':
    test_uri = "mast:JWST/product/jw02731-o001_t001_miri_f560w_i2d.fits"
    hdul = get_fits_from_uri(test_uri)
    if hdul:
        print(hdul.info())