from astroquery.mast import Observations
import os
import pathlib

def download_file(mast_uri: str, file_name: str) -> bool:
    '''downloads the fits file based on the mast_uri'''
    if pathlib.Path(file_name).exists():
        os.remove(file_name)

    download_status = Observations.download_file(mast_uri, local_path = file_name)
    if download_status[0] == 'COMPLETE':
        print(f"download of {mast_uri} successful")
        return True
    else:
        print(f"download of {mast_uri} failed. error: {download_status}")
        return False


if __name__ == '__main__':
    download_file("mast:JWST/product/jw01685013001_04101_00004_nrcb4_i2d.fits", 'mast_download.fits')