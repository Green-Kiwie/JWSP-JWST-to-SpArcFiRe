from astroquery.mast import Observations
import os

def download_file(mast_uri: str, file_name: str) -> None:
    '''downloads the fits file based on the mast_uri'''
    try:
        os.remove(file_name)
    except:
        pass

    download_status = Observations.download_file(mast_uri, local_path = file_name)
    if download_status[0] == 'COMPLETE':
        print(f"download of {mast_uri} successful")
    else:
        print(f"download of {mast_uri} failed. error: {download_status}")


if __name__ == '__main__':
    download_file("mast:JWST/product/jw01685013001_04101_00004_nrcb4_i2d.fits", 'mast_download.fits')