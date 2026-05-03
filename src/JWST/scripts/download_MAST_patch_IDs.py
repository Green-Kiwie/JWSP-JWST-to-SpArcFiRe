'''
Downloads L3 JWST sky patches from MAST as a .csv with columns:
- dataURI: The URI to download the file from MAST
- productFilename: The filename of the product (e.g., 'jw1234567890_i2d.fits')
- obsID: The observation ID associated with the product
- target_name: The name of the target observed
- obs_collection: The collection the observation belongs to (e.g., 'JWST')
- calib_level: The calibration level of the product (should be 3 for L3 products)
- productType: The type of product (e.g., 'image', 'spectrum')

'''


from astroquery.mast import Observations
import pandas as pd
import time
import os

print("Querying all JWST observations...")
obs_table = Observations.query_criteria(obs_collection='JWST', calib_level=3)
total_obs = len(obs_table)

batch_size = 500
output_file = "jwst_L3_sky_patches.csv"

if not os.path.exists(output_file):
    pd.DataFrame().to_csv(output_file, index=False)

print(f"Processing {total_obs} observations to extract L3 image IDs...")

for i in range(0, total_obs, batch_size):
    try:
        subset = obs_table[i : i + batch_size]
        products = Observations.get_product_list(subset)
        
        df_batch = products.to_pandas()
        
        # FILTER 1: Only FITS files
        # FILTER 2: Only '_i2d.fits' (Level 3 2D mosaiced images)
        l3_images = df_batch[df_batch['productFilename'].str.endswith('_i2d.fits', na=False)]
        
        if not l3_images.empty:
            l3_images.to_csv(output_file, mode='a', header=(i==0), index=False)
        
        print(f"Progress: {min(i + batch_size, total_obs)}/{total_obs} observations processed...")
        
    except Exception as e:
        print(f"Error at index {i}: {e}. Pausing and continuing...")
        time.sleep(5)

print(f"Complete! All Level 3 IDs saved to {output_file}")