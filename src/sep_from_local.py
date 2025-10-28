'''
Runs sep on pre-existing local FITS files.

Usage:
    python sep_from_local.py --file /path/to/image.fits
    python sep_from_local.py --directory /path/to/fits/directory
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))
import argparse
import glob
import numpy as np
import sep_pjw as sep
import sep_helpers as sh
from typing import List

def validate_fits_file(filepath: str) -> bool:
    '''Validate that a file is a readable FITS file.'''
    if not os.path.isfile(filepath):
        return False
    if not filepath.lower().endswith('.fits'):
        return False
    return True


def get_fits_files(input_path: str) -> List[str]:
    '''Return a single FITS file or list of FITS files from a directory.'''
    input_path = os.path.abspath(input_path)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path does not exist: {input_path}")
    
    # Single file
    if os.path.isfile(input_path):
        if not validate_fits_file(input_path):
            raise ValueError(f"Not a valid FITS file: {input_path}")
        return [input_path]
    
    # Directory
    elif os.path.isdir(input_path):
        fits_files = glob.glob(os.path.join(input_path, '**', '*.fits'), recursive=True)
        if not fits_files:
            raise FileNotFoundError(f"No FITS files found in directory: {input_path}")
        return sorted(fits_files)
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def process_fits_file(filepath: str, output_dir: str) -> bool:
    '''Process single FITS file with SEP and save cropped objects.'''
    try:    
        image_data = sh.get_main_fits_data(filepath)                # np.ndarray
        image_meta_data = sh.get_relevant_fits_meta_data(filepath)  # dict
        
        if image_data is None:
            print(f"ERROR: Could not load image data from {filepath}")
            return False
        
        bkg = sh.get_image_background(image_data)
        bkgless_data = sh.subtract_bkg(image_data)
        celestial_objects = sh.extract_objects(bkgless_data, bkg)

        #bkg = sep.Background(image_data.astype(np.float32))
        #bkg_rms = bkg.globalrms
        #backgroundless_data = image_data - bkg
        #celestial_objects = sep.extract(backgroundless_data.astype(np.float32), thresh=3.0, err=bkg_rms)
        
        #print(f"Number of objects detected: {len(celestial_objects)}")
        sh.extract_objects_to_file(
            image_data=image_data,
            image_meta_data=image_meta_data,
            file_name=os.path.basename(filepath),
            celestial_objects=celestial_objects,
            output_dir=output_dir
        )
        
        return True
        
    except Exception as e:
        print(f"ERROR processing {filepath}: {str(e)}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run SEP on local FITS files with improved galaxy naming',
        epilog="""
Examples:
  # Single file
  python sep_from_local.py --file image.fits
  
  # Directory
  python sep_from_local.py --directory /path/to/fits/
        """
    )
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file',
        type=str,
        help='Path to a single FITS file'
    )
    input_group.add_argument(
        '--directory',
        type=str,
        help='Path to directory containing FITS files'
    )
    input_group.add_argument(
        '--output',
        type=str,
        default='sep_output',
        help='Output directory for cropped objects (default: sep_output)'
    )
    
    args = parser.parse_args()
    input_path = args.file or args.directory
    print(f"Input path: {input_path}")
    
    try:
        fits_files = get_fits_files(input_path)
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        
        successful = 0
        failed = 0
        
        for i, filepath in enumerate(fits_files, 1):
            if len(fits_files) > 1:
                print(f"\n[{i}/{len(fits_files)}]", end=' ')
            
            if process_fits_file(filepath, output_dir, fits_files):
                successful += 1
            else:
                failed += 1
        
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Total files:      {len(fits_files)}")
        print(f"Successful:       {successful}")
        print(f"Failed:           {failed}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*70}")
        
        if failed > 0:
            sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
