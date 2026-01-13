'''
Runs SEP on FITS files in discovery mode: identifies groups and max crop dimensions
without storing image data. This produces a grouping map for later use.

Usage:
    python sep_discovery.py --directory /path/to/fits/ --output groups.json
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

import argparse
import glob
import sep_helpers as sh
import file_handling_class as fhc
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
    
    if os.path.isfile(input_path):
        if not validate_fits_file(input_path):
            raise ValueError(f"Not a valid FITS file: {input_path}")
        return [input_path]
    
    elif os.path.isdir(input_path):
        fits_files = glob.glob(os.path.join(input_path, '**', '*.fits'), recursive=True)
        if not fits_files:
            raise FileNotFoundError(f"No FITS files found in directory: {input_path}")
        return sorted(fits_files)
    else:
        raise ValueError(f"Invalid input path: {input_path}")


def format_bytes(size_bytes: int) -> str:
    '''Convert bytes to human-readable format (B, KB, MB, GB, TB).'''
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def get_file_size(filepath: str) -> int:
    '''Get file size in bytes. Returns 0 if file doesn't exist.'''
    if os.path.exists(filepath):
        return os.path.getsize(filepath)
    return 0


def print_json_status(json_path: str, file_count: int, group_count: int, object_count: int) -> None:
    '''Print current JSON file size and discovery statistics.'''
    file_size = get_file_size(json_path)
    size_str = format_bytes(file_size)
    
    print(f"\n  JSON file size:     {size_str}")
    print(f"  Groups discovered:  {group_count}")
    print(f"  Objects catalogued: {object_count}")
    print(f"  Files processed:    {file_count}")


def process_fits_file_discovery(filepath: str, discovery_instance: fhc.Thumbnail_Handling,
                                min_size: int = 20, max_size: int = 10000) -> bool:
    try:
        image_data = sh.get_main_fits_data(filepath)
        image_meta_data = sh.get_relevant_fits_meta_data(filepath)
        if image_data is None:
            print(f"  ERROR: Could not load image data")
            return False

        bkg_model = sh.get_background_model(image_data)
        bkgless_data = image_data - bkg_model.back()
        celestial_objects = sh.extract_objects(bkgless_data, bkg_model.rms())

        basename = os.path.basename(filepath)
        object_count = 0
        for i, obj in enumerate(celestial_objects):
            crop_data = sh._extract_object(obj, image_data, padding=1.5,
                                           min_size=min_size, max_size=max_size,
                                           verbosity=0)
            if crop_data is not None:
                updated_meta_data = sh._update_meta_data(obj, crop_data, image_meta_data,
                                                         source_filename=basename)
                gal_pos = fhc.Gal_Pos(updated_meta_data)
                ra, dec = gal_pos.get_pos()
                height, width = crop_data.shape
                discovery_instance.add_object_discovery_only(ra, dec, height, width, basename)
                object_count += 1

        print(f"  Found and catalogued {object_count} objects")
        return True
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback; traceback.print_exc()
        return False


def main():
    '''Main entry point for discovery mode.'''
    parser = argparse.ArgumentParser(
        description='Discover coordinate groups in FITS files without storing image data',
        epilog="""
            Examples:
            # Scan directory
            python sep_discovery.py --directory /path/to/fits/ --output groups.json
            
            # Single file
            python sep_discovery.py --file image.fits --output groups.json
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--file', type=str, help='Path to a single FITS file')
    input_group.add_argument('--directory', type=str, help='Path to directory containing FITS files')
    
    parser.add_argument('--output', type=str, default='group_discovery.json',
                       help='Output JSON file for group metadata (default: group_discovery.json)')
    parser.add_argument('--min-size', type=int, default=20,
                       help='Minimum object size in pixels')
    parser.add_argument('--max-size', type=int, default=10000,
                       help='Maximum object size in pixels')
    parser.add_argument('--detailed', action='store_true', default=False,
                       help='Print detailed group information')
    parser.add_argument('--checkpoint', type=int, default=100,
                       help='Save and print JSON size every N files')
    
    args = parser.parse_args()
    input_path = args.file or args.directory
    
    try:
        fits_files = get_fits_files(input_path)
        output_path = os.path.abspath(args.output)
        
        print(f"\n{'='*70}")
        print("FITS FILE GROUP DISCOVERY")
        print(f"{'='*70}")
        print(f"Input path:      {input_path}")
        print(f"FITS files:      {len(fits_files)}")
        print(f"Output file:     {output_path}")
        print(f"Checkpoint:      Every {args.checkpoint} files")
        print(f"{'='*70}\n")
        
        # Create discovery instance (no data stored, only metadata)
        discovery = fhc.Thumbnail_Handling(None, discovery_mode=True)
        
        successful = 0
        failed = 0
        
        for i, filepath in enumerate(fits_files, 1):
            basename = os.path.basename(filepath)
            print(f"[{i}/{len(fits_files)}] {basename}")
            
            if process_fits_file_discovery(filepath, discovery, args.min_size, args.max_size):
                successful += 1
            else:
                failed += 1
            
            # Checkpoint: save and report JSON size every N files
            if i % args.checkpoint == 0 or i == len(fits_files):
                print(f"\n{'─'*70}")
                print(f"CHECKPOINT: {i}/{len(fits_files)} files processed")
                print(f"{'─'*70}")
                
                summary = discovery.get_discovery_summary()
                discovery.save_discovery_metadata(output_path)
                print_json_status(output_path, i, summary['group_count'], summary['total_members'])
                print(f"{'─'*70}\n")
        
        print(f"\n{'='*70}")
        print("DISCOVERY COMPLETE")
        print(f"{'='*70}\n")
        
        # Print results
        discovery.print_discovery_report(detailed=args.detailed, detail_limit=30)
        
        # Final save and report
        discovery.save_discovery_metadata(output_path)
        
        summary = discovery.get_discovery_summary()
        final_size = get_file_size(output_path)
        
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Files scanned:          {len(fits_files)}")
        print(f"Successful:             {successful}")
        print(f"Failed:                 {failed}")
        print(f"Groups found:           {summary['group_count']}")
        print(f"Total objects:          {summary['total_members']}")
        print(f"Final JSON file size:   {format_bytes(final_size)}")
        print(f"Metadata saved to:      {output_path}")
        print(f"{'='*70}\n")
        
        if failed > 0:
            sys.exit(1)
    
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()