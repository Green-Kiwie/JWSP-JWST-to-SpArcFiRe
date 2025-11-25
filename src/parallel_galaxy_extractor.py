#!/usr/bin/env python3
'''
Parallel Galaxy Extractor for JWST FITS Files

This script processes JWST FITS files in parallel, extracting galaxies using SEP
and grouping them by coordinates across different wavebands. Designed for 
concurrent execution with multiple workers sharing a common work queue.

Key Features:
- Filesystem-based work queue for coordination between workers
- SQLite database for coordinate-based galaxy grouping
- Within-file deduplication to handle SEP multi-detections
- Atomic file operations to prevent race conditions
- Per-worker statistics tracking

Usage example:
    python parallel_galaxy_extractor.py \\
        --worker-id 0 \\
        --input src/output \\
        --output galaxy_crops \\
        --db-path galaxy_coords.db \\
        --queue-dir work_queue \\
        --stats-dir stats \\
        --min-size 15 \\
        --max-size 100 \\
        --verbosity 1
        --mode full \\
        --
'''

import os
import sys
import glob
import time
import hashlib
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from coordinate_database import CoordinateDatabase
from memory_manager import MemoryManager


class WorkQueue:
    '''Filesystem-based work queue for parallel processing with memory management.
    
    Uses atomic file operations to coordinate work distribution among
    multiple worker processes without requiring locks. Integrates memory
    management to enforce global memory budget and handles retry queue
    for skipped large files.
    '''
    
    def __init__(self, queue_dir: str, stats_dir: str, all_files: List[str], max_memory_gb: float = 80.0):
        '''Initialize work queue with memory management.
        
        Args:
            queue_dir: Directory for queue state
            stats_dir: Directory for stats and memory tracking
            all_files: List of all files to process
            max_memory_gb: Global memory budget in GB
        '''
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # Create work items
        self.work_file = self.queue_dir / 'work_items.txt'
        if not self.work_file.exists():
            with open(self.work_file, 'w') as f:
                for filepath in sorted(all_files):
                    f.write(f"{filepath}\n")
        
        self.claimed_dir = self.queue_dir / 'claimed'
        self.completed_dir = self.queue_dir / 'completed'
        self.skipped_dir = self.queue_dir / 'skipped'  
        self.claimed_dir.mkdir(exist_ok=True)
        self.completed_dir.mkdir(exist_ok=True)
        self.skipped_dir.mkdir(exist_ok=True)
        
        # Initialize memory manager
        self.mem_manager = MemoryManager(stats_dir, max_total_memory_gb=max_memory_gb)
    
    def _get_file_hash(self, filepath: str) -> str:
        '''Get short hash of filepath for unique identification.'''
        return hashlib.md5(filepath.encode()).hexdigest()[:16]
    
    def claim_next_file(self, worker_id: int) -> Optional[str]:
        '''Claim next available file for processing, respecting memory budget.
        
        Files that exceed available memory are moved to retry queue and will
        be retried when memory becomes available. Continuously retries skipped
        files when memory becomes available.
        
        Returns filepath if work available, None if all work claimed/completed.
        '''
        # Refresh claimed/completed/skipped lists
        claimed_hashes = set(f.name for f in self.claimed_dir.iterdir())
        completed_hashes = set(f.name for f in self.completed_dir.iterdir())
        skipped_hashes = set(f.name for f in self.skipped_dir.iterdir())
        
        # First, try files from normal work queue
        with open(self.work_file, 'r') as f:
            all_files = [line.strip() for line in f if line.strip()]
        
        for filepath in all_files:
            file_hash = self._get_file_hash(filepath)
            
            # Skip already completed/claimed files
            if file_hash in claimed_hashes or file_hash in completed_hashes:
                continue
            
            # Check memory before claiming
            can_process, reason = self.mem_manager.can_process_file(filepath)
            
            if not can_process:
                # File exceeds memory budget - move to retry queue
                skip_file = self.skipped_dir / file_hash
                if file_hash not in skipped_hashes:
                    try:
                        with open(skip_file, 'w') as f:
                            f.write(f"{filepath}\n{reason}\n{time.time()}\n")
                    except (IOError, OSError):
                        pass
                continue
            
            # Attempt to claim file
            claim_file = self.claimed_dir / file_hash
            try:
                with open(claim_file, 'x') as f:
                    f.write(f"worker_{worker_id}\n{filepath}\n{time.time()}\n")
                
                # Register memory for this file
                estimated_mem = self.mem_manager.estimate_file_memory_usage(filepath)
                self.mem_manager.register_worker_memory(worker_id, filepath, estimated_mem)
                
                # Remove from skipped if it was there
                skip_file = self.skipped_dir / file_hash
                if skip_file.exists():
                    skip_file.unlink()
                
                return filepath
            except FileExistsError: # Another worker claimed first
                continue
        
        # Second, retry files from skipped queue (waiting for memory)
        skipped_files = [f for f in self.skipped_dir.iterdir()]
        for skip_file in skipped_files:
            try:
                with open(skip_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 0:
                        filepath = lines[0].strip()
                        
                        # Check if now has memory available
                        can_process, reason = self.mem_manager.can_process_file(filepath)
                        
                        if can_process:
                            # Try to claim it
                            file_hash = self._get_file_hash(filepath)
                            claim_file = self.claimed_dir / file_hash
                            
                            try:
                                with open(claim_file, 'x') as cf:
                                    cf.write(f"worker_{worker_id}\n{filepath}\n{time.time()}\n")
                                
                                # Register memory for this file
                                estimated_mem = self.mem_manager.estimate_file_memory_usage(filepath)
                                self.mem_manager.register_worker_memory(worker_id, filepath, estimated_mem)
                                
                                # Remove from skipped
                                skip_file.unlink()
                                
                                return filepath
                            except FileExistsError:
                                # Another worker claimed first, try next skipped file
                                continue
            except (IOError, OSError):
                continue
        
        # No work available and no skipped files ready
        return None
    
    def mark_completed(self, filepath: str, worker_id: int):
        '''Mark file as completed and unregister memory.'''
        # Unregister memory for this worker
        self.mem_manager.unregister_worker_memory(worker_id)
        
        file_hash = self._get_file_hash(filepath)
        
        # Move from claimed to completed
        claim_file = self.claimed_dir / file_hash
        complete_file = self.completed_dir / file_hash
        
        with open(complete_file, 'w') as f:
            f.write(f"worker_{worker_id}\n{filepath}\n{time.time()}\n")
        
        # Remove claim file if it exists
        if claim_file.exists():
            claim_file.unlink()


class ProcessingStats:
    '''Track per-worker processing statistics.'''
    
    def __init__(self, stats_file: str):
        '''Initialize stats tracker.
        
        Args:
            stats_file: Path to JSON stats file
        '''
        self.stats_file = Path(stats_file)
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.files_processed = 0
        self.bytes_read = 0
        self.bytes_written = 0
        self.galaxies_extracted = 0
        self.mode = 'full'  
    
    def update(self, bytes_read: int = 0, bytes_written: int = 0, galaxies: int = 0):
        '''Update statistics.'''
        self.files_processed += 1
        self.bytes_read += bytes_read
        self.bytes_written += bytes_written
        self.galaxies_extracted += galaxies
        self._save()
    
    def _save(self):
        '''Save statistics to JSON file.'''
        elapsed = time.time() - self.start_time
        
        stats = {
            'start_time': self.start_time,
            'elapsed_seconds': elapsed,
            'files_processed': self.files_processed,
            'bytes_read': self.bytes_read,
            'bytes_written': self.bytes_written,
            'galaxies_extracted': self.galaxies_extracted,
            'stage': self.mode,
            'read_bandwidth_mbps': (self.bytes_read / (1024**2)) / elapsed if elapsed > 0 else 0,
            'write_bandwidth_mbps': (self.bytes_written / (1024**2)) / elapsed if elapsed > 0 else 0,
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


def process_single_fits(
    filepath: str,
    output_dir: str,
    db: CoordinateDatabase,
    min_size: int = 20,
    max_size: int = 100,
    verbosity: int = 1,
    mode: str = 'full',
    visualize_crops: bool = False,
    viz_output_dir: Optional[str] = None
) -> Tuple[int, int]:
    '''Process a single FITS file.
    
    Args:
        filepath: Path to FITS file
        db: Coordinate database for grouping
        output_dir: Output directory for cropped galaxies
        min_size: Minimum crop size in pixels
        max_size: Maximum crop size in pixels
        verbosity: Verbosity level (0=quiet, 1=normal, 2=verbose)
        mode: Processing mode ('full'=scan+crop, 'scan-only'=db only, 'crop-only'=use existing db)
        visualize_crops: Whether to generate visualization PNG
        viz_output_dir: Directory for visualization PNG files
    
    Returns:
        Tuple of (file_size, galaxies_extracted)
    '''
    from galaxy_naming import round_coordinate
    import sep_helpers as sh
    from waveband_extraction import extract_waveband_from_filename
    from visualization import create_visualization
    
    try:
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        
        if verbosity >= 1:
            print(f"Processing {filename}...")
        
        # Load FITS
        image_meta_data = sh.get_relevant_fits_meta_data(filepath)
        image_data = sh.get_main_fits_data(filepath)
        
        # Extract waveband from filename
        waveband = extract_waveband_from_filename(filename)
        
        # Run SEP detection
        bkg = sh.get_image_background(image_data)
        bkgless_data = sh.subtract_bkg(image_data)
        celestial_objects = sh.extract_objects(bkgless_data, bkg, pixel_stack_limit=5000000, sub_object_limit=50000)
        
        if verbosity >= 1:
            print(f"{len(celestial_objects)} objects found.")
        
        galaxies_saved = 0
        duplicates_skipped = 0
        
        # Track positions already processed in this file (for deduplication)
        # Key: (ra_rounded, dec_rounded), Value: (crop_size, object_index)
        processed_positions = {}
        
        # Track galaxy detections for visualization
        galaxy_detections_for_viz = []
        
        # Process each detected object
        for i, obj in enumerate(celestial_objects):
            cropped_data = sh._extract_object(obj, image_data, padding=1.5, min_size=min_size, max_size=max_size, verbosity=0)
            
            if cropped_data is None:
                continue
            
            height, width = cropped_data.shape
            crop_size = height * width
            
            try:
                obj_ra, obj_dec = sh._get_gal_position(obj, image_meta_data)
            except Exception as e:
                if verbosity >= 2:
                    print(f"  Skipping object {i}: could not determine position - {e}")
                continue
            
            # Check for duplicates within this file
            ra_key = round_coordinate(obj_ra, decimals=6)
            dec_key = round_coordinate(obj_dec, decimals=6)
            position_key = (ra_key, dec_key)
            
            if position_key in processed_positions:
                prev_size, prev_idx = processed_positions[position_key]
                if crop_size <= prev_size:
                    duplicates_skipped += 1
                    if verbosity >= 2:
                        print(f"  Skipping duplicate at {ra_key:.6f}, {dec_key:.6f} (smaller, {crop_size} <= {prev_size})")
                    continue
                else:
                    if verbosity >= 2:
                        print(f"  Replacing duplicate at {ra_key:.6f}, {dec_key:.6f} with larger detection ({crop_size} > {prev_size})")
            
            # Mark this position as processed
            processed_positions[position_key] = (crop_size, i)
            
            # In crop-only mode, don't add to DB again (avoid duplicates)
            if mode == 'crop-only':
                # Look up existing galaxy instance by coordinates AND source file
                result = db.find_galaxy_version_by_source(obj_ra, obj_dec, filename, pos_tolerance=0.5/3600.0)
                if result is None:
                    if verbosity >= 2:
                        print(f"  Warning: Galaxy at {obj_ra:.6f}, {obj_dec:.6f} from {filename} not found in database (skipping)")
                    continue
                group_id, version, canonical_ra, canonical_dec = result
            else:
                group_id, version, canonical_ra, canonical_dec = db.add_galaxy(obj_ra, obj_dec, height, width, waveband, filename)
            
            # Get group info to determine max dimensions
            group_info = db.get_group_info(group_id)
            max_height = group_info['max_height']
            max_width = group_info['max_width']
            
            # If this detection is LARGER than group max, update the database
            if height > max_height or width > max_width:
                new_max_height = max(height, max_height)
                new_max_width = max(width, max_width)
                db.update_group_max_dimensions(group_id, new_max_height, new_max_width)
                max_height = new_max_height
                max_width = new_max_width

            # Standardize crop to group's max dimensions
            if height != max_height or width != max_width:
                cropped_data = sh.standardize_crop_dimensions(cropped_data, max_height, max_width)
            
            # Only save if not in scan-only mode
            if mode != 'scan-only':
                # Update metadata
                updated_meta_data = sh._update_meta_data(obj, cropped_data, image_meta_data, source_filename=filename)
                
                # Generate filename using canonical group coordinates (all galaxies in same group use same name)
                canonical_ra_key = round_coordinate(canonical_ra, decimals=6)
                canonical_dec_key = round_coordinate(canonical_dec, decimals=6)
                output_filename = f"{canonical_ra_key:.6f}_{canonical_dec_key:.6f}_{waveband}_v{version}.fits"
                output_path = os.path.join(output_dir, output_filename)
                
                # Save to FITS
                sh.save_to_fits(output_path, cropped_data, updated_meta_data)
                galaxies_saved += 1
            
            if verbosity >= 2:
                print(f"  Saved: {output_filename} (group {group_id}, v{version})")
            
            # Track for visualization
            if visualize_crops and viz_output_dir:
                galaxy_detections_for_viz.append((
                    obj['x'],
                    obj['y'],
                    width,
                    height
                ))
        
        # Generate visualization if requested
        if visualize_crops and viz_output_dir and galaxy_detections_for_viz:
            viz_output_dir_path = Path(viz_output_dir)
            viz_output_dir_path.mkdir(parents=True, exist_ok=True)
            
            # Generate PNG filename based on input FITS filename
            base_name = os.path.splitext(filename)[0]
            viz_filename = f"{base_name}_galaxies.png"
            viz_path = os.path.join(viz_output_dir, viz_filename)
            
            success = create_visualization(
                image_data,
                galaxy_detections_for_viz,
                viz_path,
                max_dimension=2048,
                circle_color=(0, 255, 0),
                circle_width=2
            )
            
            if success and verbosity >= 1:
                print(f"  Visualization saved: {viz_filename}")
        
        if verbosity >= 1:
            msg = f"[Worker] Completed {filename} ({galaxies_saved} galaxies"
            if duplicates_skipped > 0:
                msg += f", {duplicates_skipped} duplicates skipped"
            msg += ")"
            print(msg)
        
        return file_size, galaxies_saved
        
    except Exception as e:
        print(f"ERROR processing {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0, 0


def worker_process(
    worker_id: int,
    input_dir: str,
    output_dir: str,
    db_path: str,
    queue_dir: str,
    stats_dir: str,
    stats_file: str,
    min_size: int = 20,
    max_size: int = 100,
    verbosity: int = 1,
    mode: str = 'full',
    visualize_crops: bool = False,
    viz_output_dir: Optional[str] = None,
    max_memory_gb: float = 80.0
):
    '''Main worker process loop with memory management.'''
    
    print(f"[Worker {worker_id}] Starting...")
    
    # Initialize database connection
    db = CoordinateDatabase(db_path, position_tolerance_arcsec=0.5)
    
    # Initialize stats with mode/stage
    stats = ProcessingStats(stats_file)
    stats.mode = mode  # Store mode as stage for monitoring
    
    # Get work queue with memory management
    all_files = sorted(glob.glob(os.path.join(input_dir, '*.fits')))
    work_queue = WorkQueue(queue_dir, stats_dir, all_files, max_memory_gb=max_memory_gb)
    
    files_processed = 0
    
    while True:
        # Claim next file
        filepath = work_queue.claim_next_file(worker_id)
        
        if filepath is None:
            # No more work
            print(f"[Worker {worker_id}] No more files to process")
            break
        
        # Process file
        start_time = time.time()
        bytes_read, galaxies = process_single_fits(
            filepath, 
            output_dir,
            db, 
            min_size=min_size, 
            max_size=max_size, 
            verbosity=verbosity,
            mode=mode,
            visualize_crops=visualize_crops,
            viz_output_dir=viz_output_dir
        )
        elapsed = time.time() - start_time
        
        # Update stats
        stats.update(bytes_read=bytes_read, galaxies=galaxies)
        files_processed += 1
        
        # Mark as completed
        work_queue.mark_completed(filepath, worker_id)
        
        # Progress message
        bandwidth_mbps = (bytes_read / (1024**2)) / elapsed if elapsed > 0 else 0
        print(f"[Worker {worker_id}] Completed {os.path.basename(filepath)} "
              f"({elapsed:.1f}s, {bytes_read/(1024**2):.1f}MB, {galaxies} galaxies) | "
              f"Rate: {bandwidth_mbps:.2f} MB/s")
    
    print(f"[Worker {worker_id}] Finished. Processed {files_processed} files.")


def main():
    '''Main entry point.'''
    parser = argparse.ArgumentParser(
        description='Parallel galaxy extractor for JWST FITS files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--worker-id', type=int, required=True,
                        help='Unique worker ID (0, 1, 2, ...)')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory containing FITS files')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for cropped galaxies')
    parser.add_argument('--db-path', type=str, required=True,
                        help='Path to SQLite coordinate database')
    parser.add_argument('--queue-dir', type=str, required=True,
                        help='Directory for work queue state')
    parser.add_argument('--stats-dir', type=str, required=True,
                        help='Directory for worker statistics')
    parser.add_argument('--min-size', type=int, default=20,
                        help='Minimum crop size in pixels (default: 20)')
    parser.add_argument('--max-size', type=int, default=100,
                        help='Maximum crop size in pixels (default: 100)')
    parser.add_argument('--verbosity', type=int, default=1,
                        help='Verbosity level: 0=quiet, 1=normal, 2=verbose (default: 1)')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'scan-only', 'crop-only'],
                        help='Processing mode: full (scan+crop), scan-only (build DB only), crop-only (use existing DB)')
    parser.add_argument('--visualize-crops', default='false', action='store_true',
                        help='Generate PNG visualizations of detected galaxies with circles marking crop regions')
    
    args = parser.parse_args()
    
    # For crop-only mode, skip creating output dir if it doesn't exist yet
    # Also skip for /dev/null (scan-only mode)
    if args.mode != 'crop-only' and args.output != '/dev/null':
        os.makedirs(args.output, exist_ok=True)
    
    # Create visualization output directory if needed
    viz_output_dir = None
    if args.visualize_crops:
        viz_output_dir = os.path.join(args.output, 'visualizations')
    
    # Stats file path
    stats_file = os.path.join(args.stats_dir, f'worker_{args.worker_id}_stats.json')
    
    # Run worker
    worker_process(
        worker_id=args.worker_id,
        input_dir=args.input,
        output_dir=args.output,
        db_path=args.db_path,
        queue_dir=args.queue_dir,
        stats_dir=args.stats_dir,
        stats_file=stats_file,
        min_size=args.min_size,
        max_size=args.max_size,
        verbosity=args.verbosity,
        mode=args.mode,
        visualize_crops=args.visualize_crops,
        viz_output_dir=viz_output_dir,
        max_memory_gb=80.0
    )


if __name__ == '__main__':
    main()
