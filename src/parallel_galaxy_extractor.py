#!/usr/bin/env python3
'''
Parallel Galaxy Extractor for JWST FITS Files

This script processes JWST FITS files in parallel, extracting galaxies using SEP
and grouping them by coordinates across different wavebands.

Usage examples:
    # Run with 7 workers (recommended for long-running processes)
    python parallel_galaxy_extractor.py \ 
        --num-workers 7 \ 
        --input output \ 
        --output count \
        --db-path count.db \ 
        --queue-dir count_queue \ 
        --stats-dir count_stats \ 
        --min-size 20 \ 
        --max-size 100 \ 
        --verbosity 1 \ 
        --mode scan-only

    (one liner example) 
    
    python parallel_galaxy_extractor.py --num-workers 7 --input output --output count --db-path count.db --queue-dir count_queue --stats-dir count_stats --min-size 20 --max-size 100 --verbosity 1 --mode scan-only
'''

import os
import sys
import glob
import time
import hashlib
import argparse
import json
import signal
import traceback
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules'))

from coordinate_database import CoordinateDatabase
from memory_manager import MemoryManager

_shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[Main] Received signal {signum}, initiating graceful shutdown...")


@contextmanager
def timeout_context(seconds: int, description: str = "operation"):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"{description} timed out after {seconds} seconds")
    
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield


class WorkQueue:
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
        self.failed_dir = self.queue_dir / 'failed'
        self.claimed_dir.mkdir(exist_ok=True)
        self.completed_dir.mkdir(exist_ok=True)
        self.skipped_dir.mkdir(exist_ok=True)
        self.failed_dir.mkdir(exist_ok=True)
        
        # Initialize memory manager
        self.mem_manager = MemoryManager(stats_dir, max_total_memory_gb=max_memory_gb)
        
        # If a worker dies, its claims become stale
        self.stale_claim_timeout = 3600
    
    def _get_file_hash(self, filepath: str) -> str:
        '''Get short hash of filepath for unique identification.'''
        return hashlib.md5(filepath.encode()).hexdigest()[:16]
    
    def _is_claim_stale(self, claim_file: Path) -> bool:
        '''Check if a claim file is stale.'''
        try:
            with open(claim_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    claim_time = float(lines[2].strip())
                    return (time.time() - claim_time) > self.stale_claim_timeout
        except (IOError, OSError, ValueError):
            pass
        return False
    
    def _reclaim_stale_files(self):
        '''Reclaim files from workers that appear to have died.'''
        try:
            for claim_file in self.claimed_dir.iterdir():
                if self._is_claim_stale(claim_file):
                    try:
                        with open(claim_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) >= 2:
                                filepath = lines[1].strip()
                                print(f"[Queue] Reclaiming stale file: {os.path.basename(filepath)}")
                        claim_file.unlink()
                    except (IOError, OSError):
                        pass
        except (IOError, OSError):
            pass
    
    def get_queue_status(self) -> dict:
        '''Get current queue status for monitoring.'''
        try:
            with open(self.work_file, 'r') as f:
                total_files = len([line for line in f if line.strip()])
            
            claimed = len(list(self.claimed_dir.iterdir()))
            completed = len(list(self.completed_dir.iterdir()))
            skipped = len(list(self.skipped_dir.iterdir()))
            failed = len(list(self.failed_dir.iterdir()))
            
            return {
                'total': total_files,
                'completed': completed,
                'claimed': claimed,
                'skipped': skipped,
                'failed': failed,
                'remaining': total_files - completed - failed,
                'first_pass_done': self._is_first_pass_complete()
            }
        except (IOError, OSError):
            return {'total': 0, 'completed': 0, 'claimed': 0, 'skipped': 0, 'failed': 0, 'remaining': 0, 'first_pass_done': False}
    
    def _is_first_pass_complete(self) -> bool:
        '''Check if the first pass through original files is complete.
        
        First pass is complete when all files have been either:
        - Completed
        - Failed  
        - Skipped (waiting in retry queue)
        '''
        try:
            with open(self.work_file, 'r') as f:
                all_files = [line.strip() for line in f if line.strip()]
            
            completed_hashes = set(f.name for f in self.completed_dir.iterdir())
            skipped_hashes = set(f.name for f in self.skipped_dir.iterdir())
            failed_hashes = set(f.name for f in self.failed_dir.iterdir())
            claimed_hashes = set(f.name for f in self.claimed_dir.iterdir())
            
            for filepath in all_files:
                file_hash = self._get_file_hash(filepath)
                
                if file_hash not in completed_hashes and file_hash not in skipped_hashes and \
                   file_hash not in failed_hashes and file_hash not in claimed_hashes:
                    return False
            
            # Check that nothing is currently being processed
            return len(claimed_hashes) == 0
            
        except (IOError, OSError):
            return False
    
    def claim_next_file(self, worker_id: int) -> Optional[str]:
        '''Claim next available file for processing.
        
        Two-phase processing:
        1. First pass: Process all files from work queue, skip files that don't fit in memory
        2. Second pass: After first pass completes, retry skipped files
        
        A skipped file is permanently failed only if:
        - All other workers are idle (no memory in use)
        - The file still cannot be processed
        
        Returns filepath if work available, None if all work claimed/completed.
        '''
        # Periodically reclaim stale files from dead workers
        self._reclaim_stale_files()
        
        # Refresh claimed/completed/skipped/failed lists
        try:
            claimed_hashes = set(f.name for f in self.claimed_dir.iterdir())
            completed_hashes = set(f.name for f in self.completed_dir.iterdir())
            skipped_hashes = set(f.name for f in self.skipped_dir.iterdir())
            failed_hashes = set(f.name for f in self.failed_dir.iterdir())
        except (IOError, OSError) as e:
            print(f"[Worker {worker_id}] Warning: Error reading queue directories: {e}")
            time.sleep(1)
            return None
        
        # PHASE 1: Try files from normal work queue first
        try:
            with open(self.work_file, 'r') as f:
                all_files = [line.strip() for line in f if line.strip()]
        except (IOError, OSError) as e:
            print(f"[Worker {worker_id}] Warning: Error reading work file: {e}")
            time.sleep(1)
            return None
        
        for filepath in all_files:
            file_hash = self._get_file_hash(filepath)
            
            # Skip already completed/claimed/failed/skipped files
            if file_hash in claimed_hashes or file_hash in completed_hashes or \
               file_hash in failed_hashes or file_hash in skipped_hashes:
                continue
            
            # Check if file still exists
            if not os.path.exists(filepath):
                print(f"[Worker {worker_id}] Warning: File no longer exists: {filepath}")
                continue
            
            # Check memory before claiming
            try:
                can_process, reason = self.mem_manager.can_process_file(filepath)
            except Exception as e:
                print(f"[Worker {worker_id}] Warning: Memory check failed for {filepath}: {e}")
                continue
            
            if not can_process:
                # File exceeds memory budget - move to skipped/retry queue
                skip_file = self.skipped_dir / file_hash
                try:
                    with open(skip_file, 'w') as f:
                        f.write(f"{filepath}\n{reason}\n{time.time()}\n")
                    print(f"[Worker {worker_id}] Skipped (memory): {os.path.basename(filepath)} - {reason}")
                except (IOError, OSError):
                    pass
                continue
            
            # Attempt to claim file
            claim_file = self.claimed_dir / file_hash
            try:
                with open(claim_file, 'x') as f:
                    f.write(f"worker_{worker_id}\n{filepath}\n{time.time()}\n")
                
                # Register memory for this file
                try:
                    estimated_mem = self.mem_manager.estimate_file_memory_usage(filepath)
                    self.mem_manager.register_worker_memory(worker_id, filepath, estimated_mem)
                except Exception as e:
                    print(f"[Worker {worker_id}] Warning: Memory registration failed: {e}")
                
                return filepath
            except FileExistsError:  # Another worker claimed first
                continue
            except (IOError, OSError) as e:
                print(f"[Worker {worker_id}] Warning: Error claiming file: {e}")
                continue
        
        # PHASE 2: Only process skipped files after first pass is complete
        if not self._is_first_pass_complete():
            # First pass not done
            return None
        
        # First pass complete - now retry skipped files
        try:
            skipped_files = list(self.skipped_dir.iterdir())
        except (IOError, OSError):
            skipped_files = []
        
        if not skipped_files:
            return None
        
        # Check if all workers are idle (no memory in use)
        all_workers_idle = self.mem_manager.is_all_workers_idle()
        
        for skip_file in skipped_files:
            try:
                with open(skip_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        continue
                    
                    filepath = lines[0].strip()
                    file_hash = self._get_file_hash(filepath)
                    
                    # Check if file still exists
                    if not os.path.exists(filepath):
                        skip_file.unlink()
                        continue
                    
                    # Check if now has memory available
                    can_process, reason = self.mem_manager.can_process_file(filepath)
                    
                    if can_process:
                        # Try to claim it
                        claim_file = self.claimed_dir / file_hash
                        
                        try:
                            with open(claim_file, 'x') as cf:
                                cf.write(f"worker_{worker_id}\n{filepath}\n{time.time()}\n")
                            
                            # Register memory for this file
                            estimated_mem = self.mem_manager.estimate_file_memory_usage(filepath)
                            self.mem_manager.register_worker_memory(worker_id, filepath, estimated_mem)
                            
                            # Remove from skipped
                            skip_file.unlink()
                            
                            print(f"[Worker {worker_id}] Retrying previously skipped: {os.path.basename(filepath)}")
                            return filepath
                        except FileExistsError:
                            # Another worker claimed first
                            continue
                    
                    elif all_workers_idle:
                        # File cannot be processed even with all memory available
                        failed_file = self.failed_dir / file_hash
                        try:
                            with open(failed_file, 'w') as ff:
                                ff.write(f"worker_{worker_id}\n{filepath}\n{time.time()}\n")
                                ff.write(f"Permanently failed: {reason}\n")
                                ff.write(f"reason: File too large even with full memory budget available\n")
                            skip_file.unlink()
                            print(f"[Worker {worker_id}] Permanently failed (too large): {os.path.basename(filepath)}")
                        except (IOError, OSError):
                            pass
                        continue
                    
            except (IOError, OSError, ValueError):
                continue
        
        # No work available
        return None
    
    def mark_completed(self, filepath: str, worker_id: int):
        '''Mark file as completed and unregister memory.'''
        # Unregister memory for this worker
        try:
            self.mem_manager.unregister_worker_memory(worker_id)
        except Exception as e:
            print(f"[Worker {worker_id}] Warning: Memory unregistration failed: {e}")
        
        file_hash = self._get_file_hash(filepath)
        
        # Move from claimed to completed
        claim_file = self.claimed_dir / file_hash
        complete_file = self.completed_dir / file_hash
        
        try:
            with open(complete_file, 'w') as f:
                f.write(f"worker_{worker_id}\n{filepath}\n{time.time()}\n")
        except (IOError, OSError) as e:
            print(f"[Worker {worker_id}] Warning: Error marking completed: {e}")
        
        # Remove claim file if it exists
        try:
            if claim_file.exists():
                claim_file.unlink()
        except (IOError, OSError):
            pass
    
    def mark_failed(self, filepath: str, worker_id: int, error_msg: str, max_retries: int = 3):
        '''Mark file as failed, with retry tracking.
        
        For non-memory errors (module errors, corrupt files, etc.), files are
        retried up to max_retries times before being permanently failed.
        
        Memory errors are handled separately by the skipped queue logic.
        
        Args:
            filepath: Path to the failed file
            worker_id: ID of the worker
            error_msg: Error message describing the failure
            max_retries: Maximum number of retry attempts before permanent failure
        '''
        # Unregister memory for this worker
        try:
            self.mem_manager.unregister_worker_memory(worker_id)
        except Exception:
            pass
        
        file_hash = self._get_file_hash(filepath)
        claim_file = self.claimed_dir / file_hash
        failed_file = self.failed_dir / file_hash
        retry_file = self.queue_dir / 'retries' / file_hash
        
        # Ensure retry directory exists
        (self.queue_dir / 'retries').mkdir(exist_ok=True)
        
        # Check if this is an out-of-memory error during processing
        is_memory_error = 'memory' in error_msg.lower() or 'memoryerror' in error_msg.lower()
        
        if is_memory_error:
            # Move to skipped queue for retry when memory is available
            skip_file = self.skipped_dir / file_hash
            try:
                with open(skip_file, 'w') as f:
                    f.write(f"{filepath}\n{error_msg}\n{time.time()}\n")
                print(f"[Worker {worker_id}] Moved to retry queue (OOM): {os.path.basename(filepath)}")
            except (IOError, OSError):
                pass
            
            # Remove claim file
            try:
                if claim_file.exists():
                    claim_file.unlink()
            except (IOError, OSError):
                pass
            return
        
        # For non-memory errors, use retry counting
        retry_count = 0
        try:
            if retry_file.exists():
                with open(retry_file, 'r') as f:
                    retry_count = int(f.read().strip())
        except (IOError, OSError, ValueError):
            pass
        
        retry_count += 1
        
        if retry_count >= max_retries:
            # Move to permanently failed
            try:
                with open(failed_file, 'w') as f:
                    f.write(f"worker_{worker_id}\n{filepath}\n{time.time()}\n{error_msg}\nretries:{retry_count}\n")
                print(f"[Worker {worker_id}] File permanently failed after {retry_count} attempts: {os.path.basename(filepath)}")
                # Clean up retry file
                if retry_file.exists():
                    retry_file.unlink()
            except (IOError, OSError):
                pass
        else:
            # Record retry count, file will be picked up again
            try:
                with open(retry_file, 'w') as f:
                    f.write(str(retry_count))
                print(f"[Worker {worker_id}] File failed (attempt {retry_count}/{max_retries}): {os.path.basename(filepath)}")
            except (IOError, OSError):
                pass
        
        # Remove claim file
        try:
            if claim_file.exists():
                claim_file.unlink()
        except (IOError, OSError):
            pass


class ProcessingStats:
    '''Track per-worker processing statistics with robust file I/O.'''
    
    def __init__(self, stats_file: str):
        '''Initialize stats tracker.
        
        Args:
            stats_file: Path to JSON stats file
        '''
        self.stats_file = Path(stats_file)
        self.stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.start_time = time.time()
        self.files_processed = 0
        self.files_failed = 0
        self.bytes_read = 0
        self.bytes_written = 0
        self.galaxies_extracted = 0
        self.mode = 'full'
        self.last_file = ''
        self.errors = []
    
    def update(self, bytes_read: int = 0, bytes_written: int = 0, galaxies: int = 0, 
               filename: str = '', error: str = None):
        '''Update statistics.'''
        self.files_processed += 1
        self.bytes_read += bytes_read
        self.bytes_written += bytes_written
        self.galaxies_extracted += galaxies
        self.last_file = filename
        
        if error:
            self.files_failed += 1
            self.errors.append({
                'file': filename,
                'error': error[:500],  # Truncate long errors
                'time': datetime.now().isoformat()
            })
            # Keep only last 100 errors
            self.errors = self.errors[-100:]
        
        self._save()
    
    def _save(self):
        '''Save statistics to JSON file with error handling.'''
        elapsed = time.time() - self.start_time
        
        stats = {
            'start_time': self.start_time,
            'elapsed_seconds': elapsed,
            'files_processed': self.files_processed,
            'files_failed': self.files_failed,
            'bytes_read': self.bytes_read,
            'bytes_written': self.bytes_written,
            'galaxies_extracted': self.galaxies_extracted,
            'stage': self.mode,
            'last_file': self.last_file,
            'read_bandwidth_mbps': (self.bytes_read / (1024**2)) / elapsed if elapsed > 0 else 0,
            'write_bandwidth_mbps': (self.bytes_written / (1024**2)) / elapsed if elapsed > 0 else 0,
            'last_update': datetime.now().isoformat(),
            'recent_errors': self.errors[-10:]
        }
        
        try:
            # Write to temp file first, then rename
            temp_file = self.stats_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(stats, f, indent=2)
            temp_file.replace(self.stats_file)
        except (IOError, OSError) as e:
            print(f"Warning: Failed to save stats: {e}")


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
) -> Tuple[int, int, Optional[str]]:
    '''Process a single FITS file with comprehensive error handling.
    
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
        timeout_seconds: Maximum seconds to process a single file
    
    Returns:
        Tuple of (file_size, galaxies_extracted, error_message or None)
    '''
    from galaxy_naming import round_coordinate
    import sep_helpers as sh
    from waveband_extraction import extract_waveband_from_filename
    from visualization import create_visualization
    
    error_message = None
    
    try:
        filename = os.path.basename(filepath)
        
        # Check file exists and get size
        if not os.path.exists(filepath):
            return 0, 0, f"File not found: {filepath}"
        
        file_size = os.path.getsize(filepath)
        
        if file_size == 0:
            return 0, 0, f"Empty file: {filepath}"
        
        if verbosity >= 1:
            print(f"Processing {filename}...")
        
        # Load FITS with error handling
        try:
            image_meta_data = sh.get_relevant_fits_meta_data(filepath)
        except Exception as e:
            return file_size, 0, f"Failed to read FITS metadata: {str(e)}"
        
        try:
            image_data = sh.get_main_fits_data(filepath)
        except MemoryError:
            return file_size, 0, f"Out of memory loading FITS data"
        except Exception as e:
            return file_size, 0, f"Failed to read FITS data: {str(e)}"
        
        # Validate image data
        if image_data is None or image_data.size == 0:
            return file_size, 0, f"Invalid or empty image data"
        
        # Extract waveband from filename
        waveband = extract_waveband_from_filename(filename)
        
        # Run SEP detection with error handling
        try:
            bkg = sh.get_image_background(image_data)
            bkgless_data = sh.subtract_bkg(image_data)
            celestial_objects = sh.extract_objects(bkgless_data, bkg, pixel_stack_limit=5000000, sub_object_limit=50000)
        except MemoryError:
            return file_size, 0, f"Out of memory during SEP extraction"
        except Exception as e:
            return file_size, 0, f"SEP extraction failed: {str(e)}"
        
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
            
            # In crop-only mode, don't add to DB again (to avoid duplicates)
            if mode == 'crop-only':
                # Look up existing galaxy instance by coordinates and source file
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
            
            # Always count galaxies (for discovery/scan-only mode tracking)
            galaxies_saved += 1
            
            if verbosity >= 2:
                if mode != 'scan-only':
                    print(f"  Saved: {output_filename} (group {group_id}, v{version})")
                else:
                    print(f"  Found: group {group_id}, v{version} at RA={canonical_ra:.6f}, DEC={canonical_dec:.6f}")
            
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
            
            try:
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
            except Exception as e:
                if verbosity >= 1:
                    print(f"  Warning: Visualization failed: {e}")
        
        if verbosity >= 1:
            if mode == 'scan-only':
                msg = f"[Worker] Scanned {filename} ({galaxies_saved} galaxies discovered"
            else:
                msg = f"[Worker] Completed {filename} ({galaxies_saved} galaxies"
            if duplicates_skipped > 0:
                msg += f", {duplicates_skipped} duplicates skipped"
            msg += ")"
            print(msg)
        
        return file_size, galaxies_saved, None 
        
    except MemoryError as e:
        error_message = f"Out of memory: {str(e)}"
        print(f"ERROR processing {filepath}: {error_message}")
        import gc
        gc.collect() 
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0, 0, error_message
        
    except Exception as e:
        error_message = f"{type(e).__name__}: {str(e)}"
        print(f"ERROR processing {filepath}: {error_message}")
        if verbosity >= 2:
            traceback.print_exc()
        return os.path.getsize(filepath) if os.path.exists(filepath) else 0, 0, error_message


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
    max_memory_gb: float = 80.0,
    max_retries: int = 3
) -> int:
    '''Main worker process loop with robust error handling.
    
    Returns:
        Exit code: 0 = normal completion, 1 = error, 2 = no work available
    '''
    global _shutdown_requested
    
    print(f"[Worker {worker_id}] Starting (PID: {os.getpid()})...")
    
    # Initialize database connection with retry logic
    db = None
    for attempt in range(3):
        try:
            db = CoordinateDatabase(db_path, position_tolerance_arcsec=0.5)
            break
        except Exception as e:
            print(f"[Worker {worker_id}] Database connection attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"[Worker {worker_id}] Failed to connect to database after 3 attempts")
                return 1
    
    # Initialize stats with mode/stage
    stats = ProcessingStats(stats_file)
    stats.mode = mode
    
    # Get work queue with memory management
    try:
        all_files = sorted(glob.glob(os.path.join(input_dir, '*.fits')))
        if not all_files:
            print(f"[Worker {worker_id}] No FITS files found in {input_dir}")
            return 2
        work_queue = WorkQueue(queue_dir, stats_dir, all_files, max_memory_gb=max_memory_gb)
    except Exception as e:
        print(f"[Worker {worker_id}] Failed to initialize work queue: {e}")
        return 1
    
    files_processed = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    idle_iterations = 0
    max_idle_iterations = 60  
    
    while not _shutdown_requested:
        try:
            # Claim next file
            filepath = work_queue.claim_next_file(worker_id)
            
            if filepath is None:
                idle_iterations += 1
                
                # Check queue status to see if we're truly done
                status = work_queue.get_queue_status()
                if status['remaining'] == 0 and status['skipped'] == 0:
                    print(f"[Worker {worker_id}] All work completed")
                    break
                
                if idle_iterations >= max_idle_iterations:
                    print(f"[Worker {worker_id}] No work available after {idle_iterations} attempts, "
                          f"status: {status}")
                    if status['remaining'] == 0:
                        break
                    idle_iterations = 0
                
                time.sleep(1)
                continue
            
            idle_iterations = 0  # Reset idle counter
            
            # Process file
            start_time = time.time()
            filename = os.path.basename(filepath)
            
            try:
                bytes_read, galaxies, error = process_single_fits(
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
                
                if error:
                    # File processing failed
                    work_queue.mark_failed(filepath, worker_id, error, max_retries=max_retries)
                    stats.update(bytes_read=bytes_read, galaxies=0, filename=filename, error=error)
                    consecutive_errors += 1
                    
                    print(f"[Worker {worker_id}] Failed {filename}: {error}")
                    
                    # Check if we're having too many consecutive errors
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"[Worker {worker_id}] Too many consecutive errors ({consecutive_errors}), "
                              f"pausing for 30 seconds...")
                        time.sleep(30)
                        consecutive_errors = 0
                else:
                    # Success
                    stats.update(bytes_read=bytes_read, galaxies=galaxies, filename=filename)
                    files_processed += 1
                    consecutive_errors = 0 
                    
                    # Mark as completed
                    work_queue.mark_completed(filepath, worker_id)
                    
                    # Progress message
                    bandwidth_mbps = (bytes_read / (1024**2)) / elapsed if elapsed > 0 else 0
                    print(f"[Worker {worker_id}] Completed {filename} "
                          f"({elapsed:.1f}s, {bytes_read/(1024**2):.1f}MB, {galaxies} galaxies) | "
                          f"Rate: {bandwidth_mbps:.2f} MB/s")
                
            except MemoryError:
                # Release memory and mark as failed
                import gc
                gc.collect()
                
                work_queue.mark_failed(filepath, worker_id, "Out of memory", max_retries=max_retries)
                stats.update(bytes_read=0, galaxies=0, filename=filename, error="Out of memory")
                consecutive_errors += 1
                
                print(f"[Worker {worker_id}] Out of memory processing {filename}, pausing 10s...")
                time.sleep(10)
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                work_queue.mark_failed(filepath, worker_id, error_msg, max_retries=max_retries)
                stats.update(bytes_read=0, galaxies=0, filename=filename, error=error_msg)
                consecutive_errors += 1
                
                print(f"[Worker {worker_id}] Exception processing {filename}: {error_msg}")
                if verbosity >= 2:
                    traceback.print_exc()
                    
        except KeyboardInterrupt:
            print(f"[Worker {worker_id}] Interrupted")
            break
            
        except Exception as e:
            print(f"[Worker {worker_id}] Unexpected error in main loop: {e}")
            if verbosity >= 2:
                traceback.print_exc()
            consecutive_errors += 1
            time.sleep(5) 
    
    print(f"[Worker {worker_id}] Finished. Processed {files_processed} files.")
    return 0


def worker_wrapper(worker_id: int, args_dict: dict, result_queue: mp.Queue):
    """Wrapper function for worker process that reports completion status."""
    try:
        exit_code = worker_process(
            worker_id=worker_id,
            **args_dict
        )
        result_queue.put((worker_id, 'completed', exit_code))
    except KeyboardInterrupt:
        result_queue.put((worker_id, 'interrupted', 0))
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[Worker {worker_id}] Fatal error: {error_msg}")
        traceback.print_exc()
        result_queue.put((worker_id, 'crashed', error_msg))


def supervisor_process(
    num_workers: int,
    input_dir: str,
    output_dir: str,
    db_path: str,
    queue_dir: str,
    stats_dir: str,
    min_size: int = 20,
    max_size: int = 100,
    verbosity: int = 1,
    mode: str = 'full',
    visualize_crops: bool = False,
    viz_output_dir: Optional[str] = None,
    max_memory_gb: float = 80.0,
    respawn_delay: float = 5.0,
    max_respawns_per_worker: int = 10
):
    """Supervisor that maintains the specified number of workers.
    
    Automatically respawns workers that crash or exit unexpectedly to maintain
    the target worker count. Stops only when all work is complete.
    
    Args:
        num_workers: Target number of concurrent workers
        respawn_delay: Seconds to wait before respawning a crashed worker
        max_respawns_per_worker: Maximum times to respawn a worker before giving up
    """
    global _shutdown_requested
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"[Supervisor] Starting with {num_workers} workers")
    print(f"[Supervisor] Input: {input_dir}")
    print(f"[Supervisor] Output: {output_dir}")
    print(f"[Supervisor] Mode: {mode}")
    print(f"[Supervisor] Memory budget: {max_memory_gb}GB")
    
    # Initialize work queue to check file count
    all_files = sorted(glob.glob(os.path.join(input_dir, '*.fits')))
    if not all_files:
        print(f"[Supervisor] ERROR: No FITS files found in {input_dir}")
        return
    
    print(f"[Supervisor] Found {len(all_files)} FITS files to process")
    
    # Create shared result queue
    result_queue = mp.Queue()
    
    # Worker tracking
    workers = {}  # worker_id -> Process
    worker_respawn_counts = {}  # worker_id -> respawn count
    worker_start_times = {}  # worker_id -> start timestamp
    
    # Common args for all workers
    base_args = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'db_path': db_path,
        'queue_dir': queue_dir,
        'stats_dir': stats_dir,
        'min_size': min_size,
        'max_size': max_size,
        'verbosity': verbosity,
        'mode': mode,
        'visualize_crops': visualize_crops,
        'viz_output_dir': viz_output_dir,
        'max_memory_gb': max_memory_gb,
    }
    
    def start_worker(worker_id: int):
        """Start a new worker process."""
        stats_file = os.path.join(stats_dir, f'worker_{worker_id}_stats.json')
        args = {**base_args, 'stats_file': stats_file}
        
        p = mp.Process(
            target=worker_wrapper,
            args=(worker_id, args, result_queue),
            name=f"worker_{worker_id}"
        )
        p.start()
        workers[worker_id] = p
        worker_start_times[worker_id] = time.time()
        print(f"[Supervisor] Started worker {worker_id} (PID: {p.pid})")
    
    def check_work_remaining() -> bool:
        """Check if there's still work to do."""
        try:
            work_queue = WorkQueue(queue_dir, stats_dir, [], max_memory_gb=max_memory_gb)
            status = work_queue.get_queue_status()
            return status['remaining'] > 0 or status['skipped'] > 0
        except Exception:
            return True
    
    # Start initial workers
    for worker_id in range(num_workers):
        worker_respawn_counts[worker_id] = 0
        start_worker(worker_id)
    
    # Monitor loop
    completed_workers = set()
    
    while not _shutdown_requested:
        try:
            # Check for completed/crashed workers (non-blocking)
            while not result_queue.empty():
                try:
                    worker_id, status, info = result_queue.get_nowait()
                    
                    if status == 'completed':
                        print(f"[Supervisor] Worker {worker_id} completed normally (exit code: {info})")
                        completed_workers.add(worker_id)
                        
                    elif status == 'interrupted':
                        print(f"[Supervisor] Worker {worker_id} was interrupted")
                        completed_workers.add(worker_id)
                        
                    elif status == 'crashed':
                        print(f"[Supervisor] Worker {worker_id} crashed: {info}")
                        
                        # Check if we should respawn
                        if worker_respawn_counts[worker_id] < max_respawns_per_worker:
                            if check_work_remaining():
                                worker_respawn_counts[worker_id] += 1
                                print(f"[Supervisor] Respawning worker {worker_id} "
                                      f"(attempt {worker_respawn_counts[worker_id]}/{max_respawns_per_worker})")
                                time.sleep(respawn_delay)
                                start_worker(worker_id)
                            else:
                                print(f"[Supervisor] No work remaining, not respawning worker {worker_id}")
                                completed_workers.add(worker_id)
                        else:
                            print(f"[Supervisor] Worker {worker_id} exceeded max respawns, giving up")
                            completed_workers.add(worker_id)
                            
                except Exception as e:
                    print(f"[Supervisor] Error reading result queue: {e}")
                    break
            
            # Check for workers that died without reporting
            for worker_id, process in list(workers.items()):
                if worker_id in completed_workers:
                    continue
                    
                if not process.is_alive():
                    exit_code = process.exitcode
                    runtime = time.time() - worker_start_times.get(worker_id, time.time())
                    
                    if exit_code != 0:
                        print(f"[Supervisor] Worker {worker_id} died unexpectedly "
                              f"(exit code: {exit_code}, runtime: {runtime:.1f}s)")
                        
                        # Respawn if work remains
                        if worker_respawn_counts[worker_id] < max_respawns_per_worker:
                            if check_work_remaining():
                                worker_respawn_counts[worker_id] += 1
                                print(f"[Supervisor] Respawning worker {worker_id} "
                                      f"(attempt {worker_respawn_counts[worker_id]}/{max_respawns_per_worker})")
                                time.sleep(respawn_delay)
                                start_worker(worker_id)
                            else:
                                completed_workers.add(worker_id)
                        else:
                            print(f"[Supervisor] Worker {worker_id} exceeded max respawns")
                            completed_workers.add(worker_id)
                    else:
                        completed_workers.add(worker_id)
            
            # Check if all workers are done
            if len(completed_workers) >= num_workers:
                if not check_work_remaining():
                    print("[Supervisor] All workers completed and no work remaining")
                    break
                else:
                    # Try to restart workers
                    print("[Supervisor] Work remains but no active workers, restarting...")
                    for worker_id in range(num_workers):
                        if worker_id in completed_workers:
                            if worker_respawn_counts[worker_id] < max_respawns_per_worker:
                                completed_workers.discard(worker_id)
                                worker_respawn_counts[worker_id] += 1
                                start_worker(worker_id)
                    time.sleep(5)
            
            # Periodic status report
            active_count = sum(1 for w in workers.values() if w.is_alive())
            if active_count < num_workers and verbosity >= 1:
                work_status = check_work_remaining()
                print(f"[Supervisor] Active workers: {active_count}/{num_workers}, "
                      f"work remaining: {work_status}")
            
            time.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\n[Supervisor] Received interrupt, shutting down workers...")
            break
    
    # Graceful shutdown
    print("[Supervisor] Initiating shutdown...")
    _shutdown_requested = True
    
    # Give workers time to finish current file
    shutdown_timeout = 60
    shutdown_start = time.time()
    
    while time.time() - shutdown_start < shutdown_timeout:
        active = [w for w in workers.values() if w.is_alive()]
        if not active:
            break
        print(f"[Supervisor] Waiting for {len(active)} workers to finish...")
        time.sleep(5)
    
    # Force kill any remaining workers
    for worker_id, process in workers.items():
        if process.is_alive():
            print(f"[Supervisor] Force terminating worker {worker_id}")
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
    
    # Final status
    try:
        work_queue = WorkQueue(queue_dir, stats_dir, [], max_memory_gb=max_memory_gb)
        status = work_queue.get_queue_status()
        print(f"\n[Supervisor] Final status:")
        print(f"  Total files: {status['total']}")
        print(f"  Completed: {status['completed']}")
        print(f"  Failed: {status['failed']}")
        print(f"  Remaining: {status['remaining']}")
    except Exception as e:
        print(f"[Supervisor] Could not get final status: {e}")
    
    # Print final galaxy count from database
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM galaxy_groups")
        unique_galaxies = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM galaxy_instances")
        total_instances = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"\n[Supervisor] Galaxy discovery results:")
        print(f"  Unique galaxy positions: {unique_galaxies:,}")
        print(f"  Total instances (across wavebands): {total_instances:,}")
    except Exception as e:
        print(f"[Supervisor] Could not read galaxy count from database: {e}")
    
    print("\n[Supervisor] Shutdown complete")


def main():
    '''Main entry point.'''
    parser = argparse.ArgumentParser(
        description='Parallel galaxy extractor for JWST FITS files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with 7 parallel workers (recommended)
  python parallel_galaxy_extractor.py --num-workers 7 --input ./output --output ./galaxy_crops \\
      --db-path galaxy_coords.db --queue-dir work_queue --stats-dir stats --mode scan-only

  # Legacy single worker mode
  python parallel_galaxy_extractor.py --worker-id 0 --input ./output --output ./galaxy_crops \\
      --db-path galaxy_coords.db --queue-dir work_queue --stats-dir stats
'''
    )
    
    # Worker specification
    worker_group = parser.add_mutually_exclusive_group(required=True)
    worker_group.add_argument('--num-workers', type=int,
                              help='Number of parallel workers to run (recommended)')
    worker_group.add_argument('--worker-id', type=int,
                              help='Legacy: Unique worker ID (0, 1, 2, ...) for single worker mode')
    
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
    parser.add_argument('--visualize-crops', default=False, action='store_true',
                        help='Generate PNG visualizations of detected galaxies with circles marking crop regions')
    parser.add_argument('--max-memory-gb', type=float, default=80.0,
                        help='Global memory budget in GB (default: 80)')
    parser.add_argument('--max-respawns', type=int, default=10,
                        help='Maximum times to respawn a crashed worker (default: 10)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    if args.mode != 'crop-only' and args.output != '/dev/null':
        os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.stats_dir, exist_ok=True)
    os.makedirs(args.queue_dir, exist_ok=True)
    
    # Create visualization output directory if needed
    viz_output_dir = None
    if args.visualize_crops:
        viz_output_dir = os.path.join(args.output, 'visualizations')
        os.makedirs(viz_output_dir, exist_ok=True)
    
    if args.num_workers is not None:
        if args.num_workers < 1:
            print("Error: --num-workers must be at least 1")
            sys.exit(1)
        
        supervisor_process(
            num_workers=args.num_workers,
            input_dir=args.input,
            output_dir=args.output,
            db_path=args.db_path,
            queue_dir=args.queue_dir,
            stats_dir=args.stats_dir,
            min_size=args.min_size,
            max_size=args.max_size,
            verbosity=args.verbosity,
            mode=args.mode,
            visualize_crops=args.visualize_crops,
            viz_output_dir=viz_output_dir,
            max_memory_gb=args.max_memory_gb,
            max_respawns_per_worker=args.max_respawns
        )
    else:
        # old single worker mode
        stats_file = os.path.join(args.stats_dir, f'worker_{args.worker_id}_stats.json')
        
        exit_code = worker_process(
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
            max_memory_gb=args.max_memory_gb
        )
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
