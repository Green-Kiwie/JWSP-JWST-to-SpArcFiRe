#!/usr/bin/env python3
"""
Real-time monitoring dashboard for parallel galaxy extraction workers.
Shows MB/s throughput, files processed, ETA, per-worker stats, and processing stage.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time

def format_size(bytes_val):
    """Format bytes to human readable."""
    for unit in ['B', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}PB"

def get_worker_stats(stats_dir="prod_stats"):
    """Read all worker statistics."""
    stats_dir = Path(stats_dir)
    if not stats_dir.exists():
        return {}
    
    workers = {}
    for stat_file in sorted(stats_dir.glob("worker_*_stats.json")):
        try:
            with open(stat_file) as f:
                workers[stat_file.stem] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    return workers

def get_completed_files(queue_dir="prod_queue"):
    """Count completed files."""
    completed_dir = Path(queue_dir) / "completed"
    if not completed_dir.exists():
        return 0
    return len(list(completed_dir.glob("*")))

def get_total_files(queue_dir="prod_queue"):
    """Count total files to process."""
    work_file = Path(queue_dir) / "work_items.txt"
    if not work_file.exists():
        return 0
    with open(work_file) as f:
        return sum(1 for _ in f)

def get_claimed_files(queue_dir="prod_queue"):
    """Count files currently being processed."""
    claimed_dir = Path(queue_dir) / "claimed"
    if not claimed_dir.exists():
        return 0
    return len(list(claimed_dir.glob("*")))

def calculate_eta(completed, total, elapsed_seconds):
    """Calculate estimated time to completion."""
    if completed == 0 or elapsed_seconds == 0:
        return "N/A"
    
    rate = completed / elapsed_seconds
    remaining = total - completed
    eta_seconds = remaining / rate if rate > 0 else 0
    
    return str(timedelta(seconds=int(eta_seconds)))

def main():
    start_time = time.time()
    
    print("\033[2J\033[H")  # Clear screen
    print("=" * 120)
    print("JWST Galaxy Extraction - Real-Time Monitor".center(120))
    print("=" * 120)
    
    while True:
        try:
            elapsed = time.time() - start_time
            workers = get_worker_stats()
            completed = get_completed_files()
            total = get_total_files()
            claimed = get_claimed_files()
            
            # Calculate aggregate stats
            total_bytes_read = sum(w.get('bytes_read', 0) for w in workers.values())
            total_bytes_written = sum(w.get('bytes_written', 0) for w in workers.values())
            total_galaxies = sum(w.get('galaxies_extracted', 0) for w in workers.values())
            
            # Throughput: sum of individual worker MB/s (cumulative)
            read_mbps = sum(
                (w.get('bytes_read', 0) / w.get('elapsed_seconds', 1) / 1024 / 1024) 
                if w.get('elapsed_seconds', 0) > 0 else 0
                for w in workers.values()
            )
            write_mbps = sum(
                (w.get('bytes_written', 0) / w.get('elapsed_seconds', 1) / 1024 / 1024)
                if w.get('elapsed_seconds', 0) > 0 else 0
                for w in workers.values()
            )
            
            # Clear screen and print
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("\n" + "=" * 120)
            print("PROGRESS".center(120))
            print("=" * 120)
            print(f"Completed: {completed}/{total} files ({100*completed/total if total > 0 else 0:.1f}%)")
            print(f"In Progress: {claimed} files")
            print(f"Elapsed: {timedelta(seconds=int(elapsed))}")
            print(f"ETA: {calculate_eta(completed, total, elapsed)}")
            
            print("\n" + "=" * 120)
            print("BANDWIDTH (Cumulative)".center(120))
            print("=" * 120)
            print(f"Read Throughput: {read_mbps:.2f} MB/s")
            print(f"Write Throughput: {write_mbps:.2f} MB/s")
            print(f"Total Read: {format_size(total_bytes_read)}")
            print(f"Total Written: {format_size(total_bytes_written)}")
            
            print("\n" + "=" * 120)
            print("GALAXIES".center(120))
            print("=" * 120)
            print(f"Total Galaxies Extracted: {total_galaxies:,}")
            
            print("\n" + "=" * 120)
            print("PER-WORKER STATS".center(120))
            print("=" * 120)
            
            if not workers:
                print("No worker stats yet...")
            else:
                print(f"{'Worker':<12} {'Stage':<12} {'Files':<10} {'Galaxies':<15} {'Read MB/s':<15} {'Status':<20}")
                print("-" * 120)
                
                for worker_name in sorted(workers.keys()):
                    w = workers[worker_name]
                    worker_id = worker_name.split('_')[1]
                    files = w.get('files_processed', 0)
                    galaxies = w.get('galaxies_extracted', 0)
                    elapsed_w = w.get('elapsed_seconds', 0)
                    bytes_read = w.get('bytes_read', 0)
                    mbps = (bytes_read / elapsed_w / 1024 / 1024) if elapsed_w > 0 else 0
                    stage = w.get('stage', 'unknown')
                    
                    # Format stage display
                    if stage == 'scan-only':
                        stage_display = "📊 Scan"
                    elif stage == 'crop-only':
                        stage_display = "✂️  Crop"
                    elif stage == 'full':
                        stage_display = "🔄 Full"
                    else:
                        stage_display = "❓ Unknown"
                    
                    # Check if worker is active
                    last_update = w.get('last_update', '')
                    if last_update:
                        try:
                            last_update_dt = datetime.fromisoformat(last_update)
                            time_since = (datetime.now() - last_update_dt).total_seconds()
                            if time_since < 10:
                                status = "🟢 Active"
                            elif time_since < 30:
                                status = "🟡 Idle"
                            else:
                                status = "🔴 Stalled"
                        except (ValueError, TypeError):
                            status = "⚪ Waiting"
                    else:
                        status = "⚪ Waiting"
                    
                    print(f"Worker {worker_id:<5} {stage_display:<12} {files:<10} {galaxies:<15,} {mbps:<15.2f} {status:<20}")
            
            print("\n" + "=" * 120)
            print("Press Ctrl+C to exit | Refreshing every 2 seconds")
            print("=" * 120 + "\n")
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(2)

if __name__ == '__main__':
    main()