#!/usr/bin/env python3
"""
Real-time monitoring dashboard for parallel galaxy extraction workers.
Shows MB/s throughput, files processed, ETA, per-worker stats, and processing stage.

Usage:
    python monitor_workers.py --stats-dir count_stats --queue-dir count_queue --db-path count.db
"""

import json
import os
import sys
import argparse
import sqlite3
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

def get_database_stats(db_path):
    """Get cumulative galaxy counts from the SQLite database."""
    if not db_path or not os.path.exists(db_path):
        return None
    
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()
        
        # Get unique galaxy positions (groups)
        cursor.execute("SELECT COUNT(*) FROM galaxy_groups")
        unique_galaxies = cursor.fetchone()[0]
        
        # Get total instances (across all wavebands)
        cursor.execute("SELECT COUNT(*) FROM galaxy_instances")
        total_instances = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'unique_galaxies': unique_galaxies,
            'total_instances': total_instances
        }
    except (sqlite3.Error, Exception) as e:
        return None

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

def get_failed_files(queue_dir="prod_queue"):
    """Count permanently failed files."""
    failed_dir = Path(queue_dir) / "failed"
    if not failed_dir.exists():
        return 0
    return len(list(failed_dir.glob("*")))

def get_skipped_files(queue_dir="prod_queue"):
    """Count temporarily skipped files (waiting for memory)."""
    skipped_dir = Path(queue_dir) / "skipped"
    if not skipped_dir.exists():
        return 0
    return len(list(skipped_dir.glob("*")))

def calculate_eta(completed, total, elapsed_seconds):
    """Calculate estimated time to completion."""
    if completed == 0 or elapsed_seconds == 0:
        return "N/A"
    
    rate = completed / elapsed_seconds
    remaining = total - completed
    eta_seconds = remaining / rate if rate > 0 else 0
    
    return str(timedelta(seconds=int(eta_seconds)))

def main():
    parser = argparse.ArgumentParser(description='Monitor parallel galaxy extraction workers')
    parser.add_argument('--stats-dir', type=str, default='prod_stats',
                        help='Directory containing worker stats (default: prod_stats)')
    parser.add_argument('--queue-dir', type=str, default='prod_queue',
                        help='Directory containing work queue (default: prod_queue)')
    parser.add_argument('--db-path', type=str, default=None,
                        help='Path to SQLite database for cumulative galaxy counts (optional)')
    args = parser.parse_args()
    
    stats_dir = args.stats_dir
    queue_dir = args.queue_dir
    db_path = args.db_path
    
    # Try to auto-detect database path if not provided
    if db_path is None:
        # Common database names to check
        for candidate in ['count.db', 'prod.db', 'galaxy_coords.db']:
            if os.path.exists(candidate):
                db_path = candidate
                break
    
    start_time = time.time()
    
    print("\033[2J\033[H")  # Clear screen
    print("=" * 120)
    print("JWST Galaxy Extraction - Real-Time Monitor".center(120))
    print(f"Stats: {stats_dir} | Queue: {queue_dir} | DB: {db_path or 'not found'}".center(120))
    print("=" * 120)
    
    while True:
        try:
            elapsed = time.time() - start_time
            workers = get_worker_stats(stats_dir)
            completed = get_completed_files(queue_dir)
            total = get_total_files(queue_dir)
            claimed = get_claimed_files(queue_dir)
            failed = get_failed_files(queue_dir)
            skipped = get_skipped_files(queue_dir)
            
            # Calculate aggregate stats from current session (worker stats)
            session_bytes_read = sum(w.get('bytes_read', 0) for w in workers.values())
            session_bytes_written = sum(w.get('bytes_written', 0) for w in workers.values())
            session_galaxies = sum(w.get('galaxies_extracted', 0) for w in workers.values())
            session_files_failed = sum(w.get('files_failed', 0) for w in workers.values())
            
            # Get cumulative totals from database (the TRUE totals)
            db_stats = get_database_stats(db_path)
            
            # Throughput: sum of individual worker MB/s (current session)
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
            print(f"Skipped (memory): {skipped} files")
            print(f"Failed: {failed} files")
            print(f"Remaining: {total - completed - failed} files")
            print(f"Elapsed: {timedelta(seconds=int(elapsed))}")
            print(f"ETA: {calculate_eta(completed, total - failed, elapsed)}")
            
            print("\n" + "=" * 120)
            print("BANDWIDTH (Current Session)".center(120))
            print("=" * 120)
            print(f"Read Throughput: {read_mbps:.2f} MB/s")
            print(f"Write Throughput: {write_mbps:.2f} MB/s")
            print(f"Session Read: {format_size(session_bytes_read)}")
            print(f"Session Written: {format_size(session_bytes_written)}")
            
            print("\n" + "=" * 120)
            print("GALAXIES (Cumulative from Database)".center(120))
            print("=" * 120)
            if db_stats:
                print(f"Unique Galaxy Positions: {db_stats['unique_galaxies']:,}")
                print(f"Total Instances (all wavebands): {db_stats['total_instances']:,}")
            else:
                print(f"Session Galaxies: {session_galaxies:,} (DB not available for cumulative total)")
                print(f"  Tip: Use --db-path count.db to see cumulative totals")
            
            print("\n" + "=" * 120)
            print("PER-WORKER STATS".center(120))
            print("=" * 120)
            
            if not workers:
                print("No worker stats yet...")
            else:
                print(f"{'Worker':<12} {'Stage':<12} {'Files':<10} {'Failed':<10} {'Galaxies':<15} {'Read MB/s':<15} {'Status':<20}")
                print("-" * 120)
                
                for worker_name in sorted(workers.keys()):
                    w = workers[worker_name]
                    worker_id = worker_name.split('_')[1]
                    files = w.get('files_processed', 0)
                    files_failed = w.get('files_failed', 0)
                    galaxies = w.get('galaxies_extracted', 0)
                    elapsed_w = w.get('elapsed_seconds', 0)
                    bytes_read = w.get('bytes_read', 0)
                    mbps = (bytes_read / elapsed_w / 1024 / 1024) if elapsed_w > 0 else 0
                    stage = w.get('stage', 'unknown')
                    
                    # Format stage display
                    if stage == 'scan-only':
                        stage_display = "Scan"
                    elif stage == 'crop-only':
                        stage_display = "Crop"
                    elif stage == 'full':
                        stage_display = "Full"
                    else:
                        stage_display = "Unknown"
                    
                    # Check if worker is active
                    last_update = w.get('last_update', '')
                    if last_update:
                        try:
                            last_update_dt = datetime.fromisoformat(last_update)
                            time_since = (datetime.now() - last_update_dt).total_seconds()
                            if time_since < 60:
                                status = "🟢 Fresh"
                            elif time_since < 120:
                                status = "🟡 Working"
                            else:
                                status = "🔴 Stalled"
                        except (ValueError, TypeError):
                            status = "⚪ Waiting"
                    else:
                        status = "⚪ Waiting"
                    
                    print(f"Worker {worker_id:<5} {stage_display:<12} {files:<10} {files_failed:<10} {galaxies:<15,} {mbps:<15.2f} {status:<20}")
            
            # Show recent errors if any
            all_errors = []
            for w in workers.values():
                all_errors.extend(w.get('recent_errors', []))
            
            if all_errors:
                # Sort by time and get last 5
                all_errors.sort(key=lambda x: x.get('time', ''), reverse=True)
                recent = all_errors[:5]
                
                print("\n" + "=" * 120)
                print("RECENT ERRORS".center(120))
                print("=" * 120)
                for err in recent:
                    filename = os.path.basename(err.get('file', 'unknown'))[:40]
                    error_msg = err.get('error', 'unknown')[:60]
                    err_time = err.get('time', '')[:19]
                    print(f"  [{err_time}] {filename}: {error_msg}")
            
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