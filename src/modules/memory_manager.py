"""
Global memory manager for parallel galaxy extraction.

Tracks memory usage across all workers and enforces a global memory budget.
This prevents OOM by ensuring the combined memory usage of all workers
never exceeds the specified limit.

Memory estimation: file_size * 4 (raw data + 3x working buffers for SEP operations)
"""

import json
import os
from pathlib import Path
from typing import Tuple
import time


class MemoryManager:
    """Manages global memory usage across all workers.
    
    Uses a JSON file to track which worker is processing which file and
    how much memory it's using. Before a worker claims a file, it checks
    if the estimated memory would exceed the global budget.
    """
    
    def __init__(self, stats_dir: str, max_total_memory_gb: float = 80.0):
        """Initialize memory manager.
        
        Args:
            stats_dir: Directory to store memory_usage.json tracking file
            max_total_memory_gb: Global memory budget in GB
        """
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_file = self.stats_dir / "memory_usage.json"
        self.max_total_memory_bytes = int(max_total_memory_gb * 1024 * 1024 * 1024)
        
        # Initialize memory file if doesn't exist
        if not self.memory_file.exists():
            self._write_memory_file({})
    
    def _read_memory_file(self) -> dict:
        """Read current memory usage from JSON file."""
        try:
            if self.memory_file.exists():
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}
    
    def _write_memory_file(self, data: dict):
        """Write memory usage to JSON file."""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass
    
    def estimate_file_memory_usage(self, filepath: str, max_image_memory_gb: float = 20.0) -> int:
        """Estimate memory needed to process a file with downsampling.
        
        For files >20GB, downsampling reduces memory to 20GB * 4 = 80GB
        For files <=20GB, memory = file_size * 4 (raw + 3x working buffers)
        
        Args:
            filepath: Path to FITS file
            max_image_memory_gb: Max memory for raw image (GB) before downsampling
            
        Returns:
            Estimated memory in bytes
        """
        try:
            file_size = os.path.getsize(filepath)
            max_image_bytes = int(max_image_memory_gb * 1024 * 1024 * 1024)
            
            # If file exceeds max, it will be downsampled to max_image_memory_gb
            if file_size > max_image_bytes:
                estimated_memory = max_image_bytes * 4
            else:
                # 4x multiplier: raw data + 3 working buffer copies for SEP operations
                estimated_memory = file_size * 4
            
            return estimated_memory
        except (OSError, FileNotFoundError):
            return 0
    
    def get_active_memory_usage(self) -> int:
        """Get total memory currently allocated to all workers.
        
        Returns:
            Total memory in bytes
        """
        memory_data = self._read_memory_file()
        total = 0
        
        for worker_id, entries in memory_data.items():
            if isinstance(entries, dict):
                if 'memory_bytes' in entries:
                    total += entries['memory_bytes']
        
        return total
    
    def get_available_memory(self) -> int:
        """Get available memory within budget.
        
        Returns:
            Available memory in bytes
        """
        active = self.get_active_memory_usage()
        available = max(0, self.max_total_memory_bytes - active)
        return available
    
    def can_process_file(self, filepath: str) -> Tuple[bool, str]:
        """Check if a file can be processed within memory budget.
        
        Args:
            filepath: Path to FITS file
            
        Returns:
            Tuple of (can_process: bool, reason: str)
        """
        estimated_mem = self.estimate_file_memory_usage(filepath)
        available_mem = self.get_available_memory()
        
        if estimated_mem <= 0:
            return False, f"Cannot determine file size: {filepath}"
        
        if estimated_mem <= available_mem:
            return True, "Within budget"
        
        active = self.get_active_memory_usage()
        budget_gb = self.max_total_memory_bytes / (1024**3)
        active_gb = active / (1024**3)
        needed_gb = estimated_mem / (1024**3)
        
        return False, (
            f"Insufficient memory: need {needed_gb:.1f}GB, "
            f"have {budget_gb - active_gb:.1f}GB available "
            f"({active_gb:.1f}GB/{budget_gb:.1f}GB in use)"
        )
    
    def register_worker_memory(self, worker_id: int, filepath: str, memory_bytes: int):
        """Register memory allocation for a worker processing a file.
        
        Args:
            worker_id: ID of worker
            filepath: Path to file being processed
            memory_bytes: Estimated memory usage
        """
        memory_data = self._read_memory_file()
        
        worker_key = f"worker_{worker_id}"
        
        # Use filepath as key (one file per worker at a time)
        memory_data[worker_key] = {
            "filepath": filepath,
            "memory_bytes": memory_bytes,
            "timestamp": time.time()
        }
        
        self._write_memory_file(memory_data)
    
    def unregister_worker_memory(self, worker_id: int):
        """Unregister memory allocation when worker finishes.
        
        Args:
            worker_id: ID of worker
        """
        memory_data = self._read_memory_file()
        
        worker_key = f"worker_{worker_id}"
        if worker_key in memory_data:
            del memory_data[worker_key]
        
        self._write_memory_file(memory_data)
    
    def get_memory_status(self) -> dict:
        """Get diagnostic memory status information.
        
        Returns:
            Dictionary with current memory usage details
        """
        active = self.get_active_memory_usage()
        available = self.get_available_memory()
        budget = self.max_total_memory_bytes
        
        return {
            "budget_gb": budget / (1024**3),
            "active_gb": active / (1024**3),
            "available_gb": available / (1024**3),
            "usage_percent": (active / budget * 100) if budget > 0 else 0,
            "workers": self._read_memory_file()
        }
