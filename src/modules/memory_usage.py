'''
Memory tracking utilities for monitoring records_class growth.
'''

import sys
import numpy as np

def get_size_of_object(obj) -> int:
    '''Recursively calculate the size of a Python object in bytes.'''
    size = sys.getsizeof(obj)
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += get_size_of_object(key)
            size += get_size_of_object(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            size += get_size_of_object(item)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    
    return size

def format_bytes(size_bytes: int) -> str:
    '''Convert bytes to human-readable format (B, KB, MB, GB).'''
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def get_records_class_memory(records_class) -> dict:
    '''Analyze memory usage of a Thumbnail_Handling records_class instance.
    
    Returns a dict with:
        - total_size: total memory in bytes
        - num_groups: number of unique coordinate groups
        - num_crops: total number of cropped images staged
        - breakdown: dict with per-group statistics
    '''
    
    total_size = 0
    num_groups = 0
    num_crops = 0
    breakdown = {}
    
    for key, group in records_class._records.items():
        group_size = 0
        
        # Size of key (Gal_Pos object)
        group_size += get_size_of_object(key)
        
        # Size of group dict and its metadata
        group_size += get_size_of_object(group['members'])
        group_size += get_size_of_object(group['max_dims'])
        
        # Size of all crops in this group
        crops_size = 0
        for crop_data, metadata, filename in group['crops']:
            crops_size += get_size_of_object(crop_data)
            crops_size += get_size_of_object(metadata)
            crops_size += get_size_of_object(filename)
        
        group_size += crops_size
        total_size += group_size
        num_groups += 1
        num_crops += len(group['crops'])
        
        # Store breakdown for this group
        ra, dec = key.get_pos()
        breakdown[f"RA{ra:.4f}_DEC{dec:.4f}"] = {
            'size': group_size,
            'num_crops': len(group['crops']),
            'max_dims': group['max_dims']
        }
    
    return {
        'total_size': total_size,
        'num_groups': num_groups,
        'num_crops': num_crops,
        'breakdown': breakdown
    }

def print_memory_report(records_class, file_count: int = None, processed_files: int = None) -> None:
    '''Print a formatted memory usage report for the records_class.'''
    
    memory_info = get_records_class_memory(records_class)
    total_size = memory_info['total_size']
    num_groups = memory_info['num_groups']
    num_crops = memory_info['num_crops']
    
    print(f"\n{'='*70}")
    print("MEMORY USAGE REPORT")
    print(f"{'='*70}")
    
    if processed_files is not None and file_count is not None:
        print(f"Progress: {processed_files}/{file_count} FITS files processed")
    
    print(f"Total memory used by records_class: {format_bytes(total_size)}")
    print(f"Number of coordinate groups:        {num_groups}")
    print(f"Total cropped images staged:        {num_crops}")
    
    if num_groups > 0:
        avg_size_per_group = total_size / num_groups
        print(f"Average memory per group:           {format_bytes(int(avg_size_per_group))}")
    
    if num_crops > 0:
        avg_size_per_crop = total_size / num_crops
        print(f"Average memory per crop:            {format_bytes(int(avg_size_per_crop))}")
    
    print(f"{'='*70}\n")