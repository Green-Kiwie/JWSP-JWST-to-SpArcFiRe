'''
Module for visualizing FITS images with detected galaxies marked with circles.

Generates PNG visualizations showing the full all-sky FITS image with circles
marking the positions and sizes of detected and cropped galaxies.
'''

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def normalize_image_data(image_data: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    '''Normalize image data to 0-255 for visualization.
    
    Uses percentile-based normalization to handle astronomical data with
    bright sources and faint backgrounds.
    
    Args:
        image_data: 2D numpy array of image data
        vmin: Minimum value for normalization (if None, uses 5th percentile)
        vmax: Maximum value for normalization (if None, uses 95th percentile)
    
    Returns:
        Normalized 8-bit image data (0-255)
    '''
    if image_data.size == 0:
        return np.zeros_like(image_data, dtype=np.uint8)
    
    # Handle NaN and inf values
    data = np.copy(image_data)
    data[~np.isfinite(data)] = np.nanmedian(data[np.isfinite(data)]) if np.any(np.isfinite(data)) else 0
    
    # Set normalization bounds
    if vmin is None:
        vmin = np.percentile(data, 5)
    if vmax is None:
        vmax = np.percentile(data, 95)
    
    # Normalize to 0-1
    if vmax > vmin:
        normalized = (data - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(data)
    
    # Clip and convert to 8-bit
    normalized = np.clip(normalized, 0, 1)
    return (normalized * 255).astype(np.uint8)


def create_visualization(
    image_data: np.ndarray,
    galaxy_detections: List[Tuple[float, float, float, float]],
    output_path: str,
    max_dimension: int = 2048,
    circle_color: Tuple[int, int, int] = (0, 255, 0),
    circle_width: int = 2,
    add_labels: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> bool:
    '''Create PNG visualization of FITS image with galaxy circles.
    
    Args:
        image_data: 2D numpy array of FITS image data
        galaxy_detections: List of (x_pixel, y_pixel, width, height) tuples for detected galaxies
        output_path: Path to save output PNG file
        max_dimension: Maximum dimension for output image (larger images are downsampled)
        circle_color: RGB color tuple for circles (default: green)
        circle_width: Line width for circles in pixels
        add_labels: Whether to add text labels to circles (default: False)
        vmin: Minimum value for image normalization
        vmax: Maximum value for image normalization
    
    Returns:
        True if successful, False otherwise
    '''
    if not PIL_AVAILABLE:
        print("Warning: PIL not available. Install with: pip install pillow")
        return False
    
    if image_data.size == 0:
        print("Error: Empty image data")
        return False
    
    try:
        # Normalize image to 8-bit grayscale
        normalized_data = normalize_image_data(image_data, vmin, vmax)
        
        # Downsample if too large
        height, width = normalized_data.shape
        scale_factor = 1.0
        if max(height, width) > max_dimension:
            scale_factor = max_dimension / max(height, width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            normalized_data = np.array(Image.fromarray(normalized_data).resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            ))
        
        # Convert to RGB for color circles
        rgb_image = np.stack([normalized_data] * 3, axis=2)
        pil_image = Image.fromarray(rgb_image, mode='RGB')
        draw = ImageDraw.Draw(pil_image)
        
        # Draw circles for each detection
        for idx, (x_pix, y_pix, gal_width, gal_height) in enumerate(galaxy_detections):
            # Scale to match downsampled image
            x_scaled = x_pix * scale_factor
            y_scaled = y_pix * scale_factor
            radius_scaled = max(gal_width, gal_height) / 2.0 * scale_factor
            
            # Draw circle as bounding box for ellipse
            x0 = x_scaled - radius_scaled
            y0 = y_scaled - radius_scaled
            x1 = x_scaled + radius_scaled
            y1 = y_scaled + radius_scaled
            
            draw.ellipse([x0, y0, x1, y1], outline=circle_color, width=circle_width)
            
            # Add label if requested
            if add_labels:
                draw.text((x_scaled + radius_scaled + 5, y_scaled), f"#{idx+1}", fill=circle_color)
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(output_path, quality=95)
        
        return True
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        return False


def create_multiplot_visualization(
    image_data: np.ndarray,
    galaxy_detections: List[Tuple[float, float, float, float, int]],
    output_path: str,
    max_dimension: int = 2048,
    grid_columns: int = 3
) -> bool:
    '''Create visualization with detected galaxies in a grid layout.
    
    Args:
        image_data: 2D numpy array of FITS image data
        galaxy_detections: List of (x_pixel, y_pixel, width, height, galaxy_id) tuples
        output_path: Path to save output PNG file
        max_dimension: Maximum dimension for full image
        grid_columns: Number of columns in galaxy grid
    
    Returns:
        True if successful, False otherwise
    '''
    if not PIL_AVAILABLE:
        return False
    
    try:
        # This would create a more complex visualization with the full image
        # plus a grid of individual galaxy crops below it
        # For now, just use the simpler single-image version
        return create_visualization(
            image_data,
            [(x, y, w, h) for x, y, w, h, _ in galaxy_detections],
            output_path,
            max_dimension=max_dimension
        )
    except Exception as e:
        print(f"Error creating multiplot visualization: {str(e)}")
        return False


if __name__ == "__main__":
    # Test visualization functions
    print("Visualization module loaded successfully")
    print(f"PIL available: {PIL_AVAILABLE}")
