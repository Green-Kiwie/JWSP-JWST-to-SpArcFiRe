from pathlib import Path
from astropy.io import fits
import numpy as np
from PIL import Image

def fits_to_jpg(input_dir: str, output_dir: str = None, quality: int = 95):
    """
    Convert all .fits files in a directory to .jpg.

    Args:
        input_dir:  Path to the folder containing .fits files.
        output_dir: Where to save .jpg files (defaults to same folder as input).
        quality:    JPEG quality (1–95).
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    fits_files = list(input_path.glob("*.fits")) + list(input_path.glob("*.fit"))
    if not fits_files:
        print("No .fits files found.")
        return

    for fits_file in fits_files:
        try:
            with fits.open(fits_file) as hdul:
                # Use the first image-like HDU
                data = next(
                    hdu.data for hdu in hdul
                    if hdu.data is not None and hdu.data.ndim >= 2
                )

            # Handle multi-channel (e.g. 3D cube): take first frame
            if data.ndim == 3:
                data = data[0]

            # Normalize to 0–255
            data = data.astype(np.float64)
            data -= np.nanmin(data)
            max_val = np.nanmax(data)
            if max_val > 0:
                data /= max_val
            data = (data * 255).astype(np.uint8)

            img = Image.fromarray(data, mode="L")  # Grayscale
            out_file = output_path / (fits_file.stem + ".jpg")
            img.save(out_file, "JPEG", quality=quality)
            print(f"Saved: {out_file}")

        except Exception as e:
            print(f"Failed to convert {fits_file.name}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert FITS files to JPG format.")
    parser.add_argument("input_dir", help="Directory containing .fits files")
    parser.add_argument("-o", "--output_dir", help="Directory to save .jpg files (defaults to input directory)")
    parser.add_argument("-q", "--quality", type=int, default=95, help="JPEG quality (1–95)")

    args = parser.parse_args()
    fits_to_jpg(args.input_dir, args.output_dir, args.quality)