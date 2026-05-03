"""
Generate side-by-side SpArcFiRe composite PNGs from the logSpiralArcs-merged
outputs, mirroring the layout of the existing galaxy composite PNGs.

For each galaxy composite in galaxy_composites/<pair>/<ra>_<dec>.png, this
script finds the corresponding SpArcFiRe logSpiralArcs-merged images under
SpArcFiRe/<pair>_<telescope>/ and places a side-by-side composite at
galaxy_composites/<pair>/<ra>_<dec>-SpArcFiRe.png.

SDSS SpArcFiRe outputs use the r-band by default (configurable via SDSS_BAND).
"""

import os
import glob
from PIL import Image, ImageDraw, ImageFont

# ── Paths ───────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__),
                    "..", "outputs", "galaxy_overlap")
COMPOSITE_DIR = os.path.join(BASE, "galaxy_composites")
SPARCFIRE_DIR = os.path.join(BASE, "SpArcFiRe")

# ── SDSS band used for SpArcFiRe (u, g, r, i, z) ──────────────────────
SDSS_BAND = "r"

# ── Layout parameters (match generate_overlap_outputs.py) ──────────────
IMG_SIZE = 256
LABEL_H = 28
PADDING = 4

# ── Panel order per overlap pair (must match the original composites) ──
OVERLAP_PANELS = {
    "Hubble+SDSS+JWST": ["SDSS", "Hubble", "JWST"],
    "SDSS+JWST":        ["SDSS", "JWST"],
    "Hubble+SDSS":      ["Hubble", "SDSS"],
}


def coords_to_sparcfire_name(coord_stem):
    """Convert '149.758169_2.190735' → '149_758169_2_190735'."""
    return coord_stem.replace(".", "_")


def find_logspiral_image(pair, telescope, sf_name):
    """Return the path to the logSpiralArcs-merged PNG, or None."""
    sf_subdir = f"{pair}_{telescope}"
    sf_root = os.path.join(SPARCFIRE_DIR, sf_subdir)

    if telescope == "SDSS":
        folder = f"{sf_name}_{SDSS_BAND}"
        filename = f"{sf_name}_{SDSS_BAND}-J_logSpiralArcs-merged.png"
    else:
        folder = sf_name
        filename = f"{sf_name}-J_logSpiralArcs-merged.png"

    path = os.path.join(sf_root, folder, filename)
    return path if os.path.isfile(path) else None


def make_composite(images, labels, coord_stem):
    """Combine images into a single labelled composite (same style as originals)."""
    n = len(images)
    total_w = n * IMG_SIZE + (n - 1) * PADDING
    total_h = IMG_SIZE + LABEL_H

    composite = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = i * (IMG_SIZE + PADDING)
        resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        composite.paste(resized, (x_offset, LABEL_H))

        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_x = x_offset + (IMG_SIZE - text_w) // 2
        draw.text((text_x, 4), label, fill=(255, 255, 255), font=font)

    # Coordinate annotation
    parts = coord_stem.split("_")
    ra_str, dec_str = parts[0], parts[1]
    coord_text = f"RA={float(ra_str):.5f}  DEC={float(dec_str):.5f}"
    bbox = draw.textbbox((0, 0), coord_text, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text((total_w - text_w - 6, LABEL_H + IMG_SIZE - 18),
              coord_text, fill=(200, 200, 200), font=font)

    return composite


def process_pair(pair_name):
    """Process all galaxies for one overlap pair."""
    panels = OVERLAP_PANELS[pair_name]
    pair_composite_dir = os.path.join(COMPOSITE_DIR, pair_name)
    pngs = sorted(glob.glob(os.path.join(pair_composite_dir, "*.png")))

    created = 0
    skipped = 0
    missing = 0

    for png_path in pngs:
        basename = os.path.basename(png_path)
        # Skip already-generated SpArcFiRe composites
        if basename.endswith("-SpArcFiRe.png"):
            continue

        coord_stem = basename.removesuffix(".png")
        sf_name = coords_to_sparcfire_name(coord_stem)

        out_path = os.path.join(pair_composite_dir,
                                f"{coord_stem}-SpArcFiRe.png")

        images = []
        all_found = True
        for telescope in panels:
            path = find_logspiral_image(pair_name, telescope, sf_name)
            if path is None:
                print(f"  MISSING: {pair_name}/{telescope} for {coord_stem}")
                all_found = False
                break
            images.append(Image.open(path).convert("RGB"))

        if not all_found:
            missing += 1
            continue

        labels = [f"{t} (SpArcFiRe)" for t in panels]
        composite = make_composite(images, labels, coord_stem)
        composite.save(out_path)
        created += 1

    return created, skipped, missing


def main():
    total_created = 0
    total_missing = 0

    for pair_name in OVERLAP_PANELS:
        print(f"\n── {pair_name} ──")
        created, _, missing = process_pair(pair_name)
        total_created += created
        total_missing += missing
        print(f"  Created: {created}  |  Missing SpArcFiRe data: {missing}")

    print(f"\nDone. Created {total_created} SpArcFiRe composites "
          f"({total_missing} skipped due to missing data).")


if __name__ == "__main__":
    main()
