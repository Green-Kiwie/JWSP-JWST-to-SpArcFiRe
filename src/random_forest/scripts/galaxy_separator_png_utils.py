#!/usr/bin/env python3
"""Utilities for extracting morphology-only content from labeled PNGs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def largest_nonwhite_component_bbox(
    image: Image.Image,
    threshold: int = 245,
    padding: int = 1,
) -> tuple[int, int, int, int]:
    """Return the bounding box of the largest non-white connected component.

    Annotated training PNGs have a white canvas with the thumbnail panel plus
    text labels. The panel is typically the largest non-white component, so
    cropping to that component removes the title text before feature extraction.

    If the image has no white border and is already a raw crop, the bbox will
    naturally expand to cover nearly the entire image.
    """
    gray = image.convert('L')
    pixels = np.asarray(gray)
    mask = pixels < threshold

    if not np.any(mask):
        return 0, 0, image.width, image.height

    labeled, component_count = ndimage.label(mask)
    if component_count == 0:
        return 0, 0, image.width, image.height

    best_slice = None
    best_area = -1
    for component_id, component_slice in enumerate(ndimage.find_objects(labeled), start=1):
        if component_slice is None:
            continue
        component_area = int(np.count_nonzero(labeled[component_slice] == component_id))
        if component_area > best_area:
            best_area = component_area
            best_slice = component_slice

    if best_slice is None:
        return 0, 0, image.width, image.height

    y_slice, x_slice = best_slice
    left = max(0, x_slice.start - padding)
    top = max(0, y_slice.start - padding)
    right = min(image.width, x_slice.stop + padding)
    bottom = min(image.height, y_slice.stop + padding)
    return left, top, right, bottom


def crop_png_to_morphology_panel(
    image: Image.Image,
    threshold: int = 245,
    padding: int = 1,
) -> Image.Image:
    """Crop a PNG down to the thumbnail panel used for morphology."""
    crop_box = largest_nonwhite_component_bbox(image, threshold=threshold, padding=padding)
    return image.crop(crop_box)


def load_png_morphology_array(
    png_path: str | Path,
    *,
    auto_crop: bool = True,
    threshold: int = 245,
    padding: int = 1,
) -> np.ndarray:
    """Load a PNG as a morphology-only grayscale float64 array."""
    image = Image.open(png_path)
    if auto_crop:
        image = crop_png_to_morphology_panel(image, threshold=threshold, padding=padding)
    if image.mode != 'L':
        image = image.convert('L')
    return np.array(image, dtype=np.float64)