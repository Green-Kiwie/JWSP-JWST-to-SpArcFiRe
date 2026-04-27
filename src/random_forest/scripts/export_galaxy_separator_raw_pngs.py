#!/usr/bin/env python3
"""Export annotation-free PNGs for the galaxy separator training set.

This script avoids scanning the massive JWST thumbnail directory. Instead, it
uses the already-exported annotated PNGs in
randomforest_training_data/galaxy_separator_RF, crops out the image panel for
each file, and writes a mapping CSV from the score-named PNG back to the source
object name.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from PIL import Image

from galaxy_separator_png_utils import largest_nonwhite_component_bbox


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR = ROOT_DIR / 'randomforest_training_data' / 'galaxy_separator_RF'
DEFAULT_OUTPUT_DIR = ROOT_DIR / 'randomforest_training_data' / 'galaxy_separator_RF_raw'
DEFAULT_MAPPING_CSV = ROOT_DIR / 'output' / 'galaxy_separator_rf_raw_mapping.csv'

PREDICTIONS_CSV_CANDIDATES = (
    ROOT_DIR / 'output' / 'jwst_predictions_v2.csv',
    ROOT_DIR / 'output' / 'jwst_predictions.csv',
)


def default_predictions_csv() -> Path:
    """Return the first available JWST predictions CSV path."""
    for candidate in PREDICTIONS_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return PREDICTIONS_CSV_CANDIDATES[-1]


DEFAULT_PREDICTIONS_CSV = default_predictions_csv()

# These 18 score-named PNGs are not present in the saved JWST predictions CSV,
# so their object names are taken from the title text embedded in the images.
MANUAL_SCORE_NAME_OVERRIDES = {
    '0.39894666720181704': '187_475185_27_125177_f430m_v1',
    '0.39898666767248264': '81_085026_-70_082100_f115w_v1',
    '0.39899999887785975': '80_467920_-69_484384_f115w_v4',
    '0.39908666895495526': '80_408790_-69_481494_f356w_v3',
    '0.39912000271802145': '80_481527_-69_498994_f277w_v8',
    '0.39916666795810063': '334_420119_-16_495169_f277w_v1',
    '0.39916666819817487': '82_798141_-68_782591_f200w_v1',
    '0.39931999941666924': '80_458419_-69_515970_f200w_v6',
    '0.39933333296949663': '245_949868_-26_442991_f150w_v1',
    '0.39935333252574007': '345_880450_-62_907378_f200w_v1',
    '0.39955999826391536': '81_106901_-70_092988_f200w_v1',
    '0.39969999905261727': '80_495673_-69_481240_f200w_v1',
    '0.39971333094562095': '82_783841_-68_796817_f430m_v1',
    '0.39973333386393883': '80_502629_-69_492685_f140m_v3',
    '0.39977333415920535': '80_482315_-69_495719_f150w_v1',
    '0.39978666592389345': '334_431680_-16_483229_f277w_v1',
    '0.39985999691610535': '187_436716_27_137415_f430m_v2',
    '0.39994666777758137': '80_528927_-69_480440_f200w_v2',
}


def normalize_score_text(value: object) -> str:
    """Return a stable decimal string for score comparisons."""
    text = str(value).strip()
    try:
        decimal_value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f'invalid score value: {value!r}') from exc

    normalized = format(decimal_value, 'f')
    if '.' in normalized:
        normalized = normalized.rstrip('0').rstrip('.')
    return normalized or '0'


def load_score_name_map(predictions_csv: Path) -> dict[str, list[str]]:
    """Build a score-to-object-name map from the saved JWST predictions CSV."""
    df = pd.read_csv(predictions_csv, dtype={'name': str, 'predicted_P_spiral': str})
    mapping: dict[str, list[str]] = {}

    for row in df.itertuples(index=False):
        score_text = normalize_score_text(row.predicted_P_spiral)
        names = mapping.setdefault(score_text, [])
        if row.name not in names:
            names.append(row.name)

    for score_text, source_name in MANUAL_SCORE_NAME_OVERRIDES.items():
        mapping[score_text] = [source_name]
    return mapping

def export_raw_pngs(
    input_dir: Path,
    output_dir: Path,
    predictions_csv: Path,
    mapping_csv: Path,
    overwrite: bool,
) -> pd.DataFrame:
    """Export annotation-free PNG crops and write the source mapping CSV."""
    score_name_map = load_score_name_map(predictions_csv)
    rows: list[dict[str, object]] = []
    png_paths = sorted(input_dir.rglob('*.png'))

    if not png_paths:
        raise FileNotFoundError(f'no PNGs found under {input_dir}')

    for png_path in png_paths:
        score_text = normalize_score_text(png_path.stem)
        source_names = score_name_map.get(score_text)
        if source_names is None:
            raise KeyError(
                f'no source name found for {png_path.name}; score={score_text}. '
                'Add it to MANUAL_SCORE_NAME_OVERRIDES if needed.'
            )
        if len(source_names) != 1:
            raise ValueError(
                f'score {score_text} for {png_path.name} resolves to multiple source names: '
                f'{source_names!r}'
            )
        source_name = source_names[0]

        image = Image.open(png_path)
        crop_box = largest_nonwhite_component_bbox(image)
        raw_image = image.crop(crop_box)

        rel_path = png_path.relative_to(input_dir)
        raw_path = output_dir / rel_path
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not raw_path.exists():
            raw_image.save(raw_path)

        rows.append({
            'label_dir': rel_path.parent.as_posix(),
            'annotated_png': str(png_path),
            'raw_png': str(raw_path),
            'score_filename': png_path.name,
            'score': score_text,
            'source_name': source_name,
            'crop_left': crop_box[0],
            'crop_top': crop_box[1],
            'crop_right': crop_box[2],
            'crop_bottom': crop_box[3],
            'raw_width': raw_image.width,
            'raw_height': raw_image.height,
        })

    mapping_df = pd.DataFrame(rows)
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(mapping_csv, index=False)
    return mapping_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Export annotation-free PNGs for galaxy_separator_RF.'
    )
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--predictions-csv', type=Path, default=DEFAULT_PREDICTIONS_CSV)
    parser.add_argument('--mapping-csv', type=Path, default=DEFAULT_MAPPING_CSV)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping_df = export_raw_pngs(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        predictions_csv=args.predictions_csv.resolve(),
        mapping_csv=args.mapping_csv.resolve(),
        overwrite=args.overwrite,
    )

    print(f'Exported {len(mapping_df)} raw PNGs to {args.output_dir}')
    print(f'Mapping CSV written to {args.mapping_csv}')


if __name__ == '__main__':
    main()