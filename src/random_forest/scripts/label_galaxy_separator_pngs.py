#!/usr/bin/env python3
"""Interactively label RUN-1 separator PNGs into galaxy/non-galaxy bins.

The script walks PNGs under a source run, shows them one at a time when an
interactive matplotlib backend is available, and copies each image into either
randomforest_training_data/galaxy_separator_RF/1-galaxies or
randomforest_training_data/galaxy_separator_RF/1-not-galaxies based on the
user's response.

Every decision is appended to a CSV log so later runs can resume without
re-presenting the same source PNGs.

Example command:
    python scripts/label_galaxy_separator_pngs.py --subset accepted
    python scripts/label_galaxy_separator_pngs.py --subset rejected
    python scripts/label_galaxy_separator_pngs.py --subset accepted --limit 200
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_DIR = ROOT_DIR / 'inference' / 'galaxy-classifier-RF' / 'RUN-1' / 'galaxy'
DEFAULT_DEST_ROOT = ROOT_DIR / 'randomforest_training_data' / 'galaxy_separator_RF'
DEFAULT_LOG_CSV = ROOT_DIR / 'output' / 'galaxy_separator_label_log.csv'
GALAXY_DIRNAME = '1-galaxies'
NON_GALAXY_DIRNAME = '1-not-galaxies'
LOG_FIELDS = [
    'timestamp_utc',
    'source_png',
    'source_rel_path',
    'subset',
    'decision',
    'destination_png',
]
INTERACTIVE_BACKENDS = ('TkAgg', 'QtAgg')
NONINTERACTIVE_BACKENDS = ('agg', 'pdf', 'ps', 'svg')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Interactively label galaxy-separator PNGs into training bins.'
    )
    parser.add_argument('--source-dir', type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument('--dest-root', type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument('--log-csv', type=Path, default=DEFAULT_LOG_CSV)
    parser.add_argument(
        '--subset',
        choices=('accepted', 'rejected', 'all'),
        default='accepted',
        help=(
            "accepted: only */galaxy/*.png, rejected: only */non-galaxy/*.png, "
            "all: every PNG under source-dir."
        ),
    )
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def collect_pngs(source_dir: Path, subset: str) -> list[Path]:
    png_paths = sorted(source_dir.rglob('*.png'))
    if subset == 'accepted':
        return [path for path in png_paths if path.parent.name == 'galaxy']
    if subset == 'rejected':
        return [path for path in png_paths if path.parent.name == 'non-galaxy']
    return png_paths


def load_logged_sources(log_csv: Path) -> set[str]:
    if not log_csv.exists():
        return set()

    with log_csv.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return {
            row['source_png']
            for row in reader
            if row.get('source_png')
        }


def append_log_row(log_csv: Path, row: dict[str, str]) -> None:
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_csv.exists()

    with log_csv.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def create_viewer():
    try:
        import matplotlib

        if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'):
            for backend in INTERACTIVE_BACKENDS:
                try:
                    matplotlib.use(backend, force=True)
                    break
                except Exception:
                    continue

        import matplotlib.pyplot as plt

        backend = matplotlib.get_backend().lower()
        if any(name in backend for name in NONINTERACTIVE_BACKENDS):
            return None

        plt.ion()
        figure, axis = plt.subplots(figsize=(7, 7))
        manager = getattr(figure.canvas, 'manager', None)
        if manager is not None:
            try:
                manager.set_window_title('Galaxy Separator Labeler')
            except Exception:
                pass
        return plt, figure, axis
    except Exception:
        return None


def create_external_opener():
    candidates = [
        ('code', ['code', '--reuse-window']),
        ('code-insiders', ['code-insiders', '--reuse-window']),
        ('xdg-open', ['xdg-open']),
    ]

    for executable, command_prefix in candidates:
        resolved = shutil.which(executable)
        if resolved is not None:
            return [resolved, *command_prefix[1:]]
    return None


def open_png_externally(opener_command: list[str] | None, png_path: Path) -> bool:
    if opener_command is None:
        return False

    try:
        subprocess.run(
            [*opener_command, str(png_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except OSError:
        return False


def show_png(viewer, png_path: Path, title: str) -> bool:
    if viewer is None:
        return False

    plt, figure, axis = viewer
    image = Image.open(png_path)

    axis.clear()
    axis.imshow(image)
    axis.set_axis_off()
    axis.set_title(title)
    figure.tight_layout()
    figure.canvas.draw_idle()
    plt.pause(0.001)
    return True


def prompt_for_decision(index: int, total: int, png_path: Path, shown: bool) -> str:
    if not shown:
        print()
        print(f'[{index}/{total}] Open this PNG, then answer the prompt:')
        print(f'  {png_path}')

    while True:
        response = input(
            f'[{index}/{total}] g=galaxy n=not-galaxy s=skip q=quit > '
        ).strip().lower()

        if response in {'g', 'galaxy', '1'}:
            return 'galaxy'
        if response in {'n', 'not', 'not-galaxy', 'non-galaxy', '0'}:
            return 'not-galaxy'
        if response in {'s', 'skip'}:
            return 'skip'
        if response in {'q', 'quit'}:
            return 'quit'

        print('Enter g, n, s, or q.')


def choose_destination_path(label_dir: Path, source_path: Path, source_root: Path) -> Path:
    candidate = label_dir / source_path.name
    if not candidate.exists():
        return candidate

    try:
        if filecmp.cmp(source_path, candidate, shallow=False):
            return candidate
    except OSError:
        pass

    rel_parts = source_path.relative_to(source_root).with_suffix('').parts
    alt_stem = '__'.join(rel_parts)
    candidate = label_dir / f'{alt_stem}{source_path.suffix}'
    counter = 2

    while candidate.exists():
        candidate = label_dir / f'{alt_stem}__{counter}{source_path.suffix}'
        counter += 1

    return candidate


def make_log_row(
    source_path: Path,
    source_root: Path,
    subset: str,
    decision: str,
    destination_path: Path | None,
) -> dict[str, str]:
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'source_png': str(source_path),
        'source_rel_path': source_path.relative_to(source_root).as_posix(),
        'subset': subset,
        'decision': decision,
        'destination_png': '' if destination_path is None else str(destination_path),
    }


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    dest_root = args.dest_root.resolve()
    log_csv = args.log_csv.resolve()

    if not source_dir.is_dir():
        print(f'Error: source directory not found: {source_dir}', file=sys.stderr)
        sys.exit(1)

    galaxy_dir = dest_root / GALAXY_DIRNAME
    non_galaxy_dir = dest_root / NON_GALAXY_DIRNAME
    galaxy_dir.mkdir(parents=True, exist_ok=True)
    non_galaxy_dir.mkdir(parents=True, exist_ok=True)

    source_pngs = collect_pngs(source_dir, args.subset)
    logged_sources = load_logged_sources(log_csv)
    pending_pngs = [path for path in source_pngs if str(path) not in logged_sources]

    random.Random(args.seed).shuffle(pending_pngs)
    if args.limit is not None:
        pending_pngs = pending_pngs[:args.limit]

    if not pending_pngs:
        print('No unlabeled PNGs matched the requested source set.')
        return

    viewer = create_viewer()
    opener_command = None
    if viewer is None:
        opener_command = create_external_opener()
        if opener_command is not None:
            print(
                'Interactive viewer unavailable; each PNG will be opened with: '
                + ' '.join(opener_command)
            )
        else:
            print('Interactive viewer unavailable; the script will print each PNG path before prompting.')

    counts = {'galaxy': 0, 'not-galaxy': 0, 'skip': 0}
    total = len(pending_pngs)

    print(f'Source PNGs queued: {total}')
    print(f'Label log: {log_csv}')
    print(f'Galaxy destination: {galaxy_dir}')
    print(f'Non-galaxy destination: {non_galaxy_dir}')

    try:
        for index, png_path in enumerate(pending_pngs, start=1):
            title = png_path.relative_to(source_dir).as_posix()
            shown = show_png(viewer, png_path, title)
            if not shown:
                shown = open_png_externally(opener_command, png_path)
            decision = prompt_for_decision(index, total, png_path, shown)

            if decision == 'quit':
                break

            if decision == 'skip':
                append_log_row(
                    log_csv,
                    make_log_row(png_path, source_dir, args.subset, decision, None),
                )
                counts['skip'] += 1
                continue

            label_dir = galaxy_dir if decision == 'galaxy' else non_galaxy_dir
            destination_path = choose_destination_path(label_dir, png_path, source_dir)
            if not destination_path.exists():
                shutil.copy2(png_path, destination_path)

            append_log_row(
                log_csv,
                make_log_row(png_path, source_dir, args.subset, decision, destination_path),
            )
            counts[decision] += 1
            print(f'  {decision:11s} -> {destination_path.name}')
    finally:
        if viewer is not None:
            plt, figure, _ = viewer
            plt.close(figure)

    print()
    print('Session summary:')
    print(f"  galaxy:     {counts['galaxy']}")
    print(f"  not-galaxy: {counts['not-galaxy']}")
    print(f"  skipped:    {counts['skip']}")


if __name__ == '__main__':
    main()