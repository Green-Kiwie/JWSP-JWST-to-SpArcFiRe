# Archived Scripts

These scripts are **deprecated** legacy code, preserved for reference only.
They have been superseded by `parallel_galaxy_extractor.py`.

| Script | Purpose | Why Archived |
|--------|---------|-------------|
| `sep_one_file.py` | Interactive single-file extraction with `input()` prompts | Not suitable for batch processing |
| `sep_from_mast.py` | Downloads from MAST + extraction | `_run_sep()` was commented out; download logic moved elsewhere |
| `sep_from_local.py` | Local file extraction | Contains bugs (e.g., `import trackback`); replaced by parallel extractor |
| `sep_discovery.py` | Discovery-mode scan | Functionality merged into parallel extractor's `scan-only` mode |

**Do not use these scripts for production runs.** Use `parallel_galaxy_extractor.py` instead.
