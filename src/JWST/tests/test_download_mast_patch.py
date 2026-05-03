import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_MAST_patch.py"
SPEC = importlib.util.spec_from_file_location("download_MAST_patch", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadMastPatchTests(unittest.TestCase):
    def test_load_manifest_rows_reads_headered_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "jwst_L3_sky_patches.csv"
            manifest_path.write_text(
                "dataURI,productFilename,calib_level\n"
                "mast:JWST/product/a_i2d.fits,a_i2d.fits,3\n"
                "mast:JWST/product/b_i2d.fits,b_i2d.fits,2\n",
                encoding="utf-8",
            )

            rows = MODULE.load_manifest_rows(manifest_path)

        self.assertEqual(rows, [("mast:JWST/product/a_i2d.fits", "a_i2d.fits")])

    def test_load_manifest_rows_falls_back_to_legacy_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "legacy.csv"
            manifest_path.write_text(
                ",".join(
                    [
                        "71737988",
                        "JWST",
                        "image",
                        "obsid",
                        "desc",
                        "S",
                        "mast:JWST/product/legacy_i2d.fits",
                        "SCIENCE",
                        "",
                        "I2D",
                        "",
                        "CALJWST",
                        "1.9.6",
                        "1059",
                        "legacy_i2d.fits",
                        "119358720",
                        "71738039",
                        "PUBLIC",
                        "3",
                        "F212N",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = MODULE.load_manifest_rows(manifest_path)

        self.assertEqual(rows, [("mast:JWST/product/legacy_i2d.fits", "legacy_i2d.fits")])

    def test_download_sky_patches_uses_manifest_rows(self):
        downloads = []

        def fake_downloader(url, destination):
            downloads.append((url, destination))
            Path(destination).write_text("fits", encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "jwst_L3_sky_patches.csv"
            download_dir = Path(tmpdir) / "downloads"
            manifest_path.write_text(
                "dataURI,productFilename,calib_level\n"
                "mast:JWST/product/a_i2d.fits,a_i2d.fits,3\n",
                encoding="utf-8",
            )

            result = MODULE.download_sky_patches(
                manifest_path,
                download_directory=download_dir,
                downloader=fake_downloader,
                logger=lambda _message: None,
            )

        self.assertEqual(result, {"downloaded": 1, "skipped": 0, "failed": 0, "total": 1})
        self.assertEqual(
            downloads,
            [("https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:JWST/product/a_i2d.fits", str(download_dir / "a_i2d.fits"))],
        )


if __name__ == "__main__":
    unittest.main()