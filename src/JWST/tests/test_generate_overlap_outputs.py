import importlib.util
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_overlap_outputs.py"
SPEC = importlib.util.spec_from_file_location("generate_overlap_outputs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_fits_bytes(data):
    buf = BytesIO()
    pyfits.PrimaryHDU(np.asarray(data)).writeto(buf)
    return buf.getvalue()


def _make_wcs_fits_bytes(data, ra=1.0, dec=2.0, pixel_scale_deg=None):
    pixel_scale = pixel_scale_deg or (MODULE.FOV_DEG / np.asarray(data).shape[0])

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [((np.asarray(data).shape[1] + 1) / 2), ((np.asarray(data).shape[0] + 1) / 2)]
    wcs.wcs.cdelt = np.array([-pixel_scale, pixel_scale])
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    buf = BytesIO()
    pyfits.PrimaryHDU(np.asarray(data), header=wcs.to_header()).writeto(buf)
    return buf.getvalue()


def _write_parent_fits(path, data, ra=1.0, dec=2.0, pixel_scale_deg=None):
    pixel_scale = pixel_scale_deg or (MODULE.FOV_DEG / np.asarray(data).shape[0])

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [((np.asarray(data).shape[1] + 1) / 2), ((np.asarray(data).shape[0] + 1) / 2)]
    wcs.wcs.cdelt = np.array([-pixel_scale, pixel_scale])
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    primary = pyfits.PrimaryHDU()
    science = pyfits.ImageHDU(np.asarray(data), name="SCI", header=wcs.to_header())
    pyfits.HDUList([primary, science]).writeto(path)


def _make_blob_image(radius_px, intensity=220):
    y_grid, x_grid = np.indices((MODULE.IMG_SIZE, MODULE.IMG_SIZE), dtype=np.float64)
    cy = (MODULE.IMG_SIZE - 1) / 2.0
    cx = (MODULE.IMG_SIZE - 1) / 2.0
    mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= (radius_px ** 2)
    arr = np.zeros((MODULE.IMG_SIZE, MODULE.IMG_SIZE), dtype=np.uint8)
    arr[mask] = intensity
    return Image.fromarray(arr, "L").convert("RGB")


class GenerateOverlapOutputsTests(unittest.TestCase):
    def test_native_render_params_use_aggressive_hubble_defaults(self):
        params = MODULE._native_render_params({"scope": "Hubble"})

        self.assertEqual(params["background_percentile"], 8.0)
        self.assertEqual(params["upper_percentile"], 99.9)
        self.assertEqual(params["asinh_a"], MODULE.HST_NATIVE_ASINH_A)
        self.assertEqual(params["gamma"], 1.1)

    def test_native_render_params_use_relaxed_jwst_defaults(self):
        params = MODULE._native_render_params({"scope": "JWST"})

        self.assertEqual(params["background_percentile"], 5.0)
        self.assertEqual(params["upper_percentile"], 99.9)
        self.assertEqual(params["asinh_a"], MODULE.JWST_NATIVE_ASINH_A)
        self.assertEqual(params["gamma"], 0.8)

    def test_match_panel_sizes_to_sdss_zooms_smaller_native_panels(self):
        sdss = _make_blob_image(72)
        hubble = _make_blob_image(48)
        jwst = _make_blob_image(36)

        adjusted = MODULE._match_panel_sizes_to_sdss(
            [sdss, hubble, jwst],
            ["SDSS", "Hubble", "JWST"],
        )

        sdss_radius = MODULE._measure_panel_apparent_radius(sdss)
        hubble_before = MODULE._measure_panel_apparent_radius(hubble)
        jwst_before = MODULE._measure_panel_apparent_radius(jwst)
        hubble_after = MODULE._measure_panel_apparent_radius(adjusted[1])
        jwst_after = MODULE._measure_panel_apparent_radius(adjusted[2])

        self.assertIs(adjusted[0], sdss)
        self.assertGreater(hubble_after, hubble_before)
        self.assertGreater(jwst_after, jwst_before)
        self.assertLess(abs(hubble_after - sdss_radius), abs(hubble_before - sdss_radius))
        self.assertLess(abs(jwst_after - sdss_radius), abs(jwst_before - sdss_radius))

    def test_render_monochrome_fits_applies_gamma(self):
        input_data = np.linspace(0, 50, 128 * 128, dtype=np.float32).reshape(128, 128)

        low_gamma = MODULE._render_monochrome_fits(
            input_data,
            {
                "scope": "Hubble",
                "background_percentile": 0.0,
                "upper_percentile": 99.0,
                "asinh_a": 0.03,
                "gamma": 1.0,
            },
        )
        high_gamma = MODULE._render_monochrome_fits(
            input_data,
            {
                "scope": "Hubble",
                "background_percentile": 0.0,
                "upper_percentile": 99.0,
                "asinh_a": 0.03,
                "gamma": 1.6,
            },
        )

        self.assertIsNotNone(low_gamma)
        self.assertIsNotNone(high_gamma)
        self.assertLess(np.asarray(high_gamma).mean(), np.asarray(low_gamma).mean())

    def test_fetch_panel_reprojects_sdss_rgb_onto_reference_grid(self):
        input_data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)

        def fake_request(hips_id, _ra, _dec, fmt="jpg"):
            self.assertEqual(fmt, "fits")
            if hips_id == "CDS/P/SDSS9/i":
                return _make_wcs_fits_bytes(input_data + 10)
            if hips_id == "CDS/P/SDSS9/r":
                return _make_wcs_fits_bytes(input_data + 20)
            if hips_id == "CDS/P/SDSS9/g":
                return _make_wcs_fits_bytes(input_data + 30)
            raise AssertionError(f"unexpected band: {hips_id}")

        real_reproject = MODULE._reproject_to_reference_grid
        reproject_calls = []

        def tracking_reproject(*args, **kwargs):
            reproject_calls.append(True)
            return real_reproject(*args, **kwargs)

        with patch.object(MODULE, "_hips_request", side_effect=fake_request):
            with patch.object(MODULE, "_reproject_to_reference_grid", side_effect=tracking_reproject):
                img, quality_ok = MODULE.fetch_panel("CDS/P/SDSS9/color", 1.0, 2.0)

        self.assertTrue(quality_ok)
        self.assertEqual(img.size, (MODULE.IMG_SIZE, MODULE.IMG_SIZE))
        self.assertEqual(len(reproject_calls), len(MODULE.SDSS_RGB_BANDS))

    def test_fetch_panel_uses_manifest_numeric_extension(self):
        input_data = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_hubble.fits"
            manifest_path = Path(tmpdir) / "native_source_manifest.csv"
            _write_parent_fits(parent_path, input_data)
            manifest_path.write_text(
                "ra,dec,scope,provider,path,ext\n"
                f"1.0,2.0,Hubble,local_fits_cutout,{parent_path},1\n",
                encoding="utf-8",
            )

            MODULE._load_native_source_manifest.cache_clear()
            try:
                with patch.object(MODULE, "SOURCE_MANIFEST_PATH", str(manifest_path)):
                    img, quality_ok = MODULE.fetch_panel(MODULE.HST_HIPS_SOURCE, 1.0, 2.0)
            finally:
                MODULE._load_native_source_manifest.cache_clear()

        self.assertTrue(quality_ok)
        self.assertEqual(img.size, (MODULE.IMG_SIZE, MODULE.IMG_SIZE))

    def test_fetch_panel_uses_manifest_override_for_hubble(self):
        input_data = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_hubble.fits"
            manifest_path = Path(tmpdir) / "native_source_manifest.csv"
            _write_parent_fits(parent_path, input_data)
            manifest_path.write_text(
                "ra,dec,scope,provider,path\n"
                f"1.0,2.0,Hubble,local_fits_cutout,{parent_path}\n",
                encoding="utf-8",
            )

            MODULE._load_native_source_manifest.cache_clear()
            try:
                with patch.object(MODULE, "SOURCE_MANIFEST_PATH", str(manifest_path)):
                    with patch.object(MODULE, "_hips_request", side_effect=AssertionError("HiPS should not be called")):
                        img, quality_ok = MODULE.fetch_panel(MODULE.HST_HIPS_SOURCE, 1.0, 2.0)
            finally:
                MODULE._load_native_source_manifest.cache_clear()

        self.assertTrue(quality_ok)
        self.assertEqual(img.size, (MODULE.IMG_SIZE, MODULE.IMG_SIZE))

    def test_download_fits_uses_manifest_override_for_hubble(self):
        input_data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_hubble.fits"
            manifest_path = Path(tmpdir) / "native_source_manifest.csv"
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()
            _write_parent_fits(parent_path, input_data)
            manifest_path.write_text(
                "ra,dec,scope,provider,path\n"
                f"1.0,2.0,Hubble,local_fits_cutout,{parent_path}\n",
                encoding="utf-8",
            )

            MODULE._load_native_source_manifest.cache_clear()
            try:
                with patch.object(MODULE, "SOURCE_MANIFEST_PATH", str(manifest_path)):
                    with patch.object(MODULE, "_hips_request", side_effect=AssertionError("HiPS should not be called")):
                        created = MODULE._download_fits(
                            1.0,
                            2.0,
                            MODULE.HST_HIPS_SOURCE,
                            "",
                            str(output_dir),
                        )
            finally:
                MODULE._load_native_source_manifest.cache_clear()

            self.assertTrue(created)
            output_path = next(output_dir.glob("*.fits"))
            with pyfits.open(output_path) as hdu:
                self.assertEqual(hdu[0].data.tolist(), input_data.tolist())

    def test_fetch_panel_renders_local_hubble_cutout(self):
        input_data = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_hubble.fits"
            _write_parent_fits(parent_path, input_data)

            img, quality_ok = MODULE.fetch_panel(
                {
                    "provider": "local_fits_cutout",
                    "path": str(parent_path),
                    "scope": "Hubble",
                },
                1.0,
                2.0,
            )

        self.assertTrue(quality_ok)
        self.assertEqual(img.size, (MODULE.IMG_SIZE, MODULE.IMG_SIZE))
        self.assertGreater(np.asarray(img).sum(), 0)

    def test_build_reference_panel_cutout_passes_fobj_to_wcs(self):
        input_data = np.arange(128 * 128, dtype=np.float32).reshape(128, 128)

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_hubble.fits"
            _write_parent_fits(parent_path, input_data)

            real_wcs = MODULE.WCS
            wcs_calls = []

            def tracking_wcs(*args, **kwargs):
                wcs_calls.append(kwargs.get("fobj"))
                return real_wcs(*args, **kwargs)

            with patch.object(MODULE, "WCS", side_effect=tracking_wcs):
                cutout_data, cutout_header = MODULE._build_reference_panel_cutout(
                    {
                        "provider": "local_fits_cutout",
                        "path": str(parent_path),
                        "scope": "Hubble",
                    },
                    1.0,
                    2.0,
                )

        self.assertIsNotNone(cutout_data)
        self.assertIsNotNone(cutout_header)
        self.assertTrue(any(call is not None for call in wcs_calls))

    def test_fetch_panel_rejects_local_cutout_with_too_many_nans(self):
        input_data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
        input_data[0, 0] = np.nan

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_bad.fits"
            _write_parent_fits(parent_path, input_data)

            _, quality_ok = MODULE.fetch_panel(
                {
                    "provider": "local_fits_cutout",
                    "path": str(parent_path),
                    "scope": "Hubble",
                },
                1.0,
                2.0,
            )

        self.assertFalse(quality_ok)

    def test_fetch_panel_rejects_blank_jwst_fits(self):
        image_buf = BytesIO()
        Image.new("RGB", (2, 2), (10, 20, 30)).save(image_buf, format="JPEG")

        def fake_request(_hips_id, _ra, _dec, fmt="jpg"):
            if fmt == "jpg":
                return image_buf.getvalue()
            if fmt == "fits":
                return _make_fits_bytes(np.zeros((2, 2)))
            raise AssertionError(f"unexpected format: {fmt}")

        with patch.object(MODULE, "_hips_request", side_effect=fake_request):
            _, quality_ok = MODULE.fetch_panel("ESAVO/P/JWST/NIRCam_Imaging", 1.0, 2.0)

        self.assertFalse(quality_ok)

    def test_fetch_panel_flips_valid_jwst_image_horizontally(self):
        pixels = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 0]],
            ],
            dtype=np.uint8,
        )
        image_buf = BytesIO()
        Image.fromarray(pixels, "RGB").save(image_buf, format="PNG")

        def fake_request(_hips_id, _ra, _dec, fmt="jpg"):
            if fmt == "jpg":
                return image_buf.getvalue()
            if fmt == "fits":
                return _make_fits_bytes(np.array([[0, 1], [2, 3]], dtype=np.float32))
            raise AssertionError(f"unexpected format: {fmt}")

        with patch.object(MODULE, "_hips_request", side_effect=fake_request):
            img, quality_ok = MODULE.fetch_panel("ESAVO/P/JWST/NIRCam_Imaging", 1.0, 2.0)

        self.assertTrue(quality_ok)
        self.assertEqual(np.asarray(img).tolist(), np.fliplr(pixels).tolist())

    def test_fetch_panel_rotates_valid_jwst_image_180_when_requested(self):
        pixels = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 0]],
            ],
            dtype=np.uint8,
        )
        image_buf = BytesIO()
        Image.fromarray(pixels, "RGB").save(image_buf, format="PNG")

        def fake_request(_hips_id, _ra, _dec, fmt="jpg"):
            if fmt == "jpg":
                return image_buf.getvalue()
            if fmt == "fits":
                return _make_fits_bytes(np.array([[0, 1], [2, 3]], dtype=np.float32))
            raise AssertionError(f"unexpected format: {fmt}")

        with patch.object(MODULE, "_hips_request", side_effect=fake_request):
            img, quality_ok = MODULE.fetch_panel(
                "ESAVO/P/JWST/NIRCam_Imaging",
                1.0,
                2.0,
                jwst_transform="rotate_180",
            )

        self.assertTrue(quality_ok)
        self.assertEqual(np.asarray(img).tolist(), np.rot90(pixels, 2).tolist())

    def test_fetch_panel_applies_flip_then_rotate_when_requested(self):
        pixels = np.array(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 0]],
            ],
            dtype=np.uint8,
        )
        image_buf = BytesIO()
        Image.fromarray(pixels, "RGB").save(image_buf, format="PNG")

        def fake_request(_hips_id, _ra, _dec, fmt="jpg"):
            if fmt == "jpg":
                return image_buf.getvalue()
            if fmt == "fits":
                return _make_fits_bytes(np.array([[0, 1], [2, 3]], dtype=np.float32))
            raise AssertionError(f"unexpected format: {fmt}")

        with patch.object(MODULE, "_hips_request", side_effect=fake_request):
            img, quality_ok = MODULE.fetch_panel(
                "ESAVO/P/JWST/NIRCam_Imaging",
                1.0,
                2.0,
                jwst_transform=["flip_horizontal", "rotate_180"],
            )

        self.assertTrue(quality_ok)
        self.assertEqual(np.asarray(img).tolist(), np.flipud(pixels).tolist())

    def test_download_fits_skips_blank_jwst(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                MODULE,
                "_hips_request",
                return_value=_make_fits_bytes(np.zeros((2, 2), dtype=np.float32)),
            ):
                created = MODULE._download_fits(
                    1.0,
                    2.0,
                    "ESAVO/P/JWST/NIRCam_Imaging",
                    "",
                    tmpdir,
                )

            self.assertFalse(created)
            self.assertEqual(list(Path(tmpdir).glob("*.fits")), [])

    def test_download_fits_writes_local_hubble_cutout(self):
        input_data = np.arange(1, 17, dtype=np.float32).reshape(4, 4)

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_hubble.fits"
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()
            _write_parent_fits(parent_path, input_data)

            created = MODULE._download_fits(
                1.0,
                2.0,
                {
                    "provider": "local_fits_cutout",
                    "path": str(parent_path),
                    "scope": "Hubble",
                },
                "",
                str(output_dir),
            )

            self.assertTrue(created)
            output_path = next(output_dir.glob("*.fits"))
            with pyfits.open(output_path) as hdu:
                self.assertEqual(hdu[0].data.tolist(), input_data.tolist())

    def test_download_fits_applies_jwst_transform_for_local_source(self):
        input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_jwst.fits"
            output_dir = Path(tmpdir) / "out"
            output_dir.mkdir()
            _write_parent_fits(parent_path, input_data)

            created = MODULE._download_fits(
                1.0,
                2.0,
                {
                    "provider": "local_fits_cutout",
                    "path": str(parent_path),
                    "scope": "JWST",
                },
                "",
                str(output_dir),
                jwst_transform="rotate_180",
            )

            self.assertTrue(created)
            output_path = next(output_dir.glob("*.fits"))
            with pyfits.open(output_path) as hdu:
                self.assertEqual(hdu[0].data.tolist(), np.rot90(input_data, 2).tolist())

    def test_download_fits_flips_jwst_data_horizontally(self):
        input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(MODULE, "_hips_request", return_value=_make_fits_bytes(input_data)):
                created = MODULE._download_fits(
                    1.0,
                    2.0,
                    "ESAVO/P/JWST/NIRCam_Imaging",
                    "",
                    tmpdir,
                )

            self.assertTrue(created)
            output_path = next(Path(tmpdir).glob("*.fits"))
            with pyfits.open(output_path) as hdu:
                self.assertEqual(hdu[0].data.tolist(), np.fliplr(input_data).tolist())

    def test_download_fits_rotates_jwst_data_180_when_requested(self):
        input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(MODULE, "_hips_request", return_value=_make_fits_bytes(input_data)):
                created = MODULE._download_fits(
                    1.0,
                    2.0,
                    "ESAVO/P/JWST/NIRCam_Imaging",
                    "",
                    tmpdir,
                    jwst_transform="rotate_180",
                )

            self.assertTrue(created)
            output_path = next(Path(tmpdir).glob("*.fits"))
            with pyfits.open(output_path) as hdu:
                self.assertEqual(hdu[0].data.tolist(), np.rot90(input_data, 2).tolist())

    def test_download_fits_applies_flip_then_rotate_when_requested(self):
        input_data = np.array([[1, 2], [3, 4]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(MODULE, "_hips_request", return_value=_make_fits_bytes(input_data)):
                created = MODULE._download_fits(
                    1.0,
                    2.0,
                    "ESAVO/P/JWST/NIRCam_Imaging",
                    "",
                    tmpdir,
                    jwst_transform=["flip_horizontal", "rotate_180"],
                )

            self.assertTrue(created)
            output_path = next(Path(tmpdir).glob("*.fits"))
            with pyfits.open(output_path) as hdu:
                self.assertEqual(hdu[0].data.tolist(), np.flipud(input_data).tolist())


if __name__ == "__main__":
    unittest.main()