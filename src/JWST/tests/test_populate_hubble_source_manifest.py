import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "populate_hubble_source_manifest.py"
SPEC = importlib.util.spec_from_file_location("populate_hubble_source_manifest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _make_wcs(ra=1.0, dec=2.0, shape=(20, 20)):
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [((shape[1] + 1) / 2), ((shape[0] + 1) / 2)]
    wcs.wcs.cdelt = np.array([-0.0001, 0.0001])
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def _write_hubble_like_parent(path, ra=1.0, dec=2.0):
    wcs = _make_wcs(ra=ra, dec=dec)
    header = wcs.to_header()
    sci = fits.ImageHDU(np.ones((20, 20), dtype=np.float32), name="SCI", header=header)
    err = fits.ImageHDU(np.full((20, 20), 5.0, dtype=np.float32), name="ERR", header=header)
    hdul = fits.HDUList([fits.PrimaryHDU(), sci, err, sci.copy(), err.copy()])
    hdul.writeto(path)


class PopulateHubbleSourceManifestTests(unittest.TestCase):
    def test_covering_extension_prefers_science_hdu(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_hubble.fits"
            _write_hubble_like_parent(parent_path)

            covers_coord, ext_index, ext_name = MODULE.covering_extension(
                parent_path,
                SkyCoord(1.0 * u.deg, 2.0 * u.deg),
                allowed_names={"SCI"},
            )

        self.assertTrue(covers_coord)
        self.assertEqual(ext_index, 1)
        self.assertEqual(ext_name, "SCI")

    def test_list_hubble_products_prefers_drizzled_science_products(self):
        obs_row = {"obs_id": "test_obs"}
        product_rows = [
            {
                "productFilename": "j8pu6sobq_flc.fits",
                "productType": "SCIENCE",
                "calib_level": 2,
                "dataURI": "mast:test/flc",
            },
            {
                "productFilename": "j8pu6s010_drz.fits",
                "productType": "SCIENCE",
                "calib_level": 3,
                "dataURI": "mast:test/drz",
            },
            {
                "productFilename": "hst_9822_6s_acs_wfc_f814w_j8pu6s_drc.fits",
                "productType": "SCIENCE",
                "calib_level": 3,
                "dataURI": "mast:test/drc",
            },
        ]

        with patch.object(MODULE.Observations, "get_product_list", return_value=product_rows):
            ranked = MODULE.list_hubble_products(obs_row, {})

        self.assertEqual(
            [row["product_filename"] for row in ranked],
            [
                "hst_9822_6s_acs_wfc_f814w_j8pu6s_drc.fits",
                "j8pu6s010_drz.fits",
                "j8pu6sobq_flc.fits",
            ],
        )

    def test_existing_row_without_science_extension_name_is_not_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parent_path = Path(tmpdir) / "parent_hubble.fits"
            _write_hubble_like_parent(parent_path)

            self.assertFalse(
                MODULE.is_existing_hubble_row_usable(
                    {
                        "scope": "Hubble",
                        "path": str(parent_path),
                        "ext": "2",
                    }
                )
            )


if __name__ == "__main__":
    unittest.main()