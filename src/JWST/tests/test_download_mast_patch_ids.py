import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_MAST_patch_IDs.py"
SPEC = importlib.util.spec_from_file_location("download_MAST_patch_IDs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeMastMissions:
    def __init__(self, observation_batches, product_batches):
        self._observation_batches = list(observation_batches)
        self._product_batches = list(product_batches)
        self.query_calls = []
        self.product_calls = []

    def query_criteria(self, **kwargs):
        self.query_calls.append(kwargs)
        if self._observation_batches:
            return self._observation_batches.pop(0)
        return pd.DataFrame()

    def get_product_list(self, datasets):
        self.product_calls.append(datasets)
        if self._product_batches:
            return self._product_batches.pop(0)
        return pd.DataFrame()


class DownloadMastPatchIdsTests(unittest.TestCase):
    def test_build_query_kwargs_matches_portal_filters(self):
        query_kwargs = MODULE.build_query_kwargs()

        self.assertEqual(query_kwargs["select_cols"], [])
        self.assertEqual(query_kwargs["productLevel"], "3")
        self.assertEqual(query_kwargs["instrume"], MODULE.JWST_INSTRUMENTS)
        self.assertEqual(query_kwargs["exp_type"], MODULE.JWST_EXP_TYPES)

    def test_normalize_query_results_keeps_only_level3_i2d_images(self):
        frame = pd.DataFrame(
            [
                {
                    "filename": "jw_good_a_i2d.fits",
                    "productLevel": 3.0,
                    "dataproduct_type": "image",
                    "obs_id": "obs-a",
                    "target_name": "target-a",
                    "instrume": "NIRCAM",
                    "exp_type": "NRC_IMAGE",
                    "filters": "F200W",
                    "proposal_id": "1234",
                    "size": 42,
                    "dataRights": "PUBLIC",
                },
                {
                    "filename": "jw_bad_level_i2d.fits",
                    "productLevel": "2",
                    "dataproduct_type": "image",
                },
                {
                    "filename": "jw_bad_type_i2d.fits",
                    "productLevel": "3",
                    "dataproduct_type": "spectrum",
                },
                {
                    "filename": "jw_bad_suffix_x1d.fits",
                    "productLevel": "3",
                    "dataproduct_type": "image",
                },
                {
                    "productFilename": "jw_good_b_i2d.fits",
                    "calib_level": "3",
                    "dataproduct_type": "image",
                    "dataURI": "mast:JWST/product/jw_good_b_i2d.fits",
                    "obs_collection": "JWST",
                },
                {
                    "productFilename": "jw_good_b_i2d.fits",
                    "calib_level": "3",
                    "dataproduct_type": "image",
                    "dataURI": "mast:JWST/product/jw_good_b_i2d.fits",
                    "obs_collection": "JWST",
                },
            ]
        )

        normalized = MODULE.normalize_query_results(frame)

        self.assertEqual(list(normalized["productFilename"]), ["jw_good_a_i2d.fits", "jw_good_b_i2d.fits"])
        self.assertEqual(
            list(normalized["dataURI"]),
            [
                "mast:JWST/product/jw_good_a_i2d.fits",
                "mast:JWST/product/jw_good_b_i2d.fits",
            ],
        )
        self.assertEqual(list(normalized["calib_level"]), ["3", "3"])
        self.assertEqual(list(normalized["obs_collection"]), ["JWST", "JWST"])

    def test_normalize_query_results_keeps_product_list_i2d_rows_without_explicit_level(self):
        frame = pd.DataFrame(
            [
                {
                    "filename": "jw_live_i2d.fits",
                    "uri": "mast:JWST/product/jw_live_i2d.fits",
                    "dataset": "obs-live",
                    "instrument_name": "NIRCAM",
                    "access": "PUBLIC",
                    "type": "science",
                    "category": "science",
                    "obs_collection": "JWST",
                    "size": 100,
                },
                {
                    "filename": "jw_live_cat.ecsv",
                    "uri": "mast:JWST/product/jw_live_cat.ecsv",
                    "dataset": "obs-live",
                    "instrument_name": "NIRCAM",
                    "access": "PUBLIC",
                },
                {
                    "filename": "jw_live_i2d.jpg",
                    "uri": "mast:JWST/product/jw_live_i2d.jpg",
                    "dataset": "obs-live",
                    "instrument_name": "NIRCAM",
                    "access": "PUBLIC",
                },
            ]
        )

        normalized = MODULE.normalize_query_results(frame)

        self.assertEqual(list(normalized["productFilename"]), ["jw_live_i2d.fits"])
        self.assertEqual(list(normalized["dataURI"]), ["mast:JWST/product/jw_live_i2d.fits"])
        self.assertEqual(list(normalized["calib_level"]), ["3"])
        self.assertEqual(list(normalized["instrume"]), ["NIRCAM"])
        self.assertEqual(list(normalized["dataRights"]), ["PUBLIC"])

    def test_export_sky_patch_manifest_pages_until_empty_batch(self):
        first_observation_batch = pd.DataFrame(
            [
                {
                    "ArchiveFileID": "obs-1",
                    "fileSetName": "obs-1",
                    "productLevel": "3",
                    "instrume": "NIRCAM",
                    "exp_type": "NRC_IMAGE",
                    "program": "1001",
                    "targprop": "target-1",
                    "access": "PUBLIC",
                },
                {
                    "ArchiveFileID": "obs-2",
                    "fileSetName": "obs-2",
                    "productLevel": "3",
                    "instrume": "NIRCAM",
                    "exp_type": "NRC_IMAGE",
                    "program": "1002",
                    "targprop": "target-2",
                    "access": "PUBLIC",
                },
            ]
        )
        second_observation_batch = pd.DataFrame(
            [
                {
                    "ArchiveFileID": "obs-3",
                    "fileSetName": "obs-3",
                    "productLevel": "3",
                    "instrume": "MIRI",
                    "exp_type": "MIR_IMAGE",
                    "program": "1003",
                    "targprop": "target-3",
                    "access": "PUBLIC",
                }
            ]
        )
        first_product_batch = pd.DataFrame(
            [
                {
                    "filename": "jw_first_i2d.fits",
                    "dataset": "obs-1",
                    "uri": "mast:JWST/product/jw_first_i2d.fits",
                    "instrument_name": "NIRCAM",
                    "access": "PUBLIC",
                },
                {
                    "filename": "jw_skip_x1d.fits",
                    "dataset": "obs-2",
                    "uri": "mast:JWST/product/jw_skip_x1d.fits",
                    "instrument_name": "NIRCAM",
                    "access": "PUBLIC",
                },
            ]
        )
        second_product_batch = pd.DataFrame(
            [
                {
                    "filename": "jw_second_i2d.fits",
                    "dataset": "obs-3",
                    "uri": "mast:JWST/product/jw_second_i2d.fits",
                    "instrument_name": "MIRI",
                    "access": "PUBLIC",
                }
            ]
        )
        fake_client = FakeMastMissions(
            [first_observation_batch, second_observation_batch, pd.DataFrame()],
            [first_product_batch, second_product_batch],
        )
        log_messages = []

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "jwst_L3_sky_patches.csv"
            written_rows = MODULE.export_sky_patch_manifest(
                fake_client,
                output_file=output_path,
                batch_size=2,
                sleep_seconds=0,
                max_retries=1,
                logger=log_messages.append,
            )
            exported = pd.read_csv(output_path)

        self.assertEqual(written_rows, 2)
        self.assertEqual(list(exported["productFilename"]), ["jw_first_i2d.fits", "jw_second_i2d.fits"])
        self.assertEqual(len(fake_client.query_calls), 3)
        self.assertEqual(fake_client.query_calls[0]["limit"], 2)
        self.assertEqual(fake_client.query_calls[0]["offset"], 0)
        self.assertEqual(fake_client.query_calls[1]["offset"], 2)
        self.assertEqual(fake_client.query_calls[2]["offset"], 3)
        self.assertEqual(len(fake_client.product_calls), 2)
        self.assertTrue(any("Processed batch 1" in message for message in log_messages))
        self.assertTrue(any("Complete." in message for message in log_messages))


if __name__ == "__main__":
    unittest.main()