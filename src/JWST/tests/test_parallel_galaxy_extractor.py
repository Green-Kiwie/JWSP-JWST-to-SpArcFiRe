import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "parallel_galaxy_extractor.py"
SPEC = importlib.util.spec_from_file_location("parallel_galaxy_extractor", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParallelGalaxyExtractorTests(unittest.TestCase):
    def test_build_thumbnail_output_path_prefixes_instrument_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename, output_path = MODULE.build_thumbnail_output_path(
                tmpdir,
                80.483566,
                -69.482915,
                "nircam",
                "clear-f070w",
                1,
                1,
                10000,
            )

            self.assertEqual(
                filename,
                "80.483566_-69.482915_nircam_clear-f070w_v1.fits",
            )
            self.assertEqual(Path(output_path).name, filename)

    def test_build_thumbnail_output_path_keeps_boundary_instance_in_first_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename, output_path = MODULE.build_thumbnail_output_path(
                tmpdir,
                10.0,
                20.0,
                None,
                "f200w",
                10000,
                10000,
                10000,
            )

            self.assertEqual(Path(output_path).parent.name, "1")
            self.assertTrue((Path(tmpdir) / "1").is_dir())
            self.assertEqual(Path(output_path).name, filename)

    def test_build_thumbnail_output_path_rolls_over_after_capacity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filename, output_path = MODULE.build_thumbnail_output_path(
                tmpdir,
                10.0,
                20.0,
                None,
                "f200w",
                10001,
                10001,
                10000,
            )

            self.assertEqual(Path(output_path).parent.name, "2")
            self.assertTrue((Path(tmpdir) / "2").is_dir())
            self.assertEqual(Path(output_path).name, filename)

    def test_build_thumbnail_output_path_rejects_invalid_capacity(self):
        with self.assertRaises(ValueError):
            MODULE.build_thumbnail_output_path(
                "/tmp",
                10.0,
                20.0,
                None,
                "f200w",
                1,
                1,
                0,
            )


if __name__ == "__main__":
    unittest.main()