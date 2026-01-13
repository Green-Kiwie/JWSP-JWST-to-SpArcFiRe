import io
import sys
import unittest
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import patch

original_sys_path = sys.path.copy()
import sep_one_file as sof
sys.path = original_sys_path

import view_thumbnail as vtbn
import sep_helpers as sh



class test_pipeline(unittest.TestCase):
    def setUp(self) -> None:
        self._output_string = io.StringIO()
        sys.stdout = self._output_string
        plt.switch_backend('Agg')
        
    def add_string_to_input(self, input: str):
        sys.stdin = io.StringIO(input)

    def read_output_stream(self) -> str:
        output = self._output_string.getvalue()
        self._output_string = io.StringIO()
        sys.stdout = self._output_string
        return output
    
    def print_debug(self, *print_input) -> None:
        sys.stdout = sys.__stdout__
        print(*print_input)
        sys.stdout = self._output_string
    
    def test_view_thumbnail_fully_working_single_file(self):
        test_filepath = r"test_data/image1/object_9.fits"
        self.add_string_to_input(test_filepath)
        #assume that checking thumbnail information will is sufficient to check function
        vtbn.run()
        output = self.read_output_stream()[47:]

        meta_data = sh.get_all_fits_meta_data(test_filepath)
        vtbn._print_thumbnail_info(meta_data)
        correct_output = self.read_output_stream()

        self.assertEqual(output, correct_output)

    def test_view_thumbnail_fully_working_directory(self):
        test_filepath = r"test_data/image1/"
        self.add_string_to_input(test_filepath)
        vtbn.run()
        output = self.read_output_stream()[23:]

        expected_str = ''
        for file in Path(test_filepath).iterdir():
            meta_data = sh.get_all_fits_meta_data(file)
            vtbn._print_thumbnail_info(meta_data)
            expected_str += "Thumbnail information: \n" + self.read_output_stream()
        
        self.assertEqual(output, expected_str)

    def test_view_thumbanil_given_invalid_filepath(self):
        test_filepath = "testdata/pft.fits"
        self.add_string_to_input(test_filepath)
        try:
            vtbn.run()
        except:
            pass
        output = self.read_output_stream()

        expected = "filename or directory: Invalid file, name. Please give another file name.\nfilename or directory: "

        self.assertEqual(output, expected)

    def test_sep_one_file_fully_working(self):
        test_input = ["sample_data/image3.fits", "N", "N", "N", "0"]
        with patch("builtins.input", side_effect=test_input):
            sof.run()
            output = self.read_output_stream()
        expected = """354 objects found.
path currently exists, deleting file contents.
Directory 'output/image3' created successfully.
0 images exists. All are added to record object for duplicate prevention
Saved: output/image3/object_4.fits
Saved: output/image3/object_10.fits
Saved: output/image3/object_12.fits
Saved: output/image3/object_61.fits
Saved: output/image3/object_102.fits
Saved: output/image3/object_130.fits
Saved: output/image3/object_133.fits
Saved: output/image3/object_176.fits
Saved: output/image3/object_189.fits
Saved: output/image3/object_190.fits
Saved: output/image3/object_192.fits
Saved: output/image3/object_230.fits
Saved: output/image3/object_231.fits
Saved: output/image3/object_264.fits
Saved: output/image3/object_276.fits
Saved: output/image3/object_317.fits
total files cropped: 16\n"""

        self.assertEqual(expected, output)





if __name__ == "__main__":
    unittest.main()