""" Unit tests for functions in data_processor.py

    * test_get_random_matrix - positive and negative tests
      for if 2D, random array is produced, test for error assertions

    * test_get_file_dimensions - test if csv file is read in and
      if proper file dimensions are given

    * test_write_matrix_to_file - test if matrix is written to csv
      file properly
"""

import unittest
import os
import numpy as np
import pandas as pd
import sys
import csv

sys.path.append('../')

import data_processor as dp  # nopep8


class BaseTestCases:
    class BaseTest(unittest.TestCase):
        def common_test(self):
            x = 4
            self.assertEqual(x, 4)


class TestDataProcessor(BaseTestCases.BaseTest):

    def test_get_random_matrix(self):

        # test if 2-D array is produced:

        # seed for reproducibility
        np.random.seed(7)
        array2D_fcn = dp.get_random_matrix(2, 2, 7)
        fcn_size = array2D_fcn.size
        arr2D_rand = np.random.rand(2, 2)
        test_size = arr2D_rand.size
        # test the size of the arrays
        self.assertEqual(fcn_size, test_size)
        arr1D_rand = np.random.rand(1, 1)
        test2_size = arr1D_rand.size
        # negative assertion for array size
        self.assertNotEqual(fcn_size, test2_size)

        # test if random array is produced:

        # seed for reproducibility
        np.random.seed(7)
        arr_unif_test = np.random.rand(2, 2)
        arr_unif_fail = np.random.rand(1, 1)
        array_unif = dp.get_random_matrix(2, 2, 7)
        # positive and negative tests for proper array output
        np.testing.assert_array_equal(arr_unif_test, array_unif)
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_array_equal,
                                 arr_unif_fail,
                                 array_unif)

        # test error raising:

        self.assertRaises(TypeError, dp.get_random_matrix, 'a', 'b', 7)
        self.assertRaises(TypeError, dp.get_random_matrix, 2, 2, 'a')
        self.assertRaises(ValueError, dp.get_random_matrix, -1, 0, 7)
        self.assertRaises(ValueError, dp.get_random_matrix, 2, 2, 0)

    def test_get_file_dimensions(self):

        # test if csv is read in correctly, modified for next test
        file_contents_fcn = dp.get_file_dimensions('../iris.data')
        file_contents_test = pd.read_csv('../iris.data', sep=',', header=None)
        self.assertEqual(file_contents_fcn, file_contents_test.shape)

        # test if dimensions of tabular data outputted correctly
        file_dim_fcn = dp.get_file_dimensions('../iris.data')
        num_rows = len(file_contents_test.index)
        columns = file_contents_test.columns
        num_cols = len(columns)
        file_dim_test = (num_rows, num_cols)
        # positive and negative tests for file dimensions
        self.assertEqual(file_dim_fcn, file_dim_test)
        self.assertNotEqual(file_dim_fcn, (151, 5))

    def test_write_matrix_to_file(self):
        np.random.seed(7)
        file_matrix_fcn = dp.write_matrix_to_file(2, 2, 7, 'twoDArray.csv')
        matrix = dp.get_random_matrix(2, 2, 7)
        with open('twoDArrayTest.csv', 'w', newline='') as f:
            writer = csv.writer(f, delimiter=',', )
            for row in matrix:
                writer.writerow(row)

        # test if the two files have the same dimensions
        self.assertEqual(dp.get_file_dimensions('twoDArray.csv'),
                         dp.get_file_dimensions('twoDArrayTest.csv'))
        self.assertNotEqual(dp.get_file_dimensions('twoDArray.csv'),
                            dp.get_file_dimensions('../iris.data'))


if __name__ == '__main__':
    unittest.main()
