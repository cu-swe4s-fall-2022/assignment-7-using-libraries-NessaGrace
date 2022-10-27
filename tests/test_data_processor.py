import unittest
import os
import numpy as np
import pandas as pd
#from pandas.testing import assert_frame_equal
import sys
import csv

sys.path.append('../')

import data_processor as dp # nopep8

class BaseTestCases:
    class BaseTest(unittest.TestCase):
        def common_test(self):
            x = 4
            self.assertEqual(x, 4)

class TestDataProcessor(BaseTestCases.BaseTest):

    def test_get_random_matrix(self):

        # test if 2-D array is produced:

        np.random.seed(7)
        array2D_fcn = dp.get_random_matrix(2, 2, 7)
        fcn_size = array2D_fcn.size
        arr2D_rand = np.random.rand(2, 2)
        test_size = arr2D_rand.size
        self.assertEqual(fcn_size, test_size)
        arr1D_rand = np.random.rand(1, 1)
        test2_size = arr1D_rand.size
        self.assertNotEqual(fcn_size, test2_size)


        # test if random array is produced:

        np.random.seed(7)
        arr_unif_test = np.random.rand(2, 2)
        arr_unif_fail = np.random.rand(1, 1)
        array_unif = dp.get_random_matrix(2, 2, 7)
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

        # test if csv is read in correctly
        #WORKS file_contents_fcn = dp.get_file_dimensions('../iris.data')
       
        #file_contents_fcn.columns = [''] * len(file_contents_fcn.columns)
        #file_contents_fcn = file_contents_fcn.astype('object').dtypes
        #file_contents_fcn = file_contents_fcn.astype('object').dtypes
        #for row in file_contents_fcn:
         #   print(row)
        #print(type(file_contents_fcn))
        
      # WORKS print(file_contents_fcn)
       # rows = []
        #with open('../iris.data', newline='') as csvfile:
         #   file_contents = csv.reader(csvfile, delimiter=',')
          #  for row in file_contents:
           #     file_contents = rows.append(row)
        
        #print(rows)
        
        #WORKS file_contents_test = pd.DataFrame(rows)
        #file_contents_test = file_contents_test.drop(150)
        
        #file_contents_test.iloc[:,0] = None
        # file_contents_test.columns = [''] * len(file_contents_test.columns)
        #file_contents_test.rows = [''] * len(file_contents_test.rows)
        #file_contents_test = file_contents_test.squeeze()
        #print(type(file_contents_test))
        #file_contents_test.astype('float64').dtypes

        #WORKS print(file_contents_test)
        #pd.testing.assert_frame_equal(file_contents_fcn, file_contents_test, check_dtype=False)

        #self.assertEqual(file_contents_fcn, file_contents_test)


        # working test for reading in file, modified for next test
        file_contents_fcn = dp.get_file_dimensions('../iris.data')
        file_contents_test = pd.read_csv('../iris.data', sep=',', header=None)
        self.assertEqual(file_contents_fcn, file_contents_test.shape)

        # test if dimensions of tabular data outputted correctly
        file_dim_fcn = dp.get_file_dimensions('../iris.data')
        num_rows = len(file_contents_test.index)
        columns = file_contents_test.columns
        num_cols = len(columns)
        file_dim_test = (num_rows, num_cols)
        self.assertEqual(file_dim_fcn, file_dim_test)
        self.assertNotEqual(file_dim_fcn, (151,5))


if __name__ == '__main__':
    unittest.main()
