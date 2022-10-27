import unittest
import os
import numpy as np
import pandas as pd
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

    #@classmethod
    #def setUpClass(cls):
       # cls.test_csv = "iris.data"
      #  f = open(cls.test_csv, 'r')

       # test = ['apple', 'orange', 'strawberry', 'banana', 'kiwi']

        #for string in test:
         #   f.write(string + '\n')

     #   f.close()

    #@classmethod
    #def tearDownClass(cls):
        #os.remove(cls.test_csv)


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
        file_contents_fcn = dp.get_file_dimensions('../iris.data')
        with open('../iris.data', newline='') as csvfile:
            file_contents_test = csv.reader(csvfile, delimiter=',')
            #for row in file_contents_test:
             #   print(' '.join(row))
        self.assertEqual(file_contents_fcn, file_contents_test)


if __name__ == '__main__':
    unittest.main()
