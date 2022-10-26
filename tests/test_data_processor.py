import unittest
import os
import numpy as np
import sys

sys.path.append('../')

import data_processor as dp # nopep8

class BaseTestCases:
    class BaseTest(unittest.TestCase):
        def common_test(self):
            x = 4
            self.assertEqual(x, 4)

class TestDataProcessor(BaseTestCases.BaseTest):

    @classmethod
    def setUpClass(cls):
        cls.arr_2D = np.array([[1,2,3,4], [5,6,7,8]])
        cls.arr_1D = np.array([1,2,3,4])
        cls.arr_3D = np.array([[1,2,3,4], [5,6,7,8], [1,2,3,4]])

    @classmethod
    def tearDownClass(cls):
        cls.arr_2D = None
        cls.arr_1D = None
        cls.arr_3D = None

    def test_get_random_matrix(self):

        # test if 2-D array is produced
        array2D_test = self.arr_2D
        array1D_test = self.arr_1D
        array3D_test = self.arr_3D
        array2D_fcn = dp.get_random_matrix(2, 2)
        self.assertEqual(array2D_test, array2D_fcn)
        self.assertNotEqual(array2D_fcn, array1D_test)
        self.assertNotEqual(array2D_fcn, array3D_test)


if __name__ == '__main__':
    unittest.main()
