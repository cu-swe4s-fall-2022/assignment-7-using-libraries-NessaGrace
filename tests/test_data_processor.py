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

    # note: class set up, tear down used for previous test versions
#    @classmethod
 #   def setUpClass(cls):
  #      cls.arr_2D = np.array([[1,2], [3,4]], dtype='f')
   #     cls.arr_1D = np.array([1,2], dtype='f')
    #    cls.arr_3D = np.array([[1,2], [3,4], [5,6]], dtype='f')


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


if __name__ == '__main__':
    unittest.main()
