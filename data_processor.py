import numpy as np

def get_random_matrix(num_rows, num_columns, prime):

    if type(num_rows) != int or type(num_columns) != int:
        raise TypeError('incompatible data type')

    if type(prime) != int:
        raise TypeError('incompatible data type')

    if num_rows <= 0 or num_columns <=0 or prime <= 0:
        raise ValueError('all input parameters must be > 0')

    np.random.seed(prime)

    array_2D = np.random.rand(num_rows, num_columns)

    return array_2D

def get_file_dimensions(file_name):
	return (0,0)

def write_matrix_to_file(num_rows, num_columns, file_name):
	return None
