import numpy as np

def get_random_matrix(num_rows, num_columns, prime):

    np.random.seed(prime)

    array_2D = np.random.rand(num_rows, num_columns)

    return array_2D
    raise TypeError('incompatible data type')

def get_file_dimensions(file_name):
	return (0,0)

def write_matrix_to_file(num_rows, num_columns, file_name):
	return None
