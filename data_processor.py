import numpy as np

def get_random_matrix(num_rows, num_columns, prime):
    #array_2D = np.zeros((num_rows, num_columns))
    #array_2D[0][0] = 1
    #array_2D[0][1] = 2
    #array_2D[1][0] = 3
    #array_2D[1][1] = 4

    np.random.seed(prime)

    array_2D = np.random.rand(num_rows, num_columns)

    return array_2D

def get_file_dimensions(file_name):
	return (0,0)

def write_matrix_to_file(num_rows, num_columns, file_name):
	return None