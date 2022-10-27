import numpy as np
import pandas as pd


def get_random_matrix(num_rows, num_columns, seed):

    if type(num_rows) != int or type(num_columns) != int:
        raise TypeError('incompatible data type')

    if type(seed) != int:
        raise TypeError('incompatible data type')

    if num_rows <= 0 or num_columns <= 0 or seed <= 0:
        raise ValueError('all input parameters must be > 0')

    np.random.seed(seed)

    array_2D = np.random.rand(num_rows, num_columns)

    return array_2D


def get_file_dimensions(file_name):
    file_contents = pd.read_csv(file_name, sep=',', header=None)
    file_dim = file_contents.shape
    return file_dim


def write_matrix_to_file(num_rows, num_columns, seed, file_name):
    matrix = get_random_matrix(num_rows, num_columns, seed)
    output_file = np.savetxt(file_name, matrix, delimiter=',', fmt='%1.16f')
    return output_file
