import numpy as np
import pandas as pd


def get_random_matrix(num_rows, num_columns, seed):
    """Generate matrix of random floating point numbers
       sampled from a uniform distribution

       Parameters
       ----------
       num_rows: integer > 0
                 Number of rows in matrix being generated.

       num_columns: integer > 0
                    Number of columns in matrix desired.

       seed: prime integer > 0
             Seed value to get reproducible results.

       Returns
       -------
       array_2D: numpy array
                 Matrix of random floating point numbers
                 sampled from a uniform distribution [0, 1)
    """

    # check for improper input
    if type(num_rows) != int or type(num_columns) != int:
        raise TypeError('incompatible data type')

    if type(seed) != int:
        raise TypeError('incompatible data type')

    if num_rows <= 0 or num_columns <= 0 or seed <= 0:
        raise ValueError('all input parameters must be > 0')

    # feed in seed for reproducible random results
    np.random.seed(seed)

    array_2D = np.random.rand(num_rows, num_columns)

    return array_2D


def get_file_dimensions(file_name):
    """Produce file dimensions of a csv file of tabular data.

       Parameters:
       ----------
       file_name: string
                  The name of the csv file.

       Returns:
       --------
       file_dim: tuple
                 Dimensions of the file's tabular data.
    """

    file_contents = pd.read_csv(file_name, sep=',', header=None)
    file_dim = file_contents.shape
    return file_dim


def write_matrix_to_file(num_rows, num_columns, seed, file_name):
    """Write a random matrix sampled from a uniform distribution
       to a csv file.

       Parameters:
       ----------
       num_rows: integer > 0
                 Number of rows in matrix being generated

       num_columns: integer > 0
                    Number of columns in matrix being generated

       seed: integer > 0
             Seed value to get reproducible results

       file_name: string
                  Name of file to write matrix to

       Returns:
       -------
       output_file: csv file
                    Csv file that contains the matrix.
    """

    matrix = get_random_matrix(num_rows, num_columns, seed)
    output_file = np.savetxt(file_name, matrix, delimiter=',', fmt='%1.16f')
    return output_file
