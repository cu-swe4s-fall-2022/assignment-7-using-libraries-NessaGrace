# Using Libraries

## Project Description:
This project creates three functions for processing random matrices
and csv files. Specifically, get_random_matrix() produces a random
matrix with contents sampled from a uniform distribution. The function
get_file_dimensions() produces the file dimensions for a csv file's 
tabular data. The function write_matrix_to_file() writes a matrix
produced with get_random_matrix() to a csv file. All of these
functions are included in data_processor.py, which relies on the
libraries numpy and pandas. The file plotter.py produces a boxplot,
scatterplot, and multi-panel plot of the two prior plots of data
from the csv file iris.data. It uses the libraries matplotlib,
argparse, and pandas.

## A Note on Unit and Functional Tests:
All tests are included in the tests directory. The unit tests are
included in test_data_processor.py, which uses the libraries numpy,
pandas, unittest, os, sys, and csv. All functions in the 
data_processor.py file are tested, with positive, negative, and error
assertions considered to see if the expected output is generated
for each test. The functional tests uses the Stupid Simple baSh
testing framework, available in the tests directory as ssshtest.
The script plotter.py is tested for proper execution here.

## How to Use the Project:

**Files**
Only the file iris.data (the csv file containing all data used in 
plotter.py) is required as input.

**Example**
You can run plotter.py with the following:
python plotter.py --file_name iris.data

You can run the unit tests by:
python tests/test_data_processor.py

You can run the functional tests by:
bash tests/test_plotter.sh

**Command Line Arguments/Parameters**
The file name parameter for plotter.py is the csv file with the iris
dataset.

## How to Install Software:
    - The libraries numpy, pandas, and matplotlib are necessary for this
      project to run correctly. To install them, run the following in the
      command line:
          - `conda install numpy`
          - `conda install pandas`
          - `conda install matplotlib`
    - To run the functional tests using the Stupid Simple baSh Framework,
      you will also need to install the network utility `wget`. `wget` can
      be downloaded from http://gnuwin32.sourceforge.net/packages/wget.htm
      and will need to be installed with `conda install wget` prior to 
      running the functional tests.
