#!/bin/bash

test -e ssshtest || wget -q 'https://raw.githubusercontent.com/ryanlayer/ssshtest/master/ssshtest'
. ssshtest

run plotter python ../plotter.py --file_name iris.data
assert_exit_code 0
