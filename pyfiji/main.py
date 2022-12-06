"""pH_weka.py: Contains functions for segmentation of phase contrast images, using manually trained Weka classifiers as the backend."""

__author__ = "Maya Peters-Kostman", "Joseph Incandela"
__credits__ = ["Maya Peters-Kostman", "Joseph Incandela"]

import os
import sys
#Find path to pyfiji folder
scriptpath = sys.argv[0]
pyfiji_path = os.path.dirname(scriptpath)
#Append sys.path to enable import of desired classes from pyfiji scripts.
sys.path.append(str(pyfiji_path))
from pH_weka import WekaPhase

# Set source directory for images
srcdir = r'F:\Experiments\{}\tifs'.format(exp)
# Set directory for segmentation output
outdir = r'F:\Experiments\{}\weka segmentation'.format(exp)
# Choose which trained segmentation model to use with dataset
classdir = r'F:\scripts\imagej/weka border segmentation 3.model'


#Do analysis here
