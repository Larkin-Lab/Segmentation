"""pH_weka.py: Contains functions for segmentation of phase contrast images, using manually trained Weka classifiers as the backend."""

__author__ = "Maya Peters-Kostman", "Joseph Incandela"
__credits__ = ["Maya Peters-Kostman", "Joseph Incandela"]

import os
import sys
import time
#Find path to pyfiji folder
scriptpath = sys.argv[0]
pyfiji_path = os.path.dirname(scriptpath)
#Append sys.path to enable import of desired classes from pyfiji scripts.
sys.path.append(str(pyfiji_path))
from pH_weka import WekaPhase


# Set source directory for images
srcdir = r'C:\Users\jtincan\Desktop\F0312\tifs'
# Set directory for segmentation output
outdir = r'C:\Users\jtincan\Desktop\F0312\weka_segmentation'
# Choose which trained segmentation model to use with dataset. Models are in Github\Segmentation\weka_models folder.
classpath = os.path.join(os.path.dirname(pyfiji_path),r'weka_models\Cellasic_BO4F_MPK_12062022.model') 

#initialize WekaPhase class
segmenter = WekaPhase(source_dir=srcdir,output_dir=outdir,classifier_path=classpath)
filenames = segmenter.filenames
process = segmenter.process
run = segmenter.run

#Do analysis here
#t0=time.time()
#run()
#print "Time Elapsed: ", time.time()-t0