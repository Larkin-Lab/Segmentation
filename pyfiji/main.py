"""main.py: Space for using pH_weka and other jython methods for processing phase images."""

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
macroprocess = segmenter.macroprocess
run = segmenter.run
gpu_run=segmenter.gpu_run

#Do analysis here
t0=time.time()
macroprocess(filenames[100])
print "Time Elapsed: ", time.time()-t0
#
#t0=time.time()
#process(filenames[100])
#print "Time Elapsed: ", time.time()-t0

