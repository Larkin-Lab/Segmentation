"""pH_weka.py: Contains functions for segmentation of phase contrast images, using manually trained Weka classifiers as the backend."""

__author__ = "Maya Peters-Kostman", "Joseph Incandela"
__credits__ = ["Maya Peters-Kostman", "Joseph Incandela"]

import os
import sys
import time
from trainableSegmentation import WekaSegmentation, Weka_Segmentation
from ij import IJ, ImagePlus
from java.util.concurrent import Callable
from java.util.concurrent import Executors, TimeUnit

class WekaPhase(object):
	def __init__(self,source_dir=None,output_dir=None,classifier_path=None):
		self.srcdir = source_dir
		self.outdir = output_dir
		self.classpath = classifier_path
		#Set up data files to be iterate over, filter for only channel 1 .tiffs
		for self.root, self.directories, self.filenames in os.walk(source_dir):
			self.filenames.sort();
			for filename in self.filenames: # Check for file extension
				if not filename.endswith('.tif') or 'C0001' not in filename:
					self.filenames.remove(filename)
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)
			print "Created new directory {}".format(output_dir)
		print "Number of phase images to process: ", len(self.filenames)
		return None
	
#	def run(self,threadcount=None):
#		'''Function run allows function process to be run in parallel with varying number of threads.'''
#		pool = Executors.newFixedThreadPool(threadcount) # Define Threads
#		process = [self.process(filename=image) for image in self.filenames] # define the task to do in a multithreaded way
#		pool.invokeAll(process) # use all defined threads to process the images
#		self.shutdown_and_await_termination(pool=pool, timeout=5)
#		return None
#	
	def run(self):
		'''Iterate process function over all pH tif files.'''
		for filename in self.filenames:
			self.process(filename)
		print "run complete"
		
	def process(self, filename=None):
		'''Function perfoming the segmentation of a phase contrast image with trained weka function.'''
		image = IJ.openImage(os.path.join(self.srcdir,filename))
		weka = WekaSegmentation() # Create weka instance.
		weka.setTrainingImage(image) # Set image to segment (Not in init)
		weka.loadClassifier(self.classpath) # loads manually trained classifier
		segmented_image = weka.applyClassifier(image, 0, False) # apply classifier and get results. (Not in init)
		# assign same LUT as in GUI. Within WEKA GUI, right-click on classified image and use Command Finder to save the "LUT" within Fiji.app\luts
		#lut = LutLoader.openLut(r'C:\Users\angu312\Documents\Fiji.app\luts\Classification result.lut')
		#segmented_image.getProcessor().setLut(lut)
		# Save the image as a .tif file
		IJ.saveAs(segmented_image, "Tif", os.path.join(self.outdir, filename.replace('.tif', '') + '_weka'))
		image.close()
		return None

	def shutdown_and_await_termination(self,pool=None, timeout=None):
		'''Function for shutting down the pool taken from: http://www.jython.org/jythonbook/en/1.0/Concurrency.html'''
		pool.shutdown()
		try:
			if not pool.awaitTermination(timeout, TimeUnit.SECONDS):
				pool.shutdownNow()
			if (not pool.awaitTermination(timeout, TimeUnit.SECONDS)):
				print >> sys.stderr, "Pool did not terminate"
		except InterruptedException, ex:
			# (Re-)Cancel if current thread also interrupted
			pool.shutdownNow()
			# Preserve interrupt status
			Thread.currentThread().interrupt()
		return None