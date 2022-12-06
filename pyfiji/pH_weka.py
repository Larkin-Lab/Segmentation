"""pH_weka.py: Contains functions for segmentation of phase contrast images, using manually trained Weka classifiers as the backend."""

__author__ = "Maya Peters-Kostman", "Joseph Incandela"
__credits__ = ["Maya Peters-Kostman", "Joseph Incandela"]

import os
import sys
from trainableSegmentation import WekaSegmentation, Weka_Segmentation
from ij import IJ, ImagePlus


class WekaPhase(object):
	def __init__(self,source_dir=None,output_dir=None,classifier_dir=None):
		os.chdir('')
		self.srcdir = source_dir
		self.outdir = output_dir
		self.classdir = classifier_dir
		#Set up data files to be iterate over, filter for only channel 1 .tiffs
		for self.root, self.directories, self.filenames in os.walk(source_dir):
			self.filenames.sort();
			for filename in self.filenames:
				# Check for file extension
				if not filename.endswith('.tif') or not 'C0001' in filename:
					self.filenames.remove(filename)
		return None
		
	def run(self):
		for self.root, self.directories, self.filenames in os.walk(self.srcdir):
			self.filenames.sort();
			for filename in self.filenames:
				# Check for file extension
				if not filename.endswith('.tif') or not 'C0001' in filename:
					continue
				process(srcDir, dstDir, root, filename, classifier_Dir)
			
	def process(self, srcDir, dstDir, currentDir, fileName, classifier_Dir):
		print "Processing:"
		
		# Opening the image
		print "Open image file", fileName
		image = IJ.openImage(os.path.join(srcDir, fileName))
		
		weka = WekaSegmentation()
		weka.setTrainingImage(image)
		
		# loads manually trained classifier
		weka.loadClassifier(classifier_Dir)
		# apply classifier and get results
		segmented_image = weka.applyClassifier(image, 0, False)
		# assign same LUT as in GUI. Within WEKA GUI, right-click on classified image and use Command Finder to save the "LUT" within Fiji.app\luts
		#lut = LutLoader.openLut(r'C:\Users\angu312\Documents\Fiji.app\luts\Classification result.lut')
		#segmented_image.getProcessor().setLut(lut)
		
		# Saving the image as a .tif file
		saveDir = dstDir
		if not os.path.exists(saveDir):
			os.makedirs(saveDir)
		print "Saving to", saveDir
		IJ.saveAs(segmented_image, "Tif", os.path.join(saveDir, fileName.replace('.tif', '') + '_weka'))
		image.close()