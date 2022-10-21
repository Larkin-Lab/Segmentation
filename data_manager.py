"""
Created on Fri Oct 21 16:16:09 2022

@author: Joey Incnandela
"""
import os
import sys
import traceback
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

class Manager(object):
	"""Initialize this class in a folder with images in .tif format. 
	Used to manage pre-processed image data located within initialization folder."""
	def __init__(self,data_path=None):
		self.data_path = data_path
		self.allfiles = os.listdir(self.data_path)

	def list_data(self,imtype=".tif"):
		"""Get list of specific image data types stored in local folder. This function is general and works for more than just .tif."""
		data_list = [stack for stack in self.allfiles if imtype.upper() or imtype.lower() in stack]
		return data_list

	def check_file(self,filename=None):
		'''Test filename and determine dimensionality to check file can be properly operated on by the data manager methods. 
		Max size is 4D numpy array for multichannel timelapse.
		If file passes the check, return information on the file.'''
		filename=str(filename)
		if '.tif' not in filename:
			raise ExtensionError()
		else:
			if filename not in self.allfiles:
				raise FileNameError(filename,self.data_path)
			else:
				file_path = self.data_path+'\\'+filename
				stack_shape = io.imread(file_path).shape
				dimensions = len(stack_shape)
				if dimensions not in range(2,5):
					raise DimensionalityError()
				else:
					print('"{}" File Size: '.format(filename), os.path.getsize(file_path)//1000,'KB')
					print('Image Stack Shape ({}D): '.format(dimensions),stack_shape)
					# For a single .tif image 
					if dimensions==2:
						print('Image Size: ',stack_shape[0] ,' x ',stack_shape[1])
					# For a single-channel .tif stack
					elif dimensions==3:
						print('Number of images in stack: ', stack_shape[0])
						print('Image Size: ',stack_shape[1] ,' x ',stack_shape[2])
					#For a multi-channel .tif stack
					elif dimensions==4:
						print('Number of channels in stack: ', stack_shape[1])
						print('Number of images per channel: ', stack_shape[0])
						print('Image Size: ',stack_shape[2] ,' x ',stack_shape[3])
					return stack_shape, dimensions, file_path;

	def preview(self, filename=None,index=0):
		"""Get a preview of image data file, without loading all of it to memory. For multichannel stacks, displays one image per channel."""
		stack_shape, dimensions, file_path = self.check_file(filename);
		# For a single .tif image 
		if dimensions==2:
			preview_image = io.imread(file_path)
			io.imshow(preview_image)
			plt.show()
			input('Press Any Key')
			plt.close()
		# For a single-channel .tif stack
		elif dimensions==3:
			preview_image = io.imread(file_path)[index]
			io.imshow(preview_image)
			plt.show()
			input('Press Any Key')
			plt.close()
		#For a multi-channel .tif stack
		elif dimensions==4:
			for i in range(stack_shape[1]):
				preview_image = io.imread(file_path)[index]
				print('\n\n Viewing Channel {}'.format(i))
				io.imshow(preview_image[i])
				plt.show()
				input('Press Any Key')
				plt.close()
		return preview_image

	def read(self,filename=None):
		'''Read in a .tif image file and cast it as a the appropriate ND array size.'''
		file_path=self.check_file(filename)[2];
		return io.imread(file_path)



class ExtensionError(Exception):
	"""Raised if image file extension is not a tiff."""
	def __init__(self):
		super().__init__('Image/Stack must be in .tiff format')

class FileNameError(Exception):
	"""Raised if filename is not in folder."""
	def __init__(self, filename=None,folder=None):
		super().__init__('No file named "{}" in folder "{}"'.format(filename,folder))

class DimensionalityError(Exception):
	"""Raised if image dimensions are outside operable range."""
	def __init__(self):
		super().__init__('Image/Stack dimensionality is not within operable range.')
