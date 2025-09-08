from image.exif import K_and_inv_from_exif
from PIL import Image
import glob
import numpy as np
import numpy.typing as npt

class ImageData:
	'''
	A class to manage image data, including loading images, processing them into grayscale,
	and extracting camera parameters from EXIF data.

	### Attributes:
	- images (list[Image]): a list of PIL images.
	- paths (list[str]): a list of the paths to each image.
	- K (np.ndarray): the calibration matrix of the camera used to take the images.
	- K_inv (np.ndarray): the inverse of K.
	- H, W (float, float): The height and width of the images.
	- initial_pair (tuple[int, int]): the initial pair to initialize stereo vision.
	- pixel treshold (int): the treshold for error when deciding what sift points are inliers and not. 

	'''

	__INITIAL_PAIR = {
			1: (0, 1), 2: (0, 8), 3: (3, 5), 4: (4, 9), 5: (3, 6),
			6: (1, 3), 7: (2, 5), 8: (3, 6), 9: (3, 5), 10: (3, 9), 
			11: (0, 5), 12: (12, 20)
	}
	
	__PIXEL_TRESHOLD = {
			1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12:1
	}


	def __init__(self, dataset: int) -> None:
		'''
		Initializes the ImageData object for a specific dataset.

		### Args:
		- dataset (int): The dataset ID to load and process.
		'''
		
		path = 'data/' + str(dataset) + '/'
		self.images, self.paths = self._load_images(path)  # Load images from the dataset path
		self.grayscales = self._images_to_gray_arr()  # Convert images to grayscale arrays

		# Extract camera intrinsic matrix and its inverse from EXIF data
		self.K, self.K_inv = K_and_inv_from_exif(self.paths[0])
		self.H, self.W = 2*self.K[0,2], 2*self.K[1,2]

		# Retrieve initial pair and pixel threshold for the given dataset
		self.initial_pair = self.__class__.__INITIAL_PAIR[dataset]
		self.pixel_treshold = self.__class__.__PIXEL_TRESHOLD[dataset]

	def _load_images(self, path
			) -> tuple[list[Image.Image], list[str]]:
		
		'''
		Loads every image in a folder containing (presumably) only images.

		### Parameters:
		- path (str): path to the image folder, should end with ''/''.
		
		### Returns
		- images (list[Image.Image]): list of images loaded using PIL.
		- paths (list[str]): list of paths to each specific image
		'''

		images = []
		paths = []
		
		for f in glob.iglob(f'{path}*'):

			images.append(Image.open(f))
			paths.append(f)
			

		return images, paths
	
	def _images_to_gray_arr(self,) -> list[npt.NDArray]:
		'''
		Conwerts a PIL image to an array of grayscale values.
		### Parameters:
		- images (list[Image.Image]): a list of PIL images.
		### Returns:
		- grayscales (list[np.ndarray]): a list of the images in grayscale format, stored in ndarrays.
		'''

		grayscales = []

		for img in self.images:

			gray = img.convert('L')
			grayscales.append(np.array(gray))

		return grayscales