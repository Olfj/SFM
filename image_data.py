from image import load_images, images_to_grayscale_arr
from exif import K_and_inv_from_exif

class ImageData:
	'''
	A class to manage image data, including loading images, processing them into grayscale,
	and extracting camera parameters from EXIF data.

	Attributes:
		__INITIAL_PAIR (dict): A private dictionary mapping dataset IDs to initial image pairs.
		__PIXEL_TRESHOLD (dict): A private dictionary mapping dataset IDs to pixel thresholds.
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

		# Args:

		- dataset (int): The dataset ID to load and process.

		# Attributes:
		- images (list): A list of images loaded from the specified dataset path.
		- grayscales (numpy.ndarray): A numpy array of grayscale images derived from the loaded images.
		- K (numpy.ndarray): The camera intrinsic matrix extracted from the first image's EXIF data.
		- K_inv (numpy.ndarray): The inverse of the camera intrinsic matrix.
		- initial_pair (tuple): The initial image pair for the dataset.
		- pixel_treshold (int): The pixel threshold for the dataset.
		'''
		path = 'data/' + str(dataset) + '/'
		self.images, self.paths = load_images(path)  # Load images from the dataset path
		self.grayscales = images_to_grayscale_arr(self.images)  # Convert images to grayscale arrays

		# Extract camera intrinsic matrix and its inverse from EXIF data
		self.K, self.K_inv = K_and_inv_from_exif(self.paths[0])
		self.H, self.W = 2*self.K[0,2], 2*self.K[1,2]

		# Retrieve initial pair and pixel threshold for the given dataset
		self.initial_pair = self.__class__.__INITIAL_PAIR[dataset]
		self.pixel_treshold = self.__class__.__PIXEL_TRESHOLD[dataset]