from PIL import Image
import numpy as np
import numpy.typing as npt
import glob


def load_images(
		path : str
		) -> tuple[list[Image.Image], list[str]]:
	
	'''
	Loads every image in a folder containing (presumably) only images.

	# Parameters 

	path : path to the image folder, should end with ''/''.

	# Returns

	images : list of images loaded using PIL.
	'''

	images = []
	paths = []
	
	for f in glob.iglob(f'{path}*'):

		images.append(Image.open(f))
		paths.append(f)
		

	return images, paths

def images_to_grayscale_arr(
		images : list[Image.Image]
		) -> list[npt.NDArray]:

	grayscales = []

	for img in images:

		gray = img.convert('L')
		grayscales.append(np.array(gray))

	return grayscales