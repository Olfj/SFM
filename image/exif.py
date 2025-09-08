import numpy as np
import numpy.typing as npt
import piexif

def K_and_inv_from_exif(
		img_path : str
		) -> tuple[npt.NDArray, npt.NDArray]:
	
	'''
	Creates the calibration matrix K from the exif data of an image. Returns K and its inverse.

	### Parameters
	- img_path (str): path to image file.

	### Returns
	- K (np.ndarray): 
		3x3 matrix according to:

			pixel_x = number of pixels in a horizontal image row.
			pixel_y = like pixel_x but vertical.
			f = focal length in pixels = pixel_x * (focal_35/35).
			K = [[f, 0, pixel_x/2],
				 [0, f, pixel_y/2],
				 [0, 0, 1]]

	- K_inv (np.ndarray): the iverse of K .
	'''
	
	exif_dict = piexif.load(img_path)
	
	pixel_x = exif_dict["Exif"].get(piexif.ExifIFD.PixelXDimension)
	pixel_y = exif_dict["Exif"].get(piexif.ExifIFD.PixelYDimension)
	focal_35 = exif_dict["Exif"].get(piexif.ExifIFD.FocalLengthIn35mmFilm)

	# Fallback to ImageWidth/ImageLength if missing x/y dimension
	if pixel_x is None:
		pixel_x = exif_dict["0th"].get(piexif.ImageIFD.ImageWidth)
	if pixel_y is None:
		pixel_y = exif_dict["0th"].get(piexif.ImageIFD.ImageLength)

	f = pixel_x * (focal_35/35)
	K = np.array(
		[[f, 0, pixel_x/2],
		 [0, f, pixel_y/2],
		 [0, 0, 1]])
	
	K_inv = np.linalg.inv(K)
	
	return K, K_inv
