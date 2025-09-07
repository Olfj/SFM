import numpy as np
import numpy.typing as npt
import cv2 as cv

from typing import Sequence
from tqdm import tqdm
from essential import estimate_E_robust
from triangulate import triangulate
from image_data import ImageData

def X_and_descript_from_inital_pair(Rs, images : ImageData, eps):

	cam1, cam2 = images.initial_pair
	R = Rs[cam1]
	
	_, descriptors, matches, x1s, x2s = SIFT_matches([images.grayscales[cam1], images.grayscales[cam2]], verbose=True)
	
	X_descript = [descriptors[0][m.queryIdx] for m in matches[0]]
	match_index = [m.queryIdx for m in matches[0]]

	x1, x2 = images.K_inv @ x1s[0], images.K_inv @ x2s[0]
	E, inlier, _ = estimate_E_robust(x1, x2, images.K, eps)
	x1, x2 = x1[:,inlier], x2[:,inlier]
	_, X = triangulate(E, x1, x2)
	X = R.T @ X[:-1]
	
	X_descript = np.array([X_descript[i] for i in range(len(X_descript)) if inlier[i]])
	match_index = [match_index[i] for i in range(len(match_index)) if inlier[i]]
	
	return X, X_descript, match_index

def x_X_from_descript(X, X_descript, keys, descriptors):

	bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

	xs, Xs = [], []

	for i in range(len(keys)):

		desc = descriptors[i]
		key = keys[i]

		match = bf.match(X_descript, desc)
		match = sorted(match, key=lambda x: x.distance)
		
		#filter away the 20% worst matches
		num_matches = len(match)
		cutoff = int(num_matches*0.8)
		match = match[:cutoff]

		xi = np.ones(shape=(3,len(match)))
		Xi = np.zeros(shape=(3, len(match)))

		for i in range(len(match)):
			Xi[:,i] = X[:,match[i].queryIdx]
			xi[:-1,i] = key[match[i].trainIdx].pt

		xs.append(xi)
		Xs.append(Xi)
	
	return xs, Xs

def SIFT_matches(
		images : list[npt.NDArray], 
		verbose : bool = True
		) -> tuple[
			list[list[cv.KeyPoint]],
			list[npt.NDArray],
			list[Sequence[cv.DMatch]],
			list[npt.NDArray], 
			list[npt.NDArray]
			]:
	
	keys, descriptors = SIFT_points(*images, verbose=verbose)
	matches = match_SIFT_points(descriptors, verbose)
	x1s, x2s = extract_matches(matches, keys, verbose)

	return keys, descriptors, matches, x1s, x2s


def SIFT_points(
		*images : npt.NDArray, 
		verbose : bool = True
		) -> tuple[list[list[cv.KeyPoint]], list[npt.NDArray]]:
	
	'''
	# Parameters:

	images: NDArrays of grayscale values.

	# Return:

	keys: a list containing a list of N keypoints for every image as type cv2.KeyPoint.

	descriptors: a list  containing an N x 128 NDArray of descriptors for the same images.	
	'''

	keys = []
	descriptors = []
	sift = cv.SIFT.create()

	for image in tqdm(images, disable=not(verbose), desc='Calculating SIFT points:'):
		key_points, description = sift.detectAndCompute(image, None) # type: ignore
		keys.append(key_points)
		descriptors.append(description)

	return keys, descriptors
	
def match_SIFT_points(
		descriptors : list[npt.NDArray],
		verbose : bool = True
		) -> list[Sequence[cv.DMatch]]:

	'''

	# Parameters:

	- keys: a list containing a list of cv2.KeyPoint for each image.

	- descriptors: a list containing the corresponding descriptors for each image.

	# Returns: 

	- matches: a list of matches between images in order, i.e. matches between image 1 and image 2, image 2 and image 3 and so on.

	'''

	num_images = len(descriptors)
	matches = []
	bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

	for i in tqdm(range(num_images-1), disable=not(verbose), desc='Matching SIFT points between images:'):
		match = bf.match(descriptors[i], descriptors[i+1])
		match = sorted(match, key=lambda x: x.distance)
		
		#filter away the 20% worst matches
		num_matches = len(match)
		cutoff = int(num_matches*0.8)
		match = match[:cutoff]

		matches.append(match)
	
	return matches

def extract_matches(
		matches : list[Sequence[cv.DMatch]],
		keys : list[list[cv.KeyPoint]],
		verbose : bool = True
		) -> tuple[list[npt.NDArray], list[npt.NDArray]]:

	x1s = []
	x2s = []
	
	for i in tqdm(range(len(matches)), disable=not(verbose), desc='Extracting points from matches:'):
		
		x1 = np.ones((3, len(matches[i])))
		x2 = np.ones_like(x1)

		for j, match in enumerate(matches[i]):

			x1[:-1,j] = keys[i][match.queryIdx].pt
			x2[:-1,j] = keys[i+1][match.trainIdx].pt
		
		x1s.append(x1)
		x2s.append(x2)
	
	return x1s, x2s
