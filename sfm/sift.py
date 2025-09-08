import numpy as np
import numpy.typing as npt
import cv2 as cv

from typing import Sequence
from tqdm import tqdm
from essential import estimate_E_robust
from triangulate import triangulate
from image_data import ImageData


def X_and_descript_from_inital_pair(
		Rs : list[npt.NDArray], 
		images: ImageData, 
		eps 
	) -> tuple[npt.NDArray, npt.NDArray, list[int]]:
	'''
	Estimate initial 3D points and corresponding descriptors from the initial image pair.
	
	### Parameters
	- Rs (list[np.ndarray]): List of rotation matrices for all cameras.
	- images (ImageData): Dataset object containing grayscale images, intrinsic calibration, and initial pair info.
	- eps (float): Noise tolerance used for robust estimation of the essential matrix.

	### Returns
	- X (np.ndarray): Triangulated 3D points in camera 1 coordinates, shape (3, N).
	- X_descript (np.ndarray): SIFT descriptors corresponding to the triangulated points, shape (N, 128).
	- match_index (list[int]): Indices of keypoints in the first image that matched successfully.
	'''
	
	cam1, cam2 = images.initial_pair
	R = Rs[cam1]

	_, descriptors, matches, x1s, x2s = sift_matches(
		[images.grayscales[cam1], images.grayscales[cam2]], verbose=True
	)

	X_descript = [descriptors[0][m.queryIdx] for m in matches[0]]
	match_index = [m.queryIdx for m in matches[0]]

	x1, x2 = images.K_inv @ x1s[0], images.K_inv @ x2s[0]
	E, inlier, _ = estimate_E_robust(x1, x2, images.K, eps)
	x1, x2 = x1[:, inlier], x2[:, inlier]
	_, X = triangulate(E, x1, x2)
	X = R.T @ X[:-1]

	X_descript = np.array([X_descript[i] for i in range(len(X_descript)) if inlier[i]])
	match_index = [match_index[i] for i in range(len(match_index)) if inlier[i]]

	return X, X_descript, match_index


def x_X_from_descript(
		X : npt.NDArray, 
		X_descript : npt.NDArray, 
		keys : list[list[cv.KeyPoint]], 
		descriptors : list[npt.NDArray]
	) -> tuple[list[npt.NDArray], list[npt.NDArray]]:
	'''
	Match new 2D keypoints to existing 3D points using descriptors.
	
	### Parameters
	- X (np.ndarray): Existing 3D points, shape (3, N).
	- X_descript (np.ndarray): Descriptors corresponding to X, shape (N, 128).
	- keys (list[list[cv.KeyPoint]]): List of keypoints for each new image.
	- descriptors (list[np.ndarray]): List of descriptors corresponding to `keys`.

	### Returns
	- xs (list[np.ndarray]): List of matched 2D homogeneous points for each image, shape (3, M).
	- Xs (list[np.ndarray]): List of corresponding matched 3D points, shape (3, M).
	'''
	
	bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
	xs, Xs = [], []

	for i in range(len(keys)):
		desc = descriptors[i]
		key = keys[i]

		match = bf.match(X_descript, desc)
		match = sorted(match, key=lambda x: x.distance)

		# filter away the 20% worst matches
		num_matches = len(match)
		cutoff = int(num_matches * 0.8)
		match = match[:cutoff]

		xi = np.ones(shape=(3, len(match)))
		Xi = np.zeros(shape=(3, len(match)))

		for i in range(len(match)):
			Xi[:, i] = X[:, match[i].queryIdx]
			xi[:-1, i] = key[match[i].trainIdx].pt

		xs.append(xi)
		Xs.append(Xi)

	return xs, Xs


def sift_matches(
		images: list[npt.NDArray], verbose: bool = True
	) -> tuple[
		list[list[cv.KeyPoint]],
		list[npt.NDArray],
		list[Sequence[cv.DMatch]],
		list[npt.NDArray],
		list[npt.NDArray]]:
	'''
	Compute SIFT keypoints, descriptors, and matches between consecutive images.

	### Parameters:
	- images (list[np.ndarray]): List of grayscale images.
	- verbose (bool = True): If True, shows progress with tqdm.

	### Returns:
	- keys (list[list[cv.KeyPoint]]): Keypoints for each image.
	- descriptors (list[np.ndarray]): Descriptors for each image.
	- matches (list[Sequence[cv.DMatch]]): Matches between consecutive image pairs.
	- x1s (list[np.ndarray]): Homogeneous 2D coordinates from image i, shape (3, N).
	- x2s (list[np.ndarray]): Homogeneous 2D coordinates from image i+1, shape (3, N).
	'''
	
	keys, descriptors = SIFT_points(*images, verbose=verbose)
	matches = match_SIFT_points(descriptors, verbose)
	x1s, x2s = extract_matches(matches, keys, verbose)

	return keys, descriptors, matches, x1s, x2s


def SIFT_points(
		*images: npt.NDArray, 
		verbose: bool = True
	) -> tuple[list[list[cv.KeyPoint]], list[npt.NDArray]]:
	'''
	Extract SIFT keypoints and descriptors from one or more images.
	
	### Parameters:
	- images (np.ndarray): Grayscale images as numpy arrays.
	- verbose (bool = true): If True, shows progress with tqdm.

	### Returns:
	- keys (list[list[cv.KeyPoint]]): Detected keypoints for each image.
	- descriptors (list[np.ndarray]): Corresponding descriptors, shape (N, 128) for each image.
	'''
	
	keys = []
	descriptors = []
	sift = cv.SIFT.create()

	for image in tqdm(images, disable=not(verbose), desc="Calculating SIFT points:"):
		key_points, description = sift.detectAndCompute(image, None)  # type: ignore
		keys.append(key_points)
		descriptors.append(description)

	return keys, descriptors


def match_SIFT_points(
		descriptors: list[npt.NDArray], 
		verbose: bool = True
	) -> list[Sequence[cv.DMatch]]:
	'''
	Match SIFT descriptors between consecutive images.

	### Parameters:
	- descriptors (list[np.ndarray]): List of SIFT descriptors for each image.
	- verbose (bool = True): If True, shows progress with tqdm.

	### Returns
	- matches (list[Sequence[cv.DMatch]]): Filtered matches between consecutive images.
	'''

	num_images = len(descriptors)
	matches = []
	bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

	for i in tqdm(
		range(num_images - 1),
		disable=not(verbose),
		desc="Matching SIFT points between images:",
	):
		match = bf.match(descriptors[i], descriptors[i + 1])
		match = sorted(match, key=lambda x: x.distance)

		# filter away the 20% worst matches
		num_matches = len(match)
		cutoff = int(num_matches * 0.8)
		match = match[:cutoff]

		matches.append(match)

	return matches


def extract_matches(
		matches: list[Sequence[cv.DMatch]],
		keys: list[list[cv.KeyPoint]],
		verbose: bool = True,
	) -> tuple[list[npt.NDArray], list[npt.NDArray]]:
	'''
	Extract homogeneous coordinates of matched keypoints from matches.

	### Parameters:
	- matches (list[Sequence[cv.DMatch]]): Matches between consecutive image pairs.
	- keys (list[list[cv.KeyPoint]]): List of keypoints for each image.
	verbose (bool = true): If True, shows progress with tqdm.

	### Returns: 
	- x1s (list[np.ndarray]): Matched 2D points in image i, homogeneous coordinates (3, N).
	- x2s (list[np.ndarray]): Matched 2D points in image i+1, homogeneous coordinates (3, N).
	'''

	x1s = []
	x2s = []

	for i in tqdm(
		range(len(matches)), disable=not(verbose), desc="Extracting points from matches:"):
		
		x1 = np.ones((3, len(matches[i])))
		x2 = np.ones_like(x1)

		for j, match in enumerate(matches[i]):
			x1[:-1, j] = keys[i][match.queryIdx].pt
			x2[:-1, j] = keys[i + 1][match.trainIdx].pt

		x1s.append(x1)
		x2s.append(x2)

	return x1s, x2s

