import numpy as np
import numpy.typing as npt
from sfm.misc import pflat
from tqdm import tqdm

def estimate_ts_robust(
		xs: list[npt.NDArray],
		Xs: list[npt.NDArray],
		Rs: list[npt.NDArray],
		K: npt.NDArray,
		verbose: bool = True,
		eps: float = 1,
		num_runs: int = 20000
	):
	'''
	Estimate translations for multiple views using robust RANSAC-based 
	estimation.

	### Parameters:
	- xs (list[np.ndarray]): List of 2D point correspondences in image space (each of shape (3, N), homogeneous coordinates).
	- Xs (list[np.ndarray]): List of corresponding 3D points in world space (each of shape (4, N), homogeneous coordinates).
	- Rs (list[np.ndarray]): List of estimated rotation matrices (3x3).
	- K (np.ndarray): Camera intrinsic calibration matrix (3x3).
	- verbose (bool, default=True): If True, displays a progress bar.
	- eps (float, default=1): Pixel reprojection error threshold.
	- num_runs (int, default=20000): Number of RANSAC iterations.

	### Returns:
	- ts (list[np.ndarray]): List of estimated translation vectors (3x1).
	'''
	
	ts = []
	for i in tqdm(range(len(xs)), disable=not(verbose), desc='Estimating translations using RANSAC'):
		t, _, _ = estimate_translation_robust(xs[i], Xs[i], Rs[i], K, eps=eps, num_runs=num_runs)
		ts.append(t)
	
	return ts


def estimate_translation(
		X: npt.NDArray, 
		x: npt.NDArray, 
		R: npt.NDArray
	) -> npt.NDArray:
	'''
	Estimate translation vector given 3D-2D correspondences and a rotation 
	matrix, using a linear least squares formulation.

	### Parameters:
	- X (np.ndarray): 3D points in world coordinates (4xN, homogeneous).
	- x (np.ndarray): Corresponding 2D points in image coordinates (3xN, homogeneous).
	- R (np.ndarray): Rotation matrix (3x3).

	### Returns:
	- t (np.ndarray): Estimated translation vector (3x1).
	'''
	
	n = x.shape[1]
	A = np.zeros((2 * n, 3))
	B = np.zeros((2 * n, 1))

	for i in range(n):
		Xi = R @ X[:, i]
		A[2 * i : 2 * i + 2] = np.array([[1, 0, -x[0, i]], [0, 1, -x[1, i]]])
		B[2 * i : 2 * i + 2] = np.array([[Xi[2] * x[0, i] - Xi[0]], [Xi[2] * x[1, i] - Xi[1]]])

	t = np.linalg.lstsq(A, B, rcond=None)[0].flatten()
	return t.reshape(3, -1)


def estimate_translation_robust(
		x: npt.NDArray, 
		X: npt.NDArray,
		R: npt.NDArray, 
		K: npt.NDArray,
		eps: float = 1, 
		num_runs: int = 10000
	) -> tuple[npt.NDArray, npt.NDArray, int]:
	'''
	Robustly estimate the translation vector using RANSAC.

	Parameters:
	- x (np.ndarray): 2D image points (3xN, homogeneous coordinates).
	- X (np.ndarray): Corresponding 3D points in world coordinates (4xN, homogeneous).
	- R (np.ndarray): Rotation matrix (3x3).
	- K (np.ndarray): Camera intrinsic calibration matrix (3x3).
	- eps (float, default=1): Pixel reprojection error threshold.
	- num_runs (int, default=10000): Number of RANSAC iterations.

	Returns:
	- t (np.ndarray): Estimated translation vector (3x1).
	- inliers (np.ndarray): Boolean mask indicating inlier correspondences.
	- num_inliers (int): Number of inliers found.
	'''

	inlier_threshold = 3 * eps / K[0, 0]

	t, inliers, num_inliers = None, None, 0

	for _ in range(num_runs):
		samples = np.random.randint(low=0, high=x.shape[1], size=2) 
		current_t = estimate_translation(X[:, samples], x[:, samples], R)
		
		proj = R @ X + current_t
		if not(any((x == 0) for x in proj[-1])):  # ensure no division by zero
			proj = pflat(proj)
			dist = np.linalg.norm(x - proj, axis=0)
			within_threshold = dist < inlier_threshold
			num_within_threshold = np.sum(within_threshold)

			if num_within_threshold > num_inliers:
				t = current_t
				inliers = within_threshold
				num_inliers = num_within_threshold
	
	if inliers == None or t == None:
		raise RuntimeError('No samples without 0 in final column.')

	return t, inliers, num_inliers
