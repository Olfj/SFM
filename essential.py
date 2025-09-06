import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from triangulate import triangulate

def extract_R_from_xs(x1s, x2s, K, eps, verbose : bool = True, num_runs : int = 10000):
	
	part_1_x1s, part_1_x2s = x1s.copy(), x2s.copy()
	Es, Rs, inliers = [], [np.eye(3)], []
	
	for i in tqdm(range(len(x1s)), disable=not(verbose), desc='Estimating rotation matricies and inliers.'):
		E, inlier, _ = estimate_E_robust(x1s[i], x2s[i], K, eps, num_runs=num_runs)
		inliers.append(inlier)
		Es.append(E)
		part_1_x1s[i] = x1s[i][:,inlier]
		part_1_x2s[i] = x2s[i][:,inlier]
	

	Ps = [triangulate(Es[i], part_1_x1s[i], part_1_x2s[i])[0] for i in range(len(x1s))]

	for P in Ps:
		Rs.append(P[:,:-1] @ Rs[-1])

	return Rs, inliers

def point_line_distance_2d(
		line : npt.NDArray, 
		point : npt.NDArray
		) -> npt.NDArray:
	'''
	Returns the shortest distanse from a point to a line in two dimensions.

	# Parameters:

	line: Line in the form line[0]*x + line[1]*y + line[2] = 0
	'''
	
	return (np.abs(line[0]*point[0] + line[1]*point[1] + line[2]) / 
				 np.sqrt(line[0]*line[0] + line[1]*line[1]))

def compute_epipolar_errors(
		F : npt.NDArray, 
		x_1 : npt.NDArray, 
		x_2 : npt.NDArray
		) -> npt.NDArray:
	
	lines = F @ x_1
	points = x_2[:-1]
	
	return point_line_distance_2d(lines, points)

def force_essential(
		E_approx : npt.NDArray
		) -> npt.NDArray:
	
	U, _, V = np.linalg.svd(E_approx)
	S_prime = np.diag([1, 1, 0])
	
	return U @ S_prime @ V

def estimate_F_DLT(
		x_1 : npt.NDArray, 
		x_2 : npt.NDArray):
	# M in a (n x 9) matrix
	# https://en.wikipedia.org/wiki/Eight-point_algorithm

	n = x_1.shape[1]
	M = np.zeros(shape=(n,9))

	for i in range(n):
		M[i,:] = np.outer(x_2[:,i], x_1[:,i]).flatten()
	
	U, S, V = np.linalg.svd(M)
	return V[-1,: ].reshape((3,3))

def estimate_E_robust(
		x_1 : npt.NDArray, 
		x_2 : npt.NDArray, 
		K : npt.NDArray, 
		eps : float, 
		num_runs : int = 50000
		) -> tuple[npt.NDArray, npt.NDArray, int]:
	
	inlier_threshold = eps/K[0,0]

	E, inliers, num_inliers = None, None, 0

	if num_runs <= 0:
		raise ValueError("num_runs must be greater than zero.")	

	for _ in range(num_runs):
		
		samples = np.random.randint(low=0, high=x_1.shape[1] , size=8) 
		curr_E = estimate_F_DLT(x_1[:,samples], x_2[:,samples])
		curr_E = force_essential(curr_E)

		distances_1 = compute_epipolar_errors(curr_E.T, x_2, x_1)
		distances_2 = compute_epipolar_errors(curr_E, x_1, x_2)

		within_threshold = ((distances_1**2 + distances_2**2)/2) < inlier_threshold**2
		num_within_threshold = np.sum(within_threshold)

		if num_within_threshold > num_inliers:
			E = curr_E
			inliers = within_threshold
			num_inliers = num_within_threshold

	return E, inliers, num_inliers