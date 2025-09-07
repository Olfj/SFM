import numpy as np
import numpy.typing as npt
from misc import pflat
from tqdm import tqdm

def estimate_ts_robust(
		xs : list[npt.NDArray],
		Xs : list[npt.NDArray],
		Rs : list[npt.NDArray],
		K : npt.NDArray,
		verbose : bool = False,
		eps : float = 1,
		num_runs : int = 20000
		):
			
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
		x : npt.NDArray, 
		X : npt.NDArray,
		R : npt.NDArray, 
		K : npt.NDArray,
		eps : float = 1, 
		num_runs : int = 10000
		) -> tuple[npt.NDArray, npt.NDArray, int]:
	
	inlier_threshold = 3*eps/K[0,0]

	t, inliers, num_inliers = None, None, 0

	for _ in range(num_runs):
		
		samples = np.random.randint(low=0, high=x.shape[1] , size=2) 
		curr_t = estimate_translation(X[:,samples], x[:,samples], R)
		
		proj = R @ X + curr_t
		if not(any((x == 0) for x in proj[-1])):
			proj = pflat(proj)
			dist = np.linalg.norm(x - proj, axis=0)
			within_threshold = dist < inlier_threshold
			num_within_threshold = np.sum(within_threshold)

			if num_within_threshold > num_inliers:
				t = curr_t
				inliers = within_threshold
				num_inliers = num_within_threshold

	return t, inliers, num_inliers