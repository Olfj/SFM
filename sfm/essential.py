import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from sfm.triangulate import triangulate

def extract_R_from_xs(
        x1s, 
        x2s, 
        K, 
        eps, 
        verbose: bool = True, 
        num_runs: int = 10000
    ) -> tuple[list[npt.NDArray], list[npt.NDArray]]:

    '''
    Estimate relative rotations between multiple camera views from 
    corresponding 2D points and camera intrinsics.

    ### Parameters:
    - x1s (list of np.ndarray): List of 2D point arrays from the first view 
      (each of shape (3, N) in homogeneous coordinates).
    - x2s (list of np.ndarray): List of 2D point arrays from the second view 
      (same format as x1s).
    - K (np.ndarray): Camera intrinsic calibration matrix (3x3).
    - eps (float): Reprojection error threshold (in pixels).
    - verbose (bool, default=True): If True, shows progress bar with `tqdm`.
    - num_runs (int, default=10000): Number of RANSAC iterations for 
      essential matrix estimation.

    ### Returns:
    - Rs (list of np.ndarray): List of estimated rotation matrices (3x3).
    - inliers (list of np.ndarray): Boolean masks of inliers for each 
      essential matrix estimation.
    '''

    temp_x1s, temp_x2s = x1s.copy(), x2s.copy()
    Es, Rs, inliers = [], [np.eye(3)], []
    
    for i in tqdm(range(len(x1s)), disable=not(verbose), desc='Estimating rotation matricies and inliers.'):
        E, inlier, _ = estimate_E_robust(x1s[i], x2s[i], K, eps, num_runs=num_runs)
        inliers.append(inlier)
        Es.append(E)
        temp_x1s[i] = temp_x1s[i][:, inlier]
        temp_x2s[i] = temp_x2s[i][:, inlier]

    Ps = [triangulate(Es[i], temp_x1s[i], temp_x2s[i])[0] for i in range(len(x1s))]

    for P in Ps:
        Rs.append(P[:, :-1] @ Rs[-1])

    return Rs, inliers


def point_line_distance_2d(
        line: npt.NDArray, 
        point: npt.NDArray
    ) -> npt.NDArray:
    '''
    Compute the perpendicular distance from a 2D point to a line.

    ### Parameters:
    - line (np.ndarray): Line coefficients (a, b, c) representing a*x + b*y + c = 0.
    - point (np.ndarray): 2D point (x, y).

    ### Returns:
    - float: Shortest distance from point to line.
    '''
    return (np.abs(line[0]*point[0] + line[1]*point[1] + line[2]) / 
            np.sqrt(line[0]*line[0] + line[1]*line[1]))


def compute_epipolar_errors(
        F: npt.NDArray, 
        x_1: npt.NDArray, 
        x_2: npt.NDArray
    ) -> npt.NDArray:
    '''
    Compute epipolar constraint errors for point correspondences.

    ### Parameters:
    - F (np.ndarray): Fundamental or essential matrix (3x3).
    - x_1 (np.ndarray): Homogeneous coordinates of points in image 1 (3xN).
    - x_2 (np.ndarray): Homogeneous coordinates of points in image 2 (3xN).

    ### Returns:
    - np.ndarray: Array of epipolar distances for each correspondence.
    '''
    lines = F @ x_1
    points = x_2[:-1]
    
    return point_line_distance_2d(lines, points)


def force_essential(E_approx: npt.NDArray) -> npt.NDArray:
    '''
    Project an approximate essential matrix onto the manifold of valid
    essential matrices using SVD.

    ### Parameters:
    - E_approx (np.ndarray): Approximate essential matrix (3x3).

    ### Returns:
    - np.ndarray: Corrected essential matrix (3x3) with singular values [1, 1, 0].
    '''
    U, _, V = np.linalg.svd(E_approx)
    S_prime = np.diag([1, 1, 0])
    
    return U @ S_prime @ V


def estimate_F_DLT(x_1: npt.NDArray, x_2: npt.NDArray) -> npt.NDArray:
    '''
    Estimate the fundamental matrix using the (normalized) 8-point algorithm.

    ### Parameters:
    - x_1 (np.ndarray): Homogeneous coordinates of points in image 1 (3xN).
    - x_2 (np.ndarray): Homogeneous coordinates of points in image 2 (3xN).

    ### Returns:
    - np.ndarray: Estimated fundamental matrix (3x3).
    '''
    n = x_1.shape[1]
    M = np.zeros(shape=(n, 9))

    for i in range(n):
        M[i, :] = np.outer(x_2[:, i], x_1[:, i]).flatten()
    
    _, _, V = np.linalg.svd(M)
    return V[-1, :].reshape((3, 3))


def estimate_E_robust(
        x_1: npt.NDArray, 
        x_2: npt.NDArray, 
        K: npt.NDArray, 
        eps: float, 
        num_runs: int = 50000
    ) -> tuple[npt.NDArray, npt.NDArray, int]:
    '''
    Robustly estimate the essential matrix using RANSAC.

    ### Parameters:
    - x_1 (np.ndarray): Homogeneous coordinates of points in image 1 (3xN).
    - x_2 (np.ndarray): Homogeneous coordinates of points in image 2 (3xN).
    - K (np.ndarray): Camera intrinsic calibration matrix (3x3).
    - eps (float): Inlier threshold in pixels.
    - num_runs (int, default=50000): Number of RANSAC iterations.

    ### Returns:
    - E (np.ndarray): Estimated essential matrix (3x3).
    - inliers (np.ndarray): Boolean mask of inlier correspondences.
    - num_inliers (int): Number of inliers found.
    '''
    inlier_threshold = eps / K[0, 0]

    E, inliers, num_inliers = None, None, 0

    for _ in range(num_runs):
        samples = np.random.randint(low=0, high=x_1.shape[1], size=8) 
        current_E = estimate_F_DLT(x_1[:, samples], x_2[:, samples])
        current_E = force_essential(current_E)

        distances_1 = compute_epipolar_errors(current_E.T, x_2, x_1)
        distances_2 = compute_epipolar_errors(current_E, x_1, x_2)

        within_threshold = ((distances_1**2 + distances_2**2)/2) < inlier_threshold**2
        num_within_threshold = np.sum(within_threshold)

        if num_within_threshold > num_inliers:
            E = current_E
            inliers = within_threshold
            num_inliers = num_within_threshold

    if E == None or inliers == None:
        raise RuntimeError(f'No essential matrix found in {num_runs} RANSAC iterations.')

    return E, inliers, num_inliers
