import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from sift import SIFT_matches, disp_rand_SIFT_matches
from exif import K_and_inv_from_exif
from image import load_images, images_to_arr

def pflat(
		arr : npt.NDArray
		) -> npt.NDArray:

	if any((x == 0) for x in arr[-1]):
		raise ValueError("Bottom row can not contain any zero value.")
		
	return arr / arr[-1]

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

def force_essential(
		E_approx : npt.NDArray
		) -> npt.NDArray:
	
	U, _, V = np.linalg.svd(E_approx)
	S_prime = np.diag([1, 1, 0])
	
	return U @ S_prime @ V

def point_line_distance_2d(
		line : npt.NDArray, 
		point : npt.NDArray
		) -> npt.NDArray:
	
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

def estimate_E_robust(
		x_1 : npt.NDArray, 
		x_2 : npt.NDArray, 
		K : npt.NDArray, 
		eps : float
		) -> tuple[npt.NDArray, npt.NDArray, int]:
	
	inlier_threshold = eps/K[0,0]
	num_runs = 10000

	E, inliers, num_inliers = None, None, 0

	for i in range(num_runs):
		
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

def convert_E_to_F(
		E : npt.NDArray, 
		K_1 : npt.NDArray, 
		K_2 : npt.NDArray
		) -> npt.NDArray:
	
	return np.linalg.inv(K_1).T @ E @ np.linalg.inv(K_2)

def extract_P_from_E(
		E : npt.NDArray, 
		W : npt.NDArray
		) -> npt.NDArray:

	U, S, VT = np.linalg.svd(E)
	VT = -VT if np.linalg.det(U@VT.T) < 0 else VT
	u_3 = U[:,-1]

	P_1 = np.zeros(shape=(3,4))
	P_1[:,:-1] = U @ W @ VT
	P_1[:,-1] = u_3

	P_2 = P_1.copy()
	P_2[:,-1] = -u_3

	P_3 = P_1.copy()
	P_3[:,:-1] = U @ W.T @ VT

	P_4 = P_2.copy()
	P_4[:,:-1] = U @ W.T @ VT

	return np.array([P_1, P_2, P_3, P_4])

def triangulate_3D_point_DLT(
		P1 : npt.NDArray, 
		P2 : npt.NDArray, 
		x1 : npt.NDArray, 
		x2 : npt.NDArray
		) -> npt.NDArray:
	
	'''
	Sets up and solves the DLT equations for triangulation. 
	'''
	
	M = [x1[1] * P1[2,:] - P1[1,:],
	P1[0,:] - x1[0] * P1[2,:],
	x2[1] * P2[2,:] - P2[1,:],
	P2[0,:] - x2[0] * P2[2,:]
	]
	M = np.array(M).reshape((4,4))

	M = M.T @ M
	U, s, V = np.linalg.svd(M, full_matrices=False)
	res = V[-1]
	return res / res[-1]

def check_in_front(
		Ps : npt.NDArray,
		p_0 : npt.NDArray,
		triangulated : npt.NDArray
		) -> tuple[int, int]:
	
	camera, num_in_front, in_front = -1, 0, np.zeros(np.max(triangulated.shape))

	for i in range(4):
		projection_1 = Ps[i] @ triangulated[i].T
		projection_2 = p_0 @ triangulated[i].T

		fst_greater_than_0 = projection_1[-1] > 0
		snd_greater_than_0 = projection_2[-1] > 0

		front_sum = np.sum(fst_greater_than_0) + np.sum(snd_greater_than_0)
		
		if front_sum > num_in_front:
			in_front = np.logical_and(fst_greater_than_0, snd_greater_than_0)
			camera, num_in_front = i, front_sum 
	
	return camera, num_in_front, in_front

def camera_center_and_axis(
		matrix : npt.ArrayLike
		) -> tuple[npt.NDArray, npt.NDArray]:
	
	M = matrix[:3,:3]
	center = -np.linalg.inv(M)@matrix[:,-1]
	axis = np.linalg.det(M)*M[-1,:]
	axis /= np.linalg.norm(axis)

	return center, axis

def plot_cams(
		X : npt.ArrayLike,
		cams : npt.ArrayLike, 
		size : float = 0.1,
		title : str = '', 
		elev : float = 0, 
		azim : float = 0,
		roll : float = 0
		) -> None:
	
	ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
	ax.scatter(*X, s=size, label='X')
	ax.set_aspect('equal', adjustable='box')
	ax.view_init(elev=elev, azim=azim, roll=roll)
	ax.set_title(title)

	for cam in cams:
		cam_cent, cam_ax = camera_center_and_axis(cam)
		ax.quiver(*cam_cent, *cam_ax/2, color='orange')
	
	ax.scatter([0], [0], [0], s=5*size, label= '(0,0,0)', color='green')
	ax.legend()

	plt.show();plt.close()
	
def triangulate(
		E : npt.NDArray,
		x_1 : npt.NDArray,
		x_2 : npt.NDArray,
		) -> None:
	
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	p_0 = np.eye(3,4)
	Ps = extract_P_from_E(E,W)
	triangulated = np.zeros((4,x_1.shape[1], 4))
	
	for i in range(4):
		for j in range(x_1.shape[1]):
			triangulated[i,j] = triangulate_3D_point_DLT(p_0, Ps[i], x_1[:,j], x_2[:,j])

	camera, num_in_front, in_front = check_in_front(Ps, p_0, triangulated)

	plot_cams(triangulated[camera, in_front, :-1].T, [p_0, Ps[camera]], size=1)

def computeReprojectionError(
		P_1 : npt.NDArray, 
		P_2 : npt.NDArray, 
		X_j : npt.NDArray, 
		x_1j : npt.NDArray, 
		x_2j : npt.NDArray
		) -> tuple[float, npt.NDArray]:
	
	proj_p1 = P_1 @ X_j
	proj_p1 /= proj_p1[-1]
	proj_p2 = P_2 @ X_j
	proj_p1 /= proj_p1[-1]

	r_1 = x_1j[:-1] - proj_p1[:-1]
	r_2 = x_2j[:-1] - proj_p2[:-1]

	residual = np.concatenate((r_1, r_2))
	error = np.linalg.norm(residual)**2

	return error, residual

def get_jacobian(
		P : npt.NDArray, 
		X : npt.NDArray
		) -> npt.NDArray:

		return np.array([
				(P[-1] * (P[0] @ X)) / (P[-1] @ X) ** 2 - P[0] / (P[-1] @ X),
				((P[-1] * (P[1] @ X))) / (P[-1] @ X) ** 2 - P[1] / (P[-1] @ X)
		]) 

def linearizeReprojErr(
		P_1 : npt.NDArray,
		P_2 : npt.NDArray,
		X_j : npt.NDArray,
		x_1j : npt.NDArray,
		x_2j : npt.NDArray
		) -> tuple[npt.NDArray, npt.NDArray]:

	error, residual =  computeReprojectionError(P_1, P_2, X_j, x_1j, x_2j)
	J_1 = get_jacobian(P_1, X_j)
	J_2 = get_jacobian(P_2, X_j)

	return residual, np.vstack((J_1,J_2))

def compute_update(
		r : npt.NDArray,
		J : npt.NDArray,
		mu : float
		) -> npt.NDArray:

	return - np.linalg.lstsq(J.T @ J + mu*np.eye(4), np.eye(4), rcond=None)[0] @ (J.T @ r)

class PipeLine:

	def __init__(self, path):

		self.images = load_images(path)
		self.gray_images = images_to_arr(self.images)

	def struct_from_motion(self) -> None:
		K_1, K_1_inv = K_and_inv_from_exif(self.images[0])
		K_2, K_2_inv = K_and_inv_from_exif(self.images[1])

		x_1, x_2 = SIFT_matches(*self.gray_images)
		norm_x1 = pflat(K_1_inv @ x_1)
		norm_x2 = pflat(K_2_inv @ x_2)
		E, inliers, num_inliers = estimate_E_robust(norm_x1, norm_x2, K_1, 2)

		inlier_x1 = norm_x1[:,inliers]
		inlier_x2 = norm_x2[:,inliers]

		triangulate(E, inlier_x1, inlier_x2)

if __name__=='__main__':
	
	pipeline = PipeLine('Assignments/Project/data/1/')
	pipeline.struct_from_motion()


 



