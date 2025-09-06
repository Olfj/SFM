import numpy as np
import numpy.typing as npt
from misc import close_enough
from tqdm import tqdm

def triangulate_Xs(
		x1s : list[npt.NDArray],
		x2s : list[npt.NDArray],
		Ps : list[npt.NDArray],
		verbose : bool = True
		) -> list[npt.NDArray]:
	
	Xs = []
	for i in tqdm(range(len(Ps)-1), disable=not(verbose), desc='Triangulating matched points using known camera matricies.'):
		Xi = triangulate_known_cam(Ps[i], Ps[i+1], x1s[i], x2s[i])
		Xs.append(Xi)
	mask = close_enough(Xs)
	for i in range(len(Xs)):
		Xs[i] = Xs[i][:,mask[i]]

	return Xs

def triangulate(
		E : npt.NDArray,
		x_1 : npt.NDArray,
		x_2 : npt.NDArray,
		) -> tuple[npt.NDArray, npt.NDArray]:
	'''
	Robustly triangulate 2 sets of 2d points to a set of 3d points and its camera
	'''
	
	W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
	p_0 = np.eye(3,4)
	Ps = extract_P_from_E(E,W)
	triangulated = np.zeros((4,x_1.shape[1], 4))
	
	for i in range(4):
		for j in range(x_1.shape[1]):
			triangulated[i,j] = triangulate_3D_point_DLT(p_0, Ps[i], x_1[:,j], x_2[:,j])

	camera, _, in_front = check_in_front(Ps, p_0, triangulated)

	return Ps[camera], triangulated[camera].T

def triangulate_known_cam(
		P1 : npt.NDArray,
		P2 : npt.NDArray,
		x_1 : npt.NDArray,
		x_2 : npt.NDArray,
		) -> npt.NDArray:
	
	triangulated = np.zeros((4,x_1.shape[1]))
	
	for i in range(x_1.shape[1]):
			triangulated[:,i] = triangulate_3D_point_DLT(P1, P2, x_1[:,i], x_2[:,i])

	return triangulated[:-1,:]

def check_in_front(
		Ps : npt.NDArray,
		p_0 : npt.NDArray,
		triangulated : npt.NDArray
		) -> tuple[int, int, npt.NDArray]:
	
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