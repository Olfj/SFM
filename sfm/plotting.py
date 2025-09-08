import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def camera_center_and_axis(
		camera_matrix : npt.NDArray
		) -> tuple[npt.NDArray, npt.NDArray]:
	'''
	Calculates the camera center and orientation of a camera matrix
	### Parameters:
	- camera_matrix (np.ndarray): a 3 x 4 camera matrix.
	### Returns:
	- center (np.ndarray): the camera center (position).
	- axis (np.ndarray): the cameras principal axis (viewing direction).
	'''
	
	M = camera_matrix[:3,:3]
	center = -np.linalg.inv(M) @ camera_matrix[:,-1]
	axis = np.linalg.det(M) * M[-1,:]
	axis /= np.linalg.norm(axis)

	return center, axis

def plot_scene_and_cameras(
		X : list[npt.NDArray],
		Ps : list[npt.NDArray], 
		size : float = 0.1,
		title : str = ''
		) -> None:
	'''
	Plots a scene of 3d points and the cameras capturing the scene.
	### Parameters:
	- X (list[np.ndarray]): a list of 3d points, one array of points per camera pair (P_0,P_1), (P_1,P-2)...
	- Ps (list[np.ndarray]): a list of camera matrices.
	- size (float = 0.1): plotting size for camera axis and 3d points.
	- title (str): Title of plot 
	'''
	
	ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
	
	for x in X:
		ax.scatter(*x, s=size, color = 'b') # type: ignore
		ax.set_aspect('equal', adjustable='box')
		ax.set_title(title)

	ax.scatter([0], [0], [0], s=50*size, label= '(0,0,0)', color='green') # type: ignore  
	ax.legend()

	for cam in Ps:
		cam_cent, cam_axis = camera_center_and_axis(cam)
		ax.quiver(*cam_cent, *cam_axis/2, color='orange', linewidth=2)
	plt.show();plt.close()
