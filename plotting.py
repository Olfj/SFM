import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def camera_center_and_axis(
		matrix : npt.NDArray
		) -> tuple[npt.NDArray, npt.NDArray]:
	
	M = matrix[:3,:3]
	center = -np.linalg.inv(M)@matrix[:,-1]
	axis = np.linalg.det(M)*M[-1,:]
	axis /= np.linalg.norm(axis)

	return center, axis

def plot_cams_and_more_points(
		X : npt.ArrayLike,
		cams : npt.ArrayLike, 
		size : float = 0.1,
		title : str = ''
		) -> None:
	
	ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
	
	for x in X:
		ax.scatter(*x, s=size)
		ax.set_aspect('equal', adjustable='box')
		# ax.view_init(elev=elev, azim=azim, roll=roll)
		ax.set_title(title)

	ax.scatter([0], [0], [0], s=50*size, label= '(0,0,0)', color='green')
	ax.legend()

	for i, cam in enumerate(cams):
		cam_cent, cam_ax = camera_center_and_axis(cam)
		ax.quiver(*cam_cent, *cam_ax/2, color='orange', linewidth=2)
	plt.show();plt.close()

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