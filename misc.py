import numpy as np
import numpy.typing as npt

def Ps_from_R_t(Rs, ts):
	
	Ps = []
	
	for i in range(len(Rs)):
		P = np.zeros(shape=(3,4))
		P[:,:-1] = Rs[i]
		P[:,-1] = np.squeeze(ts[i])
		Ps.append(P)
	
	return Ps

def pflat(
		arr : npt.NDArray
		) -> npt.NDArray:

	if any((x == 0) for x in arr[-1]):
		raise ValueError("Bottom row can not contain any zero value.")
		
	return arr / arr[-1]

import numpy as np

def close_enough(point_sets):
	'''
	Filters 3D points in multiple sets based on their distance to the global center of mass.

	Args:
			point_sets (list of np.ndarray): A list of 3 x n numpy arrays of 3D points.

	Returns:
			list of np.ndarray: A list of boolean masks, where each mask corresponds to a point set and
													indicates whether the point is closer to the global center of mass than
													5 times the 90% quantile of global distances.
	'''
	# Validate input
	for points in point_sets:
		if points.shape[0] != 3:
			raise ValueError("Each input array must have a shape of 3 x n.")

	# Combine all points to calculate the global center of mass
	if len(point_sets) > 1:
		all_points = np.hstack(point_sets)
	else:
		all_points = point_sets[0]
	
	global_center_of_mass = np.mean(all_points, axis=1, keepdims=True)

	# Compute the Euclidean distances from the global center of mass for all points
	global_distances = np.linalg.norm(all_points - global_center_of_mass, axis=0)

	# Calculate the 90% quantile of the global distances
	global_quantile_90 = np.quantile(global_distances, 0.9)

	# Compute the threshold (5 times the 90% quantile)
	threshold = 2 * global_quantile_90

	# Generate masks for each point set
	masks = []
	for points in point_sets:
		distances = np.linalg.norm(points - global_center_of_mass, axis=0)
		mask = distances < threshold
		masks.append(mask)

	return masks
