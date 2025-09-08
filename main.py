from sfm.sift import sift_matches, x_X_from_descript, X_and_descript_from_inital_pair
from sfm.plotting import plot_scene_and_cameras
from sfm.essential import extract_R_from_xs
from sfm.triangulate import triangulate_Xs
from sfm.misc import pflat, Ps_from_R_t
from sfm.translation import estimate_ts_robust
from image_data import ImageData

class PipeLine:

	def __init__(self) -> None:
		pass

	def struct_from_motion(
			self, 
			dataset : int, 
			verbose : bool = True,
			num_ransac_E : int = 10000,
			num_ransac_t : int = 20000,
			) -> None:
		
		# Load image data
		images = ImageData(dataset)

		# Sift matches, x1s[i] are the sift points found to match the points in x2s[i]. For n images x1s contain the sift points for images 0,1,...,n-1 and x2s contain the sift points for images 1,2,...,n 
		keys, descriptors, _, x1s, x2s = sift_matches(images.grayscales, verbose=verbose)

		# Normalize points by K_inv
		x1s = [pflat(images.K_inv @ x1) for x1 in x1s]
		x2s = [pflat(images.K_inv @ x2) for x2 in x2s]

		# Estimate rotation matrices between images and the inliers (ponts that are close enough to the epipolar lines) of the sift matches.
		Rs, inliers = extract_R_from_xs(x1s, x2s, images.K, images.pixel_treshold, num_runs=num_ransac_E)
		x1s = [x1s[i][:,inliers[i]] for i in range(len(x1s))]
		x2s = [x2s[i][:,inliers[i]] for i in range(len(x2s))]

		# Initial pair points X, and the correspoding d2 points in camera 1 x.
		X, X_descript, _ = X_and_descript_from_inital_pair(Rs, images, images.pixel_treshold)


		# Matches with initial pair points and 2d points from camera 1 for all cameras
		xs, Xs = x_X_from_descript(X, X_descript, keys, descriptors)

		xs = [pflat(images.K_inv @ x) for x in xs]

		# Estimate t from rotations and points
		ts = estimate_ts_robust(xs, Xs, Rs, images.K, verbose=verbose, eps=images.pixel_treshold, num_runs=num_ransac_t)

		# Put rotations and translations together for camera matricies
		Ps = Ps_from_R_t(Rs, ts)

		Xs = triangulate_Xs(x1s, x2s, Ps, verbose=verbose)
		plot_scene_and_cameras(Xs, Ps)

if __name__=='__main__':
	
	pipeline = PipeLine()
	pipeline.struct_from_motion(dataset=5, verbose=True, num_ransac_E=10000, num_ransac_t=10000)







		# pixel_cords = [x[:-1,:].astype(int).T for x in xs]
		# rgb_ims = [np.array(im) for im in images.images]
		
		# colors = []
		# for i in range(len(pixel_cords)):
		# 	color = rgb_ims[i][pixel_cords[i][:,1], pixel_cords[i][:,0], :]
		# 	print(pixel_cords[i].shape)
		# 	print(color.shape)
		# 	colors.append(color)

		# plt.imshow(images.grayscales[4], cmap='gray')
		# plt.scatter(*xs[4], color='b')
		# plt.show()