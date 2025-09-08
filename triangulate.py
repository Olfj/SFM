import numpy as np
import numpy.typing as npt
from tqdm import tqdm

def triangulate_Xs(
        x1s: list[npt.NDArray],
        x2s: list[npt.NDArray],
        Ps: list[npt.NDArray],
        verbose: bool = True
    ) -> list[npt.NDArray]:
    '''
    Triangulate 3D points across multiple pairs of camera matrices.

    ### Parameters:
    - x1s (list[np.ndarray]): List of 2D points in the first image (3xN, homogeneous).
    - x2s (list[np.ndarray]): List of 2D points in the second image (3xN, homogeneous).
    - Ps (list[np.ndarray]): List of camera projection matrices (3x4).
    - verbose (bool, default=True): Whether to show a progress bar.

    ### Returns:
    - Xs (list[np.ndarray]): List of triangulated 3D points (3xN).
    '''
    
    Xs = []
    for i in tqdm(range(len(Ps)-1), disable=not(verbose), desc='Triangulating matched points using known camera matrices.'):
        Xi = triangulate_known_cam(Ps[i], Ps[i+1], x1s[i], x2s[i])
        Xs.append(Xi)
    mask = near_centroid(Xs)
    for i in range(len(Xs)):
        Xs[i] = Xs[i][:, mask[i]]

    return Xs


def triangulate(
        E: npt.NDArray,
        x_1: npt.NDArray,
        x_2: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray]:
    '''
    Triangulate a set of 2D correspondences into 3D points, given an essential matrix.

    ### Parameters:
    - E (np.ndarray): Essential matrix (3x3).
    - x_1 (np.ndarray): 2D points in first image (3xN, homogeneous).
    - x_2 (np.ndarray): 2D points in second image (3xN, homogeneous).

    ### Returns:
    - P (np.ndarray): Selected camera projection matrix (3x4).
    - X (np.ndarray): Triangulated 3D points (4xN, homogeneous).
    '''
    
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    p_0 = np.eye(3,4)
    Ps = extract_P_from_E(E, W)
    triangulated = np.zeros((4, x_1.shape[1], 4))
    
    for i in range(4):
        for j in range(x_1.shape[1]):
            triangulated[i, j] = triangulate_3D_point_DLT(p_0, Ps[i], x_1[:, j], x_2[:, j])

    camera, _, in_front = check_in_front(Ps, p_0, triangulated)

    return Ps[camera], triangulated[camera].T


def triangulate_known_cam(
        P1: npt.NDArray,
        P2: npt.NDArray,
        x_1: npt.NDArray,
        x_2: npt.NDArray,
    ) -> npt.NDArray:
    '''
    Triangulate 3D points from two sets of 2D correspondences using known camera matrices.

    ### Parameters:
    - P1 (np.ndarray): First camera projection matrix (3x4).
    - P2 (np.ndarray): Second camera projection matrix (3x4).
    - x_1 (np.ndarray): 2D points in the first image (3xN, homogeneous).
    - x_2 (np.ndarray): 2D points in the second image (3xN, homogeneous).

    ### Returns:
    - X (np.ndarray): Triangulated 3D points (3xN).
    '''
    
    triangulated = np.zeros((4, x_1.shape[1]))
    
    for i in range(x_1.shape[1]):
        triangulated[:, i] = triangulate_3D_point_DLT(P1, P2, x_1[:, i], x_2[:, i])

    return triangulated[:-1, :]


def check_in_front(
        Ps: npt.NDArray,
        p_0: npt.NDArray,
        triangulated: npt.NDArray
    ) -> tuple[int, int, npt.NDArray]:
    '''
    Check which camera configuration results in the most triangulated points in front of both cameras.

    ### Parameters:
    - Ps (np.ndarray): Candidate camera matrices (4x3x4).
    - p_0 (np.ndarray): Canonical camera matrix (3x4).
    - triangulated (np.ndarray): Triangulated points for each candidate configuration (4xN, for each camera).

    ### Returns:
    - camera (int): Index of the best camera configuration.
    - num_in_front (int): Number of points in front of the cameras.
    - in_front (np.ndarray): Boolean mask of points that are in front.
    '''
    
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
        P1: npt.NDArray, 
        P2: npt.NDArray, 
        x1: npt.NDArray, 
        x2: npt.NDArray
    ) -> npt.NDArray:
    '''
    Triangulate a single 3D point using the Direct Linear Transform (DLT) algorithm.

    ### Parameters:
    - P1 (np.ndarray): First camera projection matrix (3x4).
    - P2 (np.ndarray): Second camera projection matrix (3x4).
    - x1 (np.ndarray): Single 2D point in the first image (3, homogeneous).
    - x2 (np.ndarray): Single 2D point in the second image (3, homogeneous).

    ### Returns:
    - X (np.ndarray): Homogeneous 3D point (4,).
    '''
    M = [x1[1] * P1[2,:] - P1[1,:],
         P1[0,:] - x1[0] * P1[2,:],
         x2[1] * P2[2,:] - P2[1,:],
         P2[0,:] - x2[0] * P2[2,:]]
    M = np.array(M).reshape((4,4))

    M = M.T @ M
    _, _, V = np.linalg.svd(M, full_matrices=False)
    res = V[-1]
    return res / res[-1]


def extract_P_from_E(
        E: npt.NDArray, 
        W: npt.NDArray
    ) -> npt.NDArray:
    '''
    Extract possible camera projection matrices from the essential matrix.

    ### Parameters:
    - E (np.ndarray): Essential matrix (3x3).
    - W (np.ndarray): Skew-symmetric rotation helper matrix.

    ### Returns:
    - Ps (np.ndarray): Array of 4 possible camera projection matrices (4x3x4).
    '''
    
    U, _, VT = np.linalg.svd(E)
    VT = -VT if np.linalg.det(U @ VT.T) < 0 else VT
    u_3 = U[:, -1]

    P_1 = np.zeros(shape=(3,4))
    P_1[:, :-1] = U @ W @ VT
    P_1[:, -1] = u_3

    P_2 = P_1.copy()
    P_2[:, -1] = -u_3

    P_3 = P_1.copy()
    P_3[:, :-1] = U @ W.T @ VT

    P_4 = P_2.copy()
    P_4[:, :-1] = U @ W.T @ VT

    return np.array([P_1, P_2, P_3, P_4])


def near_centroid(point_sets):
    '''
    Filter 3D points in multiple sets based on their distance to the global centroid of all points.

    ### Parameters:
    - point_sets (list[np.ndarray]): A list of 3xN arrays of 3D points.

    ### Returns:
    - masks (list[np.ndarray]): A list of boolean masks, one per point set,
      where True indicates the point is closer to the global centroid than 
      a fixed threshold.
    '''
    
    # Validate input
    for points in point_sets:
        if points.shape[0] != 3:
            raise ValueError("Each input array must have a shape of 3 x N.")

    # Combine all points to calculate the global center of mass
    if len(point_sets) > 1:
        all_points = np.hstack(point_sets)
    else:
        all_points = point_sets[0]
    
    global_center_of_mass = np.mean(all_points, axis=1, keepdims=True)

    # Compute the Euclidean distances from the global center of mass
    global_distances = np.linalg.norm(all_points - global_center_of_mass, axis=0)

    # Calculate the 90% quantile of the global distances
    global_quantile_90 = np.quantile(global_distances, 0.9)

    # Compute the threshold (2 times the 90% quantile)
    threshold = 2 * global_quantile_90

    # Generate masks for each point set
    masks = []
    for points in point_sets:
        distances = np.linalg.norm(points - global_center_of_mass, axis=0)
        mask = distances < threshold
        masks.append(mask)

    return masks
