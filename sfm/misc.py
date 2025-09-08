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

