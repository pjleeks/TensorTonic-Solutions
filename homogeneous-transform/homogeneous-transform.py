import numpy as np

def apply_homogeneous_transform(T, points):
    # Ensure BOTH are numpy arrays
    T = np.array(T)           # <--- This fixes the 'list' object error
    points = np.array(points)
    
    if points.ndim == 1:
        points = points.reshape(1, -1)
    
    n = points.shape[0]
    ones = np.ones((n, 1))
    points_h = np.hstack([points, ones])
    
    # Now .T will work because T is an ndarray
    transformed_h = points_h @ T.T
    
    result = transformed_h[:, :3]
    return result.squeeze() if n == 1 else result