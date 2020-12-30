import numpy as np
from numpy.linalg import inv, norm

def triangulate(Kl, Kr, Twl, Twr, pl, pr, Sl, Sr):
    """
    Triangulate 3D point position from camera projections.

    The function computes the 3D position of a point landmark from the 
    projection of the point into two camera images separated by a known
    baseline. All arrays should contain float64 values.

    Parameters:
    -----------
    Kl   - 3x3 np.array, left camera intrinsic calibration matrix.
    Kr   - 3x3 np.array, right camera intrinsic calibration matrix.
    Twl  - 4x4 np.array, homogeneous pose, left camera in world frame.
    Twr  - 4x4 np.array, homogeneous pose, right camera in world frame.
    pl   - 2x1 np.array, point in left camera image.
    pr   - 2x1 np.array, point in right camera image.
    Sl   - 2x2 np.array, left image point covariance matrix.
    Sr   - 2x2 np.array, right image point covariance matrix.

    Returns:
    --------
    Pl  - 3x1 np.array, closest point on ray from left camera  (in world frame).
    Pr  - 3x1 np.array, closest point on ray from right camera (in world frame).
    P   - 3x1 np.array, estimated 3D landmark position in the world frame.
    S   - 3x3 np.array, covariance matrix for estimated 3D point.
    """
    #--- FILL ME IN ---

    # Compute baseline (right camera translation minus left camera translation).

    # Unit vectors projecting from optical center to image plane points.
    # Use variables rayl and rayr for the rays.
 
    # Projected segment lengths.
    # Use variables ml and mr for the segment lengths.
    
    # Segment endpoints.
    # Use variables Pl and Pr for the segment endpoints.

    # Then fill in with appropriate ray Jacobians. These are 
    # 3x4 matrices, but two columns are zeros (because the right
    # ray direction is not affected by the left image point and 
    # vice versa).
    drayl = np.zeros((3, 4))  # Jacobian left ray w.r.t. image points.
    drayr = np.zeros((3, 4))  # Jacobian right ray w.r.t. image points.

    # Add code here...

    # Extract camera translations
    cl = Twl[0:3, 3]
    cr = Twr[0:3, 3]

    # Compute baseline
    b = cr - cl 

    # Augment pl and pr
    pl_aug = np.vstack((pl, np.array([[1]])))
    pr_aug = np.vstack((pr, np.array([[1]])))

    # Compute unit-length rays from left and right cameras
    rayl_unnorm = Twl[0:3, 0:3] @ (inv(Kl) @ pl_aug)
    rayr_unnorm = Twr[0:3, 0:3] @ (inv(Kr) @ pr_aug)
    norm_rl = norm(rayl_unnorm)
    norm_rr = norm(rayr_unnorm)
    rayl = rayl_unnorm / norm_rl  # normalized left ray
    rayr = rayr_unnorm / norm_rr  # normalized right ray

    # Compute Jacobians of ray directions with respect to image place coordinates
    drl = (Twl[0:3, 0:3] @ inv(Kl))[:, 0:2]  # 3x2 Jacobian for unnormalized left ray
    drr = (Twr[0:3, 0:3] @ inv(Kr))[:, 0:2]  # 3x2 Jacobian for unnormalized right ray
    dnorm_rl = rayl.T @ drl  # 1x2 Jacobian for norm of left ray
    dnorm_rr = rayr.T @ drr  # 1x2 Jacobian for norm of right ray
    drayl[:, 0:2] = (drl * norm_rl - rayl_unnorm @ dnorm_rl) / norm_rl**2  # quotient rule for left ray
    drayr[:, 2:4] = (drr * norm_rr - rayr_unnorm @ dnorm_rr) / norm_rr**2  # quotient rule for right ray

    #------------------

    # Compute dml and dmr (partials wrt segment lengths).
    u = np.dot(b.T, rayl) - np.dot(b.T, rayr)*np.dot(rayl.T, rayr)
    v = 1 - np.dot(rayl.T, rayr)**2

    du = (b.T@drayl).reshape(1, 4) - \
         (b.T@drayr).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayr)*((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
 
    dv = -2*np.dot(rayl.T, rayr)*((rayr.T@drayl).reshape(1, 4) + \
        (rayl.T@drayr).reshape(1, 4))

    m = np.dot(b.T, rayr) - np.dot(b.T, rayl)@np.dot(rayl.T, rayr)
    n = np.dot(rayl.T, rayr)**2 - 1

    dm = (b.T@drayr).reshape(1, 4) - \
         (b.T@drayl).reshape(1, 4)*np.dot(rayl.T, rayr) - \
         np.dot(b.T, rayl)@((rayr.T@drayl) + (rayl.T@drayr)).reshape(1, 4)
    dn = -dv

    dml = (du*v - u*dv)/v**2
    dmr = (dm*n - m*dn)/n**2

    #--- FILL ME IN ---

    # 3D point.
    ml = np.squeeze(u/v, axis=1)  # scalar length for left ray
    mr = np.squeeze(m/n, axis=1)  # scalar length for right ray
    Pl = cl.reshape(3,1) + rayl*ml  # coordinate of left ray endpoint
    Pr = cr.reshape(3,1) + rayr*mr  # coordinate of right ray endpoint
    P = (Pl + Pr) / 2  # 3D position of triangulated landmark point

    # Compute Jacobian for P w.r.t. image points using dml and dmr.
    JP = (ml*drayl + rayl*dml + mr*drayr + rayr*dmr)/2

    # 3x3 landmark point covariance matrix (need to form
    # the 4x4 image plane covariance matrix first).
    img_plane_cov = np.vstack(( np.hstack((Sl, np.zeros((2,2)))),
                                np.hstack((np.zeros((2,2)), Sr)) ))
    S = JP @ img_plane_cov @ JP.T

    #------------------

    # Check for correct outputs...
    correct = isinstance(Pl, np.ndarray) and Pl.shape == (3, 1) and \
              isinstance(Pr, np.ndarray) and Pr.shape == (3, 1) and \
              isinstance(P,  np.ndarray) and P.shape  == (3, 1) and \
              isinstance(S,  np.ndarray) and S.shape  == (3, 3)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Pl, Pr, P, S