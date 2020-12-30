import numpy as np
from numpy.linalg import inv, norm, det
#from rpy_from_dcm import rpy_from_dcm
#from dcm_from_rpy import dcm_from_rpy
from support.estimate_motion_ls import estimate_motion_ls

##
def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])


def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy
##


def estimate_motion_ils(Pi, Pf, Si, Sf, iters):
    """
    Estimate motion from 3D correspondences.
  
    The function estimates the 6-DOF motion of a body, given a series 
    of 3D point correspondences. This method relies on NLS.
    
    Arrays Pi and Pf store corresponding landmark points before and after 
    a change in pose.  Covariance matrices for the points are stored in Si 
    and Sf. All arrays should contain float64 values.

    Parameters:
    -----------
    Pi  - 3xn np.array of points (intial - before motion).
    Pf  - 3xn np.array of points (final - after motion).
    Si  - 3x3xn np.array of landmark covariance matrices.
    Sf  - 3x3xn np.array of landmark covariance matrices.

    Outputs:
    --------
    Tfi  - 4x4 np.array, homogeneous transform matrix, frame 'i' to frame 'f'.
    """
    # Initial guess...
    Tfi = estimate_motion_ls(Pi, Pf, Si, Sf)
    C = Tfi[:3, :3]
    I = np.eye(3)
    rpy = rpy_from_dcm(C).reshape(3, 1)

    theta = np.zeros((6))

    # Iterate.
    for j in np.arange(iters):
        A = np.zeros((6, 6))
        B = np.zeros((6, 1))

        # 3. Save previous best rpy estimate.
        theta_prev = theta

        #--- FILL ME IN ---

        for i in np.arange(Pi.shape[1]):
            # Extract current before and after points
            Pi_cur = (Pi[:,i]).reshape(3,1)
            Pf_cur = (Pf[:,i]).reshape(3,1)

            # Compute inverse covariance matrix (sigma_i) for one observation 
            # (pair of corresponding landmarks).
            # This is equal to the sum of the position uncertainty in frame f
            # and the position uncertainty in frame i transformed to frame f.
            inv_sigma_i = inv(Sf[:,:,i] + C @ Si[:,:,i] @ C.T)
            
            # Compute Jacobian matrix 
            dRdr, dRdp, dRdy = dcm_jacob_rpy(C)
            J = np.hstack((dRdr @ Pi_cur, dRdp @ Pi_cur, dRdy @ Pi_cur))  # 3x3

            # Compute H and Q (from Matthies’ PhD thesis, Appendix B, pg.151)
            H = np.hstack((J, I))
            Q = Pf_cur - C @ Pi_cur + J @ rpy

            # Compute A contribution from this point correspondance
            A += H.T @ inv_sigma_i @ H

            # Compute B contribution from this point correspondance
            B += H.T @ inv_sigma_i @ Q

        #------------------

        # Solve system and check stopping criteria if desired...
        if det(A) == 0:
            print('A not invertible')
            tfi_useable = False
            break
        theta = inv(A)@B
        rpy = theta[0:3].reshape(3, 1)
        C = dcm_from_rpy(rpy)
        t = theta[3:6].reshape(3, 1)

        # Check - converged?
        diff = norm(theta - theta_prev)
        if j != 0 and norm(diff) < 1e-12:
            tfi_useable = True
            print("NLS Covergence required %d iters." % (j+1))
            break
        elif j+1 == iters:
            tfi_useable = diff < 1e-6
            print("NLS Failed to converge after %d iters!!!" % (j+1))
            #print(diff)
            break

    Tfi = np.vstack((np.hstack((C, t)), np.array([[0, 0, 0, 1]])))

    # Check for correct outputs...
    correct = isinstance(Tfi, np.ndarray) and Tfi.shape == (4, 4)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return tfi_useable, Tfi

def dcm_jacob_rpy(C):
     # Rotation - convenient!
    cp = np.sqrt(1 - C[2, 0]*C[2, 0])
    cy = C[0, 0]/cp
    sy = C[1, 0]/cp

    dRdr = C@np.array([[ 0,   0,   0],
                       [ 0,   0,  -1],
                       [ 0,   1,   0]])

    dRdp = np.array([[ 0,    0, cy],
                     [ 0,    0, sy],
                     [-cy, -sy,  0]])@C

    dRdy = np.array([[ 0,  -1,  0],
                     [ 1,   0,  0],
                     [ 0,   0,  0]])@C
    
    return dRdr, dRdp, dRdy