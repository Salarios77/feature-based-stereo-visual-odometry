from pathlib import Path
import argparse
import sys
from support.test_docker import test_docker
#----- ADD YOUR IMPORTS HERE IF NEEDED -----
import os
import numpy as np
from numpy.linalg import inv, norm
import cv2
import matplotlib.pyplot as plt
from support.triangulate import triangulate
from support.estimate_motion_ls import estimate_motion_ls
from support.estimate_motion_ils import estimate_motion_ils
from support.convert_time import convert_date_string_to_unix_seconds
import time


def intrinsics_to_K_D(fx, fy, cx, cy, k1, k2, p1, p2, k3):
    '''
    Given the a camera's intrinsic parameters, return the camera matrix (K) and 
    the distortion coefficients (D) following OpenCV notation
    '''
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    D = np.array([k1, k2, p1, p2, k3])
    return K,D


def transform_params_to_R_T(tx, ty, tz, qx, qy, qz, qw):
    '''
    Given the a camera's transformation (translation and unit quaternion) parameters 
    with respect to some reference frame, return the corresponding rotation (R) and 
    translation (T) matrices
    '''
    T = np.array([tx, ty, tz])
    qv = np.array([qx, qy, qz]).reshape(3,1)
    qv_cross = np.array([[0, -qz, qy],
                         [qz, 0, -qx],
                         [-qy, qx, 0]])
    R = (qw**2 - norm(qv)**2)*np.eye(3) - 2*qw*qv_cross + 2*qv@qv.T 
    return R,T          


def run_project(input_dir, output_dir):
    """
    Main entry point for your project code. 
    
    DO NOT MODIFY THE SIGNATURE OF THIS FUNCTION.
    """
    #---- FILL ME IN ----

    #=========================== Reference Trajectory ==========================

    # Parse reference trajectory
    t_w0_ref = []  # 3xn translations
    unix_times_ref = []
    with open(os.path.join(input_dir, 'run1_base_hr/global-pose-utm.txt')) as ref_pose_file:
        lines = ref_pose_file.readlines()[1:]
        for i, line in enumerate(lines):
            unix_times_ref.append(float(line.split(',')[0]))
            # Store initial rotation and translation
            if i == 0:
                init_transf_params = [float(param) for param in line.split(',')[1:]]
                R_init, t_init = transform_params_to_R_T(*init_transf_params)
            # Get translation for current time step and subtract the initial
            # translation to start the coordinates at the origin (0,0,0)
            xyz = [float(coord) - t_init[j] for j,coord in enumerate(line.split(',')[1:4])]
            t_w0_ref.append(xyz)
    unix_times_ref = np.asarray(unix_times_ref)
    t_w0_ref = np.asarray(t_w0_ref).T

    # Get initial homogeneous pose (modified to start at the origin (0,0,0))
    T_init = np.eye(4)
    T_init[0:3,0:3] = R_init

    #============================== Stereo Images ==============================

    # Get the paths of all the stereo image pairs from omni cameras 0 (top) and 
    # 1 (bottom), sorted chronologically
    omni0_base_path = os.path.join(input_dir, 'run1_base_hr/omni_image0')
    omni1_base_path = os.path.join(input_dir, 'run1_base_hr/omni_image1')
    omni0_img_paths = sorted(os.listdir(omni0_base_path))
    omni1_img_paths = sorted(os.listdir(omni1_base_path))
    print('# Omni_0 images: {}, # Omni_1 images: {}\n'.format(len(omni0_img_paths), 
        len(omni1_img_paths)))

    #============================ Camera Intrinsics ============================

    # Parse camera intrinsic parameters
    with open(os.path.join(input_dir, 'cameras_intrinsics.txt')) as intrinsics_file:
        [omni0_intrinsics, omni1_intrinsics] = intrinsics_file.readlines()[6:8]
        
        # params{0,1} contain fx [px], fy [px], cx [px], cy [px], k1, k2, p1, p2, k3
        intrins_params0 = [float(param) for param in omni0_intrinsics.split(',')[1:]]
        intrins_params1 = [float(param) for param in omni1_intrinsics.split(',')[1:]]

    # Get camera matrix and distortion coefficient array camera 0 (top) and 1 (bottom)
    K0, D0 = intrinsics_to_K_D(*intrins_params0)
    K1, D1 = intrinsics_to_K_D(*intrins_params1)

    #========================= Camera Transformations ==========================

    # Parse camera transformations relative to omnidirectional sensor
    with open(os.path.join(input_dir, 'rover_transforms.txt')) as transforms_file:
        [omni0_transform, omni1_transform] = transforms_file.readlines()[5:7]

        # params{0,1} contain trans_x [m], trans_y [m], trans_z [m], quat_x, quat_y, quat_z, quat_w
        transf_params0 = [float(param) for param in omni0_transform.split(',')[2:]]
        transf_params1 = [float(param) for param in omni1_transform.split(',')[2:]]

    # Get camera rotation and translation matrices relative to omnidirectional sensor (ref)
    R_ref0, t_ref0 = transform_params_to_R_T(*transf_params0)
    R_ref1, t_ref1 = transform_params_to_R_T(*transf_params1)

    # Get transformation from coordinates of camera 0 to camera 1 (transformation
    # of frame 0 with respect to frame 1)
    R_10 = inv(R_ref1) @ R_ref0
    t_10 = inv(R_ref1) @ (t_ref0 - t_ref1)
    T_10 = np.eye(4)
    T_10[0:3,0:3] = R_10
    T_10[0:3,3] = t_10
    T_01 = inv(T_10)

    #================== Retification & Undistortion Mappings ===================

    # All stereo images have fixed height and width
    width = 752
    height = 480

    # Compute rectification transforms for each stereo camera
    # R{0,1} - 3x3 rectification transform bringing unrecritifed camera coord. sys to rectified camera coord. sys
    # P{0,1} - 3x4 projection matrices from rectified camera coord. sys to rectified image coord. sys
    # roi_{top,bottom} - rectangles (ymin,xmin,ymin,xmax) inside rectified images where all pixels are valid
    R0, R1, P0, P1, _, roi_top, roi_bot = cv2.stereoRectify(K0, D0, K1, D1, (height, width), R_10, t_10)
    
    # Compute undistortion and rectification transformation maps
    topMapX, topMapY = cv2.initUndistortRectifyMap(K0, D0, R0, P0, (width, height), cv2.CV_32FC1)
    botMapX, botMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)

    #=================== Homogeneous Retification Transforms ===================
    
    # Extract 'rectified' camera intrinsic matrices
    K0_rect = P0[0:3,0:3]
    K1_rect = P1[0:3,0:3]
    
    # Extract camera translation (for projection from unrectified camera 
    # coords. to rectified image coords.)
    t0_rect = (P0[0:3, 3])
    t1_rect = (P1[0:3, 3])

    # Rotation and translation from unrectified camera coords to rectified
    # camera coords
    T_rect0 = np.eye(4)
    T_rect0[0:3,0:3] = R0
    T_rect0[0:3,3] = inv(K0_rect)@t0_rect
    T_rect1 = np.eye(4)
    T_rect1[0:3,0:3] = R1
    T_rect1[0:3,3] = inv(K1_rect)@t1_rect
    
    T_0rect = inv(T_rect0)
    T_1rect = inv(T_rect1)

    #================================ Main loop ================================

    # Set feature detector
    FEATURE_DETECTOR = 'HARRIS'
    assert (FEATURE_DETECTOR in ['HARRIS', 'SIFT', 'BRISK'])

    # Number of iterations for which to run VO
    MAX_ITER = 350

    # Initialize homogeneous pose matrix for top camera in world frame
    # as the same initial pose as the reference trajectory
    T_w0 = T_init

    # Initialize homogenous pose matrix for 6-DOF motion of cameras body for 
    # consecutive time steps which will be estimated via non-linear least squares
    T_fi = np.eye(4)

    # Initialize a np array for storing the translation from the world
    # frame to the top camera frame, to later visualize the 
    # trajectory of the robot
    t_w0 = np.zeros((3,MAX_ITER))

    # Initialize array for storing unix times
    unix_times = np.zeros((MAX_ITER))

    # Counter for storing total time spent from detection to solving NLS
    total_feature_usage_time = 0

    print('-->Starting Visual Odometry (with {} features)...'.format(FEATURE_DETECTOR))

    # 'Previous iteration' lists of top image features (after stereo matching), 
    # landmarks, and landmark covariance matrices, these will be defined 
    # during the visual odometry (VO).
    p_top_prev = None  # 2d keypoints in top image
    p_top_kp_prev = None  # 2d keypoints in top image (KeyPoint wrapper type)
    p_top_descriptors_prev = None # descriptors of keypoints in top image
    top_rectified_prev = None  # top image
    Pw_list_prev = None  # 3D landmark points for top image
    S_list_prev = None  # landmark covariance matrices for top image

    # Iterate through pairs of omni0 and omni1 images at each time step
    for t, (omni0_rel_path, omni1_rel_path) in enumerate(zip(omni0_img_paths, omni1_img_paths)):

        if t == MAX_ITER:
            break

        # Time in 'YYYY_MM_DD_hh_mm_ss_microsec' format
        date_and_time = ('_'.join(omni0_rel_path.split('_')[1:])).split('.')[0]

        # Time in Unix format
        unix_time = convert_date_string_to_unix_seconds(date_and_time)
        unix_times[t] = unix_time
        
        print('\nTime step {}/{}'.format(t,MAX_ITER-1))

        #========================= Update Pose Matrices ========================

        # Update pose of top camera in world frame
        T_w0 = T_w0 @ inv(T_fi)

        # Store the translation from world frame to top camera 
        t_w0[:,t] = T_w0[0:3, 3]

        # Compute homogenous pose matrix for bottom camera in world frame
        T_w1 = T_w0 @ T_01

        # Compute homogeneous poses, taking into account rectification, for
        # top camera in world frame and bottom camera in world frame
        T_wtop = T_w0 @ T_0rect
        T_wbot = T_w1 @ T_1rect

        #========================= Load Stereo Images =========================

        omni0_path = os.path.join(omni0_base_path, omni0_rel_path)
        omni1_path = os.path.join(omni1_base_path, omni1_rel_path)

        # Read images
        I0 = cv2.imread(omni0_path)
        I1 = cv2.imread(omni1_path)

        # Convert to grayscale
        I0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
        I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

        #========================= Undistort & Rectify =========================

        # Map the top/bottom (omni0/omni1) images to their undistorted/rectified versions
        top_rectified = cv2.remap(I0, topMapX, topMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        bot_rectified = cv2.remap(I1, botMapX, botMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # Plot example of top/bottom stereo images before and after undistortion/rectification
        if t == 0:
            plt.figure(figsize=(8,4))

            # Top images
            plt.subplot(2,2,1)
            plt.imshow(I0, cmap='gray') 
            plt.title ('Raw Top/Bottom Images')
            plt.subplot(2,2,2)
            plt.imshow(top_rectified, cmap='gray')
            plt.title ('Undistorted & Rectified Top/Bottom Images')
            # Roi
            plt.vlines(x=(roi_top[1], roi_top[3]), ymin=roi_top[0], ymax=roi_top[2])
            plt.hlines(y=(roi_top[0], roi_top[2]), xmin=roi_top[1], xmax=roi_top[3])
            # Vertical lines to show rectified
            for i in range (1, 12):
                plt.vlines(x=(roi_top[1] + i*(roi_top[3]-roi_top[1])/12), 
                           ymin=roi_top[0], ymax=roi_top[2], colors='r')

            # Bottom images
            plt.subplot(2,2,3)
            plt.imshow(I1, cmap='gray') 
            plt.subplot(2,2,4)
            plt.imshow(bot_rectified, cmap='gray')
            # Roi
            plt.vlines(x=(roi_bot[1], roi_bot[3]), ymin=roi_bot[0], ymax=roi_bot[2])
            plt.hlines(y=(roi_bot[0], roi_bot[2]), xmin=roi_bot[1], xmax=roi_bot[3])
            # Vertical lines to show rectified
            for i in range (1, 12):
                plt.vlines(x=(roi_top[1] + i*(roi_top[3]-roi_top[1])/12), 
                           ymin=roi_top[0], ymax=roi_top[2], colors='r')

            plt.savefig(os.path.join(output_dir, 'preprocessing_ex.png'))

        #========================== Feature Detection ==========================

        # Record time
        feature_usage_t0 = time.time()

        # Extract keypoints using the chosen detector for the top/bottom images
        # {top/bot}_kp - ndarray with first row = x, second row = y
        if FEATURE_DETECTOR == 'HARRIS':
            # Get Harris response map
            top_dst = cv2.cornerHarris(top_rectified, blockSize=2, ksize=3, k=0.04)
            bot_dst = cv2.cornerHarris(bot_rectified, blockSize=2, ksize=3, k=0.04)
            # Use a fixed threshold to select features from response map
            threshold = 0.05
            top_kp = np.asarray(np.where(top_dst > threshold*top_dst.max()))
            bot_kp = np.asarray(np.where(bot_dst > threshold*bot_dst.max()))
            top_kp[[0,1]] = top_kp[[1,0]]  # swap x,y
            bot_kp[[0,1]] = bot_kp[[1,0]]  # swap x,y
            
            # The top image's 'descriptors' for HARRIS will be generated 
            # during stereo matching as the flattened surrounding windows 
            # of the keypoints 
            top_descriptors = []

        elif FEATURE_DETECTOR == 'SIFT':
            sift = cv2.SIFT_create()
            # Find keypoints and corresponding descriptors 
            top_kp_wrapped, top_descriptors = sift.detectAndCompute(top_rectified, None)
            bot_kp_wrapped, _ = sift.detectAndCompute(bot_rectified, None)
            # Extract x,y coordinates from cv2.KeyPoint wrappers
            top_kp = np.asarray([np.round(kp.pt) for kp in top_kp_wrapped], dtype=int).T
            bot_kp = np.asarray([np.round(kp.pt) for kp in bot_kp_wrapped], dtype=int).T
        elif FEATURE_DETECTOR == 'BRISK':
            brisk = cv2.BRISK_create()
            # Find keypoints and corresponding descriptors 
            top_kp_wrapped, top_descriptors = brisk.detectAndCompute(top_rectified, None)
            bot_kp_wrapped, _ = brisk.detectAndCompute(bot_rectified, None)
            # Extract x,y coordinates from cv2.KeyPoint wrappers
            top_kp = np.asarray([np.round(kp.pt) for kp in top_kp_wrapped], dtype=int).T
            bot_kp = np.asarray([np.round(kp.pt) for kp in bot_kp_wrapped], dtype=int).T

        print('Detected [{},{}] features in [top,bot] images'.format(top_kp.shape[1], 
                                                                     bot_kp.shape[1]))

        # Plot example of feature detection
        if t==0:
            plt.figure()
            plt.imshow(top_rectified, cmap='gray')
            plt.title('Feature Detection on Rectified Top Image Using {}'.format(FEATURE_DETECTOR))
            plt.plot(top_kp[0, :], top_kp[1, :], 'o', c = 'r', markersize = 2)
            plt.savefig(os.path.join(output_dir, 'keypoints_{}_ex.png'.format(FEATURE_DETECTOR)))

        #=========================== Stereo Matching ===========================

        p_bot, indices_with_matches, win_sz = stereo_match(top_rectified, bot_rectified, 
                 roi_top, roi_bot, top_kp, top_descriptors, FEATURE_DETECTOR)

        # Extract the keypoints and descriptors which had matches
        p_top = (top_kp[:,indices_with_matches]).T
        if FEATURE_DETECTOR in ['SIFT', 'BRISK']:
            p_top_kp = [top_kp_wrapped[idx] for idx in indices_with_matches]
            p_top_descriptors = top_descriptors[indices_with_matches, :]
        else:
            # Convert to np array for HARRIS
            p_top_kp = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), _size=win_sz) for pt in p_top]
            p_top_descriptors = np.float32(np.asarray(top_descriptors))
            
            # Edge case - ensure there are more than 0 matches, rarely ever occurs
            if len(p_top_descriptors) == 0:
                T_fi = np.eye(4)
                continue

        print('{} Stereo matches found'.format(len(p_top)))

        # Plot example of stereo matching
        if t==0:
            plt.figure()
            plt.subplot(2,1,1)
            plt.imshow(top_rectified, cmap='gray')
            plt.title('Stereo Matching of {} Features'.format(FEATURE_DETECTOR))
            plt.plot(top_kp[0, :], top_kp[1, :], 'o', c = 'r', markersize = 2)
            plt.plot(p_top[:, 0], p_top[:, 1], 'o', c = 'g', markersize = 1)
            plt.subplot(2,1,2)
            plt.imshow(bot_rectified, cmap='gray')
            plt.plot(bot_kp[0, :], bot_kp[1, :], 'o', c = 'r', markersize = 2)
            plt.plot(p_bot[:, 0], p_bot[:, 1], 'o', c = 'g', markersize = 1)
            plt.savefig(os.path.join(output_dir, 'stereo_matching_{}_ex.png'.format(FEATURE_DETECTOR)))

        #============================ Triangulation ============================

        # Assume that image plane uncertainties are identity
        Sl = np.eye(2)
        Sr = np.eye(2)

        # To store estimated 3D landmark positions in the world frame
        Pw_list = np.zeros((3, len(p_top)))  # 3xn
        # To store covariance matrices for estimated 3D point
        S_list = np.zeros((3, 3, len(p_top)))  # 3x3xn

        # Triangulate 3D point positions from camera projections for stereo matches
        for i, (p_top_i, p_bot_i) in enumerate(zip(p_top, p_bot)):
            _, _, Pw, S = triangulate(K0_rect, K1_rect, T_wtop, T_wbot, 
                            p_top_i.reshape(2,1), p_bot_i.reshape(2,1), Sl, Sr)
            Pw_list[:,i] = Pw.squeeze()
            S_list[:,:,i] = S

        # Perform feature tracking, and motion estimation after the first iteration
        if t > 0:
            #========================= Feature Tracking ========================

            # create BFMatcher object depending on the feature detector
            if FEATURE_DETECTOR == 'HARRIS':
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            elif FEATURE_DETECTOR == 'SIFT':
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            elif FEATURE_DETECTOR == 'BRISK':
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors
            matches = bf.match(p_top_descriptors_prev,p_top_descriptors)
            print('Tracked {} features from t-1 to t'.format(len(matches)))

            # Plot example of feature tracking
            if t == 1:
                img3 = cv2.drawMatches(top_rectified_prev, p_top_kp_prev, top_rectified, 
                                       p_top_kp, matches, None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                plt.figure()
                plt.imshow(img3)
                plt.title('{} Features Tracked From Time t-1 to Time t (Using Top Stereo Images)'.format(FEATURE_DETECTOR))
                plt.savefig(os.path.join(output_dir, 'feature_tracking_{}_ex.png'.format(FEATURE_DETECTOR)))

            # Extract the indices of the p_top_descriptors_prev and 
            # p_top_descriptors from the matching
            match_indices_prev = [match.queryIdx for match in matches]
            match_indices_cur = [match.trainIdx for match in matches]

            # Extract prev/cur 3D landmark points and covariance matrices
            # using the indices from the matching
            Pi = Pw_list_prev[:, match_indices_prev]
            Pf = Pw_list[:, match_indices_cur]
            Si = S_list_prev[:, :, match_indices_prev]
            Sf = S_list[:, :, match_indices_cur]

            #=============== Matrix Weighted Point Cloud Alignment =============

            # Run non-linear least squares to estimate the 6-DOF motion of 
            # the body, using an initialization from scalar weighted point
            # cloud alignement
            tfi_useable, T_fi = estimate_motion_ils(Pi, Pf, Si, Sf, iters=100)

            # If nonlinear least squares did not converge, do not use the 
            # resulting T_fi matrix, force it to be identity (the algorithm
            # converges on nearly all iterations)
            if not tfi_useable:
                T_fi = np.eye(4)

        # Update the 'previous iteration' lists of top image features (after 
        # stereo matching), landmarks, and landmark covariance matrices
        p_top_prev = p_top
        p_top_kp_prev = p_top_kp
        p_top_descriptors_prev = p_top_descriptors
        top_rectified_prev = top_rectified
        Pw_list_prev = Pw_list
        S_list_prev = S_list

        # Record time
        total_feature_usage_time += time.time() - feature_usage_t0

    #===================== Plot Trajectory and Compute MSE =====================

    plot_and_evaluate(MAX_ITER, total_feature_usage_time, unix_times, unix_times_ref,
                      t_w0, t_w0_ref, FEATURE_DETECTOR, output_dir)

    #--------------------


def stereo_match(top_rectified, bot_rectified, roi_top, roi_bot, top_kp, 
                 top_descriptors, FEATURE_DETECTOR):
    ''' Perform stereo matching on top and bottom rectified images using a 
    1D vertical search.
    top_rectified - nxm top image
    bot_rectified - nxm bottom image
    roi_top - region of interest in top image, list: (ymin,xmin,ymin,xmax)
    roi_bot - region of interest in bottom image, list; (ymin,xmin,ymin,xmax)
    top_kp - list of k keypoints - 2xk ndarray with first row = x, second row = y
    top_descriptors - kx128 nd array of descriptors
    FEATURE_DETECTOR - 'HARRIS', 'SIFT', or 'BRISK'

    Returns:
    p_bot - nx2 of x,y pixel coordinates for correspondences in bottom image
    indices_with_matches - list of indices of top_kp which had matches
    win_sz - sliding window size
    '''

    # List to store which indices of top_kp had matches
    indices_with_matches = []  

    # List for storing 2d image point correspondences in bottom image
    p_bot = []

    # Define an odd window size, with the center point 
    # used to find per-pixel correspondences between top and bottom images.
    win_sz = 13

    # Pad the top and bottom images with zeros at the borders (amount of 
    # padding is window size divided by 2) to be able to perform SAD
    # with center of window at all pixels in original images
    m = top_rectified.shape[0]
    n = bot_rectified.shape[1]
    pad_sz = win_sz // 2
    It_padded = np.zeros((m + 2*pad_sz, n + 2*pad_sz))
    Ib_padded = np.zeros((m + 2*pad_sz, n + 2*pad_sz))
    It_padded[pad_sz:m+pad_sz, pad_sz:n+pad_sz] = top_rectified
    Ib_padded[pad_sz:m+pad_sz, pad_sz:n+pad_sz] = bot_rectified

    # Valid region in top image
    y_min,x_min,y_max,x_max = roi_top

    # Iterate over keypoints in top image
    for i, kp in enumerate(top_kp.T):  
        x = kp[0]
        y = kp[1]

        # Check if in valid region
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            continue

        # Define top image window
        It_padded_window = It_padded[y:y+win_sz, x:x+win_sz]

        # Initialize lowest SAD similarity for current pixel in top image
        lowestSAD = float("inf")
        y2_best = None

        # Iterate over the column at x=x in the bottom image from bottom to 
        # top, starting from the ymax of the bottom image's ROI to the ymin
        for y2 in range(roi_bot[2], roi_bot[0], -1):
            # Compute local similarity by taking the absolute difference
            # of the intensities in the top image window and 
            # bottom image window, and summing them.
            Ib_padded_window = Ib_padded[y2:y2+win_sz, x:x+win_sz]
            diff = np.abs(It_padded_window - Ib_padded_window)
            SAD = np.sum(diff)

            # Update disparity if current SAD measure is the lowest encountered
            if SAD < lowestSAD:
                lowestSAD = SAD
                y2_best = y2

        # Reject y2 coordinates which are the same as the y coordinate
        # of the top image's keypoint (this is an edge case which 
        # rarely occurs but needs to be avoided so triangulation
        # doesnt produce parallel rays)
        if y2_best is not None and y2_best != y:
            indices_with_matches.append(i)
            p_bot.append([x,y2_best])
            
            # The top image's 'descriptors' for HARRIS are generated 
            # as the flattened surrounding windows of the keypoints 
            if FEATURE_DETECTOR == 'HARRIS':
                top_descriptors.append(It_padded_window.reshape(win_sz*win_sz))

    p_bot = np.asarray(p_bot)

    return p_bot, indices_with_matches, win_sz


def plot_and_evaluate(MAX_ITER, total_feature_usage_time, unix_times, unix_times_ref,
                      t_w0, t_w0_ref, FEATURE_DETECTOR, output_dir):
    ''' Plot estimated and reference trajectory, compute MSE, and print statistics
    MAX_ITER - number of iterations run
    total_feature_usage_time - sum of time spent on steps from detection to NLS
    unix_times - list of unix times for each time step of VO
    unix_times_ref - list of unix times of the reference trajectory
    t_w0 - 3xMAX_ITER translations (positions) from VO
    t_w0_ref - 3xm translations (positions) from reference trajectory
    FEATURE_DETECTOR - 'HARRIS', 'SIFT', or 'BRISK'
    output_dir - output directory
    '''
    
    # Print total average time spent using features
    print('\nAverage VO time/iter (not including pre-processing): {:.2f} s\n'.format(
        total_feature_usage_time / MAX_ITER))

    # Equate the start times of the VO and reference trajectories
    time0 = unix_times[0]
    unix_times -= time0
    unix_times_ref -= time0

    # Find the final index of the reference trajectory that should be plotted until
    last_idx_ref = np.argmax(unix_times_ref > unix_times[MAX_ITER-1]) + 1

    # Find the first index of the VO trajectory that should be plotted (there is
    # a very slight offset in start time)
    first_index_VO = np.argmax(unix_times > unix_times_ref[0]) - 1

    # Plot x vs. y, t vs. x, t vs. y, t. vs z
    plt.figure(figsize=(19,4))
    plt.subplot(1,4,1)
    plt.plot(t_w0[0, first_index_VO:], t_w0[1, first_index_VO:])
    plt.plot(t_w0_ref[0, :last_idx_ref], t_w0_ref[1, :last_idx_ref])  # reference trajectory
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend(('VO', 'Reference'))
    plt.subplot(1,4,2)
    plt.plot(unix_times[first_index_VO:], t_w0[0, first_index_VO:])
    plt.plot(unix_times_ref[:last_idx_ref], t_w0_ref[0, :last_idx_ref])  # reference trajectory
    plt.xlabel('t (s)')
    plt.ylabel('x (m)')
    plt.legend(('VO', 'Reference'))
    plt.subplot(1,4,3)
    plt.plot(unix_times[first_index_VO:], t_w0[1, first_index_VO:])
    plt.plot(unix_times_ref[:last_idx_ref], t_w0_ref[1, :last_idx_ref])  # reference trajectory
    plt.xlabel('t (s)')
    plt.ylabel('y (m)')
    plt.legend(('VO', 'Reference'))
    plt.subplot(1,4,4)
    plt.plot(unix_times[first_index_VO:], t_w0[2, first_index_VO:])
    plt.plot(unix_times_ref[:last_idx_ref], t_w0_ref[2, :last_idx_ref])  # reference trajectory
    plt.xlabel('t (s)')
    plt.ylabel('z (m)')
    plt.legend(('VO', 'Reference'))
    plt.suptitle('VO Trajectory (With {} Features) vs. Reference Trajectory'.format(FEATURE_DETECTOR))
    plt.savefig(os.path.join(output_dir, 'trajectory_{}.png'.format(FEATURE_DETECTOR)))      

    # Measure MSE - the number of outputted VO points is roughly double the 
    # number of reference points so take every other point from the VO, trim
    # the last couple points from the reference values to ensure
    # the number of points in the two arrays is equal, and compute MSE
    t_w0_downsampled = t_w0[:, first_index_VO::2]
    t_w0_ref_trimmed = t_w0_ref[:, 0:t_w0_downsampled.shape[1]]
    MSE = (np.square(t_w0_ref_trimmed - t_w0_downsampled)).mean(axis=1)
    print('MSE x: {:.2f} m^2'.format(MSE[0]))
    print('MSE y: {:.2f} m^2'.format(MSE[1]))
    print('MSE z: {:.2f} m^2\n'.format(MSE[2]))
    
    with open(os.path.join(output_dir, 'MSE_{}.txt'.format(FEATURE_DETECTOR)), "w") as f:
        f.write('Average VO time/iter (not including pre-processing): {:.2f} s\n'.format(
        total_feature_usage_time / MAX_ITER))
        f.write('MSE x: {:.2f} m^2\n'.format(MSE[0]))
        f.write('MSE y: {:.2f} m^2\n'.format(MSE[1]))
        f.write('MSE z: {:.2f} m^2\n'.format(MSE[2]))


# Command Line Arguments
parser = argparse.ArgumentParser(description='ROB501 Final Project.')
parser.add_argument('--input_dir', dest='input_dir', type=str, default="./input",
                    help='Input Directory that contains all required rover data')
parser.add_argument('--output_dir', dest='output_dir', type=str, default="./output",
                    help='Output directory where all outputs will be stored.')


if __name__ == "__main__":
    
    # Parse command line arguments
    args = parser.parse_args()

    # Uncomment this line if you wish to test your docker setup
    #test_docker(Path(args.input_dir), Path(args.output_dir))

    print('\n-->Starting Execution...')

    # Run the project code
    run_project(args.input_dir, args.output_dir)