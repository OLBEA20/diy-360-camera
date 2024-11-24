import cv2
import numpy as np
import glob
import os

def verify_image_quality(img, min_size=(800, 600), max_size=(4000, 3000)):
    """
    Verify that the image meets basic quality requirements
    """
    h, w = img.shape[:2]
    if w < min_size[0] or h < min_size[1]:
        return False, "Image is too small"
    if w > max_size[0] or h > max_size[1]:
        return False, "Image is too large"
    if len(img.shape) != 3:
        return False, "Image is not in color format"
    return True, "OK"

def check_chessboard_coverage(corners, img_shape, min_coverage=0.12):
    """
    Check if the chessboard covers enough of the image
    """
    x_min, y_min = corners.reshape(-1, 2).min(axis=0)
    x_max, y_max = corners.reshape(-1, 2).max(axis=0)
    
    board_area = (x_max - x_min) * (y_max - y_min)
    image_area = img_shape[0] * img_shape[1]
    
    coverage = board_area / image_area
    print(f"Coverage: {coverage}")
    return coverage >= min_coverage

def calibrate_fisheye_camera(images_path, board_size=(9, 6), save_path='calibration_params.npz', debug_dir=None):
    """
    Enhanced fisheye camera calibration with additional checks and debug output
    
    Args:
        images_path: Path pattern to calibration images
        board_size: Tuple of (columns, rows) of internal corners
        save_path: Path to save calibration parameters
        debug_dir: Directory to save debug images (None to skip)
    """
    if debug_dir and not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    # Prepare object points
    objp = np.zeros((1, board_size[0] * board_size[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    
    objpoints = []
    imgpoints = []
    
    images = glob.glob(images_path)
    if not images:
        raise ValueError(f"No images found at {images_path}")
    
    successful_images = 0
    skipped_images = []
    
    # Process each calibration image
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to read image: {fname}")
            continue
        
        # Verify image quality
        quality_ok, quality_msg = verify_image_quality(img)
        if not quality_ok:
            print(f"Skipping {fname}: {quality_msg}")
            skipped_images.append((fname, quality_msg))
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Improve contrast
        gray = cv2.equalizeHist(gray)
        
        # Find the chessboard corners with different flags
        for flags in [
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK,
        ]:
            ret, corners = cv2.findChessboardCorners(gray, board_size, flags)
            if ret:
                break
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
            corners = cv2.cornerSubPix(gray, corners, (5,5), (-1,-1), criteria)
            
            # Check board coverage
            if not check_chessboard_coverage(corners, gray.shape):
                #print(f"Skipping {fname}: Chessboard doesn't cover enough of the image")
                skipped_images.append((fname, "Insufficient coverage"))
                continue
            
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners
            img_corners = img.copy()
            cv2.drawChessboardCorners(img_corners, board_size, corners, ret)
            
            if debug_dir:
                debug_path = os.path.join(debug_dir, f'corners_{idx}.jpg')
                cv2.imwrite(debug_path, img_corners)
            
            cv2.imshow('Chessboard Detection', cv2.resize(img_corners, (800, 600)))
            key = cv2.waitKey(500)
            if key == 27:
                cv2.destroyAllWindows()
                break
            
            successful_images += 1
            print(f"Successfully processed {fname} ({successful_images}/{len(images)})")
        else:
            skipped_images.append((fname, "No corners detected"))
    
    cv2.destroyAllWindows()
    
    if successful_images < 4:
        raise ValueError(f"Not enough successful calibration images. Found {successful_images}, need at least 4.")
    
    print(f"\nStarting calibration with {successful_images} images...")
    
    # Get image size from last successful image
    img_shape = gray.shape[::-1]
    
    # Initialize matrices
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
    
    try:
        # Calibrate camera with relaxed error criteria
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            #cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
        
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            img_shape,
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            criteria
        )
        
        print("\nCalibration Results:")
        print("RMS error:", rms)
        print("\nCamera matrix (K):")
        print(K)
        print("\nDistortion coefficients (D):")
        print(D)
        
        # Save calibration parameters
        np.savez(save_path, K=K, D=D, rms=rms)
        print(f"\nCalibration parameters saved to {save_path}")
        
        # Print summary of skipped images
        if skipped_images:
            print("\nSkipped images summary:")
            for fname, reason in skipped_images:
                print(f"- {os.path.basename(fname)}: {reason}")
        
        return K, D, rms
        
    except cv2.error as e:
        raise RuntimeError(f"Calibration failed: {str(e)}\nTry using more images or images with better chessboard visibility")

def test_calibration(image_path, K, D, output_path=None):
    """
    Test calibration parameters on an image and optionally save the result
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    if output_path:
        cv2.imwrite(output_path, undistorted_img)
    
    # Display results side by side
    comparison = np.hstack((img, undistorted_img))
    cv2.imshow('Original | Undistorted', cv2.resize(comparison, (1600, 600)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Create debug directory
        debug_dir = 'calibration_debug'
        
        # Calibrate camera
        K, D, rms = calibrate_fisheye_camera(
            images_path='calib/*.jpeg',
            board_size=(8, 6),
            save_path='fisheye_calibration.npz',
            debug_dir=debug_dir
        )
        
        # Test calibration
        test_calibration(
            'data/cam1/test00200.jpeg',
            K, D,
            output_path='undistorted_test.jpg'
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")